"""Tests for camera gesture recognition, cross-camera fusion, and the air mouse.

The fusion tests carry the most weight. Gesture *recognition* failing is obvious — you
wave at the camera and nothing happens. Fusion failing is not: it deletes a gesture
that both the recogniser and the preview agree it can see, which is what made fist and
pinch look dead while cursor control kept working.

Nothing here needs a camera, MediaPipe, or a window server.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from arc.vision.hands.fusion import (
    CONF_AGREED,
    CONF_FRONT,
    CONF_SIDE_ONLY,
    DEFAULT_MIN_CONF,
    Depth,
    Fuser,
    Gate,
    score,
)
from arc.vision.hands.gestures import PINCH_DISTANCE, recognize, two_hand_gap
from arc.vision.hands.pointer import PointerController, map_to_screen

# ── Building hands to recognise ─────────────────────────────────────────────────
#
# Landmarks are laid out in the same normalised space MediaPipe uses: the wrist at
# the origin and the hand extending upward, which on an image means *decreasing* y.


def _hand(
    *,
    curl: tuple[bool, bool, bool, bool] = (False, False, False, False),
    thumb_to_index: float = 1.0,
) -> list[tuple[float, float, float]]:
    """A synthetic hand: four fingers each extended or curled, thumb placed by distance.

    Finger tips sit at 2.0 hand-lengths from the wrist when extended and 0.5 when
    curled; PIP joints sit at 1.0 throughout. That straddles the ratio thresholds in
    both directions without encoding them.
    """
    landmarks = [(0.0, 0.0, 0.0)] * 21
    points = dict.fromkeys(range(21), (0.0, 0.0, 0.0))
    points[0] = (0.0, 0.0, 0.0)  # wrist
    points[9] = (0.0, -1.0, 0.0)  # middle MCP -> hand_scale == 1.0
    points[5] = (-0.4, -1.0, 0.0)  # index MCP
    points[17] = (0.4, -1.0, 0.0)  # pinky MCP

    for finger, (tip, pip) in enumerate(((8, 6), (12, 10), (16, 14), (20, 18))):
        x = -0.4 + finger * 0.27
        points[pip] = (x, -1.0, 0.0)
        points[tip] = (x, -0.5, 0.0) if curl[finger] else (x, -2.0, 0.0)

    # Thumb tip placed a chosen distance from the index tip, along x.
    index_tip = points[8]
    points[4] = (index_tip[0] - thumb_to_index, index_tip[1], 0.0)
    points[3] = (index_tip[0] - thumb_to_index - 0.2, index_tip[1], 0.0)

    landmarks = [points[i] for i in range(21)]
    return landmarks


def test_an_open_hand_is_open() -> None:
    assert recognize(_hand())["gesture"] == "open"


def test_a_closed_hand_is_a_fist() -> None:
    assert recognize(_hand(curl=(True, True, True, True)))["gesture"] == "fist"


def test_mediapipes_own_label_can_settle_a_fist() -> None:
    """The bundled model already classifies a fist and the original port discarded it.

    A clench slightly off-angle for the landmark ratios is still a clench, and the
    two tests failing in different situations is the point of consulting both.
    """
    half_closed = _hand(curl=(True, True, False, False))
    assert recognize(half_closed)["gesture"] != "fist"
    assert recognize(half_closed, canned="Closed_Fist")["gesture"] == "fist"


def test_pinch_is_thumb_and_index_meeting() -> None:
    hand = _hand(curl=(False, False, True, True), thumb_to_index=PINCH_DISTANCE / 2)
    assert recognize(hand)["gesture"] == "pinch"


def test_a_fist_is_not_read_as_a_pinch() -> None:
    """A closed fist also brings thumb and index together — fist has to win.

    Getting this backwards means every window move turns into a resize.
    """
    fist = _hand(curl=(True, True, True, True), thumb_to_index=0.05)
    assert recognize(fist)["gesture"] == "fist"


def test_a_pinch_is_not_read_as_a_fist() -> None:
    """Pinching curls the spare fingers, which is why the fist test stays strict."""
    pinch = _hand(curl=(False, False, True, True), thumb_to_index=0.1)
    assert recognize(pinch)["gesture"] == "pinch"


def test_a_pinch_with_the_middle_finger_out_is_still_a_pinch() -> None:
    """Pinching rarely curls the middle finger, so this pose is the common case.

    Before pinch was ordered ahead of the two-finger test it read as the cursor pose,
    which sent a two-handed resize to the mouse instead of to the window.
    """
    pinch = _hand(curl=(False, False, True, True), thumb_to_index=0.1)
    assert recognize(pinch)["gesture"] == "pinch"


def test_the_cursor_pose_is_not_stolen_by_pinch() -> None:
    """The other direction: a peace sign keeps the thumb well clear of the index."""
    cursor = _hand(curl=(False, False, True, True), thumb_to_index=1.2)
    assert recognize(cursor)["gesture"] == "two"


def test_two_fingers_up_is_the_cursor_pose() -> None:
    hand = _hand(curl=(False, False, True, True), thumb_to_index=1.0)
    result = recognize(hand)
    assert result["gesture"] == "two"
    assert result["two_spread"] is not None
    assert result["two_mid"] is not None


def test_pinch_point_sits_between_thumb_and_index() -> None:
    hand = _hand(thumb_to_index=0.4)
    x, _y = recognize(hand)["pinch_point"]
    assert x == pytest.approx((_hand(thumb_to_index=0.4)[4][0] + _hand()[8][0]) / 2.0)


def test_two_hand_gap_measures_separation() -> None:
    a = recognize(_hand(thumb_to_index=0.2))
    b = recognize(_hand(thumb_to_index=0.2))
    assert two_hand_gap(a, b) == pytest.approx(0.0)


# ── Fusion: the bug that made fist and pinch look broken ────────────────────────


def _reading(gesture: str, *, span: float = 0.2) -> dict:
    return {
        "g": {"gesture": gesture, "pinch_point": (0.5, 0.5)},
        "center": (0.5, 0.5),
        "angle": 0.0,
        "span": span,
    }


def test_agreement_scores_highest() -> None:
    gesture, confidence, agreed = score(_reading("fist"), _reading("fist"))
    assert (gesture, confidence, agreed) == ("fist", CONF_AGREED, True)


def test_the_front_camera_survives_the_side_camera_disagreeing() -> None:
    """The regression this port exists to fix.

    A fist seen edge-on is heavily foreshortened, so the side camera dissents on
    almost every frame. Scoring that below the action threshold deleted fist and
    pinch entirely, while cursor control — which never goes through fusion — kept
    working. That asymmetry is exactly what was reported.
    """
    gesture, confidence, agreed = score(_reading("fist"), _reading("open"))
    assert gesture == "fist"
    assert agreed is False
    assert confidence >= DEFAULT_MIN_CONF, "a contested front-camera reading must still act"


def test_one_camera_is_trusted_whichever_role_it_holds() -> None:
    """Unplugging the webcam left only the built-in camera, which holds the `side` role.

    Scoring that as "side-only" discounted it below the action threshold, so fist and
    pinch stopped working the moment the C920 was unplugged — while the cursor carried
    on, because it reads a view's raw hands and never passes through fusion. The
    discount is about a head-on camera failing to corroborate, not about the label.
    """
    _gesture, confidence, _agreed = score(None, _reading("fist"), dual=False)
    assert confidence >= DEFAULT_MIN_CONF


def test_a_single_camera_fist_reaches_the_caller() -> None:
    fuser = Fuser(dual=False)
    hands: list[dict] = []
    for _ in range(fuser.confirm_solo):
        hands = fuser.fuse({}, {"Right": _reading("fist")})
    assert fuser.confident(hands, "fist")


def test_nothing_seen_scores_nothing() -> None:
    assert score(None, None, dual=False) == (None, 0.0, False)


def test_a_hand_only_the_side_camera_sees_is_ignored() -> None:
    """With BOTH cameras running, a head-on view that saw nothing is real evidence
    against an edge-on detection — usually a spurious one."""
    _gesture, confidence, _agreed = score(None, _reading("fist"), dual=True)
    assert confidence == CONF_SIDE_ONLY
    assert confidence < DEFAULT_MIN_CONF


def test_a_single_camera_setup_still_acts() -> None:
    """Unplugging the side camera must degrade, not disable."""
    _gesture, confidence, _agreed = score(_reading("fist"), None)
    assert confidence == CONF_FRONT >= DEFAULT_MIN_CONF


def test_contested_fist_reaches_the_caller() -> None:
    """End to end through the fuser: disagreement costs frames, not the gesture."""
    fuser = Fuser()
    hands: list[dict] = []
    for _ in range(fuser.confirm_solo):
        hands = fuser.fuse({"Right": _reading("fist")}, {"Right": _reading("open")})
    assert [h["gesture"] for h in fuser.confident(hands, "fist")] == ["fist"]


def test_agreement_commits_faster_than_dissent() -> None:
    """Agreement is supposed to buy speed. That is the whole reason for two cameras."""
    agreeing = Fuser()
    for _ in range(agreeing.confirm_agree):
        agreed_hands = agreeing.fuse({"R": _reading("fist")}, {"R": _reading("fist")})

    contested = Fuser()
    for _ in range(contested.confirm_agree):
        contested_hands = contested.fuse({"R": _reading("fist")}, {"R": _reading("open")})

    assert agreeing.confident(agreed_hands, "fist")
    assert not contested.confident(contested_hands, "fist")


def test_two_pinching_hands_are_both_reported() -> None:
    """Resize needs both hands to survive fusion, not just the more convincing one."""
    fuser = Fuser()
    for _ in range(fuser.confirm_agree):
        hands = fuser.fuse(
            {"Left": _reading("pinch"), "Right": _reading("pinch")},
            {"Left": _reading("pinch"), "Right": _reading("pinch")},
        )
    assert len(fuser.confident(hands, "pinch")) == 2


# ── The gate ────────────────────────────────────────────────────────────────────


def test_gate_holds_until_the_gesture_repeats() -> None:
    gate = Gate(agree=2, solo=5)
    assert gate.update("fist", True) == "other"
    assert gate.update("fist", True) == "fist"


def test_gate_restarts_the_count_on_a_new_candidate() -> None:
    gate = Gate(agree=3, solo=5)
    gate.update("fist", True)
    gate.update("fist", True)
    gate.update("open", True)
    assert gate.committed == "other"


def test_gate_keeps_the_last_commitment_through_a_blip() -> None:
    """A dropped frame should not release the window mid-drag."""
    gate = Gate(agree=2, solo=5)
    gate.update("fist", True)
    gate.update("fist", True)
    assert gate.update(None, False) == "fist"


# ── Depth ───────────────────────────────────────────────────────────────────────


def test_depth_rises_as_the_hand_approaches() -> None:
    depth = Depth(ema=1.0, dead=0.0)
    far = depth.update(0.10, None)
    near = depth.update(0.30, None)
    assert far is not None and near is not None and near > far


def test_depth_ignores_jitter_inside_the_dead_zone() -> None:
    depth = Depth(ema=1.0, dead=0.5)
    depth.update(0.2, None)
    settled = depth.z
    depth.update(0.205, None)
    assert depth.z == settled
    assert depth.dz == 0.0


def test_depth_survives_a_view_dropping_out() -> None:
    """Losing the side camera mid-gesture must not jump the estimate."""
    depth = Depth(ema=1.0, dead=0.0)
    for _ in range(30):
        depth.update(0.2, 0.4)  # learn the bias between the two cameras
    both = depth.z
    depth.update(0.2, None)
    assert both is not None and depth.z is not None
    assert abs(depth.z - both) < 0.05


# ── The air mouse ───────────────────────────────────────────────────────────────


BOUNDS = (0.0, 0.0, 1000.0, 500.0)


# ── Which cameras may be used ───────────────────────────────────────────────────


DEVICES = ["Dhruvs iPhone Camera", "FaceTime HD Camera", "HD Pro Webcam C920"]


@pytest.fixture
def devices(monkeypatch) -> list[str]:
    from arc.vision.hands import cameras

    monkeypatch.setattr(cameras, "list_devices", lambda: DEVICES)
    return DEVICES


def test_the_configured_cameras_are_the_webcam_and_the_builtin(devices) -> None:
    """The two this is built around: C920 head-on, built-in camera from the side."""
    from arc.config import Config
    from arc.vision.hands.cameras import resolve

    picked = {c["role"]: resolve(c)[1] for c in Config.load().section("camera")["cameras"]}
    assert picked == {"front": "HD Pro Webcam C920", "side": "FaceTime HD Camera"}


def test_a_phone_is_never_matched_by_name(devices) -> None:
    """A phone is not part of this setup, and asking for one by name is still refused."""
    from arc.vision.hands.cameras import find_device

    assert find_device("iPhone") is None
    assert find_device("Continuity") is None


def test_a_phone_is_never_reached_by_index_either(devices) -> None:
    """Indices renumber whenever anything is plugged in, so an index can land on one."""
    from arc.vision.hands.cameras import resolve

    index, name = resolve({"role": "side", "index": 0})
    assert index is None
    assert "refusing Continuity Camera" in name


def test_the_real_cameras_are_still_reachable_by_index(devices) -> None:
    """Refusing phones must not refuse everything."""
    from arc.vision.hands.cameras import resolve

    assert resolve({"role": "front", "index": 2}) == (2, "HD Pro Webcam C920")


def test_matching_skips_past_a_phone_at_index_zero(devices) -> None:
    """The phone sits first in AVFoundation order; the webcam must still be found."""
    from arc.vision.hands.cameras import find_device

    assert find_device("C920") == 2
    assert find_device("FaceTime") == 1


# ── Not an ARC control session ──────────────────────────────────────────────────


def test_gesture_control_does_not_go_through_arcs_input_layer() -> None:
    """Turning a feature on is not ARC taking over the machine.

    Routing the cursor through arc.control.input would raise the indicator, register a
    kill-switch entry, and stop gesture control the moment the physical mouse moved —
    none of which fit a mode the user asked to be in.
    """
    for module in ("arc/vision/hands/session.py", "arc/vision/hands/__main__.py"):
        assert not any(name.startswith("arc.control") for name in _imports_of(module)), (
            f"{module} imports ARC's control layer"
        )


def _imports_of(path: str) -> set[str]:
    """Every module imported by a file, including inside functions.

    Parsed rather than grepped: these modules discuss ``arc.control.input`` in prose,
    explaining why they do not use it, and a text search cannot tell the difference.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


# ── The tools the voice command reaches ─────────────────────────────────────────


def test_the_camera_tools_are_registered() -> None:
    from arc.tools import registry

    for name in ("enable_camera_gestures", "disable_camera_gestures", "camera_gestures_status"):
        assert registry.get(name).category == "camera"


def test_turning_the_camera_on_is_mutating() -> None:
    """It opens the cameras and takes input control, so --dry-run must skip it."""
    from arc.tools import registry

    assert registry.get("enable_camera_gestures").mutating
    assert registry.get("disable_camera_gestures").mutating
    assert not registry.get("camera_gestures_status").mutating


@pytest.mark.parametrize(
    "phrase",
    ["enable camera function", "turn on the camera feature", "enable hand tracking"],
)
def test_the_spoken_phrases_reach_the_model(phrase: str) -> None:
    """The phrasings must survive into the schema the model actually sees.

    Only the docstring's *first paragraph* becomes the tool description, so putting
    these in a later section silently drops them and "enable camera function" gets a
    shrug instead of a tool call.
    """
    from arc.tools import registry

    assert phrase in registry.get("enable_camera_gestures").schema().description.lower()


def test_startup_failures_are_readable_past_mediapipes_chatter(tmp_path) -> None:
    """MediaPipe logs a dozen lines about its GL context before anything useful.

    Left in, they push the line that actually says what went wrong out of the excerpt
    the user is shown.
    """
    from arc.tools.camera import _log_tail

    log = tmp_path / "camera-gestures.log"
    log.write_text(
        "\n".join(
            [
                "I0000 gl_context.cc:407] GL version: 2.1",
                "INFO: Created TensorFlow Lite XNNPACK delegate for CPU.",
                "W0000 inference_feedback_manager.cc:121] Feedback manager requires...",
                "PlatformError: camera 'front' would not open.",
            ]
        ),
        encoding="utf-8",
    )
    tail = _log_tail(log)
    assert "would not open" in tail
    assert "XNNPACK" not in tail
    assert "gl_context" not in tail


def test_a_missing_log_still_says_something() -> None:
    from arc.tools.camera import _log_tail

    assert "no output" in _log_tail(Path("/nonexistent/camera-gestures.log"))


@pytest.mark.parametrize("phrase", ["turn off the camera feature", "stop gesture control"])
def test_the_off_phrases_reach_the_model(phrase: str) -> None:
    from arc.tools import registry

    assert phrase in registry.get("disable_camera_gestures").schema().description.lower()


def a_pointer(**kwargs) -> PointerController:
    return PointerController(BOUNDS, confirm=1, **kwargs)


def test_the_middle_of_the_frame_is_the_middle_of_the_desktop() -> None:
    assert map_to_screen(0.5, 0.5, BOUNDS) == (500.0, 250.0)


def test_the_active_margin_is_stretched_to_the_edges() -> None:
    """Hands do not reach the corners of the view, so the usable middle must fill it."""
    x, y = map_to_screen(0.12, 0.12, BOUNDS, active=0.12)
    assert (x, y) == (0.0, 0.0)


def test_mapping_is_clamped_to_the_desktop() -> None:
    x, y = map_to_screen(-0.5, 1.5, BOUNDS)
    assert (x, y) == (0.0, 500.0)


def test_fingers_together_moves_the_cursor() -> None:
    result = a_pointer().step("two", 0.20, (0.5, 0.5))
    assert result["mode"] == "mouse"
    assert result["move"] is not None


def test_spreading_the_fingers_clicks() -> None:
    assert a_pointer().step("two", 0.60, (0.5, 0.5))["click"]


def test_holding_the_spread_does_not_repeat_the_click() -> None:
    """Otherwise a held peace sign machine-guns clicks."""
    pointer = a_pointer()
    assert pointer.step("two", 0.60, (0.5, 0.5), now=1.0)["click"]
    assert not pointer.step("two", 0.60, (0.5, 0.5), now=1.1)["click"]
    assert not pointer.step("two", 0.60, (0.5, 0.5), now=5.0)["click"]


def test_bringing_the_fingers_back_together_re_arms_the_click() -> None:
    pointer = a_pointer()
    assert pointer.step("two", 0.60, (0.5, 0.5), now=1.0)["click"]
    pointer.step("two", 0.20, (0.5, 0.5), now=1.5)
    assert pointer.step("two", 0.60, (0.5, 0.5), now=2.0)["click"]


def test_the_cursor_does_not_move_mid_click() -> None:
    assert a_pointer().step("two", 0.60, (0.9, 0.9))["move"] is None


def test_the_hysteresis_band_holds_its_state() -> None:
    """Between the two thresholds nothing changes, so the pose cannot flicker."""
    pointer = a_pointer()
    pointer.step("two", 0.20, (0.5, 0.5))  # firmly together
    result = pointer.step("two", 0.41, (0.5, 0.5))  # inside the band
    assert result["mode"] == "mouse"


def test_any_other_gesture_releases_the_cursor() -> None:
    assert a_pointer().step("fist", None, None)["mode"] == "idle"


def test_a_flickered_pose_is_ignored() -> None:
    """One stray frame of the two-finger pose should not twitch the cursor."""
    pointer = PointerController(BOUNDS, confirm=3)
    assert pointer.step("two", 0.20, (0.5, 0.5))["mode"] == "arming"
    assert pointer.step("two", 0.20, (0.5, 0.5))["move"] is None


def test_re_entering_the_pose_does_not_jump_the_cursor() -> None:
    """Leaving and returning must re-seed the smoother, not drag from the old spot."""
    pointer = a_pointer(ema=0.5)
    pointer.step("two", 0.20, (0.1, 0.1))
    pointer.step(None, None, None)
    result = pointer.step("two", 0.20, (0.9, 0.9))
    assert result["move"] == pytest.approx(map_to_screen(0.9, 0.9, BOUNDS))


def test_smoothing_eases_toward_the_target() -> None:
    pointer = a_pointer(ema=0.5)
    first = pointer.step("two", 0.20, (0.5, 0.5))["move"]
    second = pointer.step("two", 0.20, (0.9, 0.5))["move"]
    target_x, _ = map_to_screen(0.9, 0.5, BOUNDS)
    assert first is not None and second is not None
    assert first[0] < second[0] < target_x
    assert not math.isnan(second[0])
