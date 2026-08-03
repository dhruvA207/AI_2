"""The gesture loop: cameras in, window and cursor commands out.

  fist (one hand)              -> grab the frontmost window and MOVE it
  pinch with BOTH hands        -> RESIZE that window
  two fingers TOGETHER         -> move the cursor
  two fingers SPREAD (peace)   -> left click
  anything else                -> release

Runs as its own process (``python -m arc.vision.hands``): OpenCV's window needs the main
thread, and MediaPipe is heavy enough to stall an agent loop sharing one with it.

**Not an ARC control session.** This is a mode you switch on and off, the same as typing
a command in a terminal to start it — your own hand is what moves the cursor, so there
is no takeover to announce. It raises no indicator, registers no kill-switch entry, and
keeps running when you use the physical mouse. Cursor output therefore goes through
:mod:`arc.vision.hands.cursor` rather than :mod:`arc.control.input`, which would impose
all three.

Stop it by asking ARC to turn the camera feature off, or with ESC or ``q`` in the
preview window.
"""

from __future__ import annotations

import math
from typing import Any

from arc.log import get_logger
from arc.vision.hands import cursor
from arc.vision.hands.fusion import Fuser
from arc.vision.hands.pointer import PointerController
from arc.vision.hands.views import CameraView
from arc.vision.hands.windows import WindowController, desktop_bounds

_log = get_logger(__name__)

# Window movement is incremental with pointer-style acceleration rather than a fixed
# hand-to-screen mapping: a slow hand places a window precisely, a fast one throws it
# across every display without the hand running out of travel.
BASE_GAIN = 0.9
FAST_GAIN = 4.0
ACCEL_KNEE = 0.012
MOVE_DEAD = 0.0015
EDGE_KEEP = 120  # px of the window that must stay reachable

MIN_WIDTH, MIN_HEIGHT = 220, 160
SMOOTH_MOVE = 0.30
SMOOTH_FAST = 0.75
SMOOTH_KNEE = 250.0
SMOOTH_SIZE = 0.25
RESIZE_DAMP = 0.55

MODE_COLOR = {
    "IDLE": (180, 180, 180),
    "MOVING": (120, 255, 120),
    "RESIZING": (80, 200, 255),
    "MOUSE": (120, 255, 120),
    "CLICK": (80, 120, 255),
}

PREVIEW_TITLE = "ARC gesture control (ESC to quit)"


def _two_finger(view: CameraView | None) -> tuple[str | None, float | None, Any]:
    """The two-finger pose from one view's raw hands.

    Reads a single view rather than the fused hands on purpose: the cursor needs a
    response on the frame it happens, and the head-on camera is where a peace sign is
    unambiguous. The window gestures are the ones worth spending confirmation frames on.
    """
    if view is None:
        return None, None, None
    for hand in view.hands:
        gesture = hand["g"]
        if gesture["gesture"] == "two":
            return "two", gesture["two_spread"], gesture["two_mid"]
    return None, None, None


def _pinch_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(
        a["pinch_point"][0] - b["pinch_point"][0],
        a["pinch_point"][1] - b["pinch_point"][1],
    )


class GestureSession:
    """Owns the cameras, the fuser, and the per-frame state machine."""

    def __init__(self, config: dict[str, Any], *, control_mouse: bool = True) -> None:
        self.config = config

        self.views: dict[str, CameraView] = {}
        for camera in config.get("cameras", []):
            view = CameraView(
                camera,
                max_hands=int(config.get("max_hands", 2)),
                mirror=bool(config.get("mirror", True)),
                track_width=int(config.get("track_width", 960)),
            )
            if view.ok:
                self.views[view.role] = view
        self.front = self.views.get("front")
        self.side = self.views.get("side")

        # Built after the cameras, because how a reading is scored depends on how many
        # of them are actually running — see fusion.score.
        fusion = config.get("fusion", {})
        self.fuser = Fuser(
            min_conf=float(fusion.get("min_conf", 0.7)),
            confirm_agree=int(fusion.get("confirm_agree", 2)),
            confirm_solo=int(fusion.get("confirm_solo", 5)),
            depth_ema=float(fusion.get("depth_ema", 0.15)),
            depth_dead=float(fusion.get("depth_dead", 0.01)),
            dual=len(self.views) > 1,
        )

        self.windows = WindowController()
        self.bounds = desktop_bounds()
        mouse = config.get("mouse", {})
        self.pointer = (
            PointerController(
                self.bounds,
                spread_click=float(mouse.get("spread_click", 0.46)),
                spread_mouse=float(mouse.get("spread_mouse", 0.36)),
                ema=float(mouse.get("ema", 0.22)),
                confirm=int(mouse.get("confirm", 2)),
                click_cooldown=float(mouse.get("click_cooldown", 0.35)),
                active=float(mouse.get("active", 0.12)),
            )
            if control_mouse
            else None
        )

        self.mode = "IDLE"
        self.pointer_mode = "idle"
        self._ref: dict[str, Any] = {}

    @property
    def dual(self) -> bool:
        return self.front is not None and self.side is not None

    def read(self) -> list[dict[str, Any]]:
        """One frame from every view, fused."""
        front = {h["label"]: h for h in (self.front.poll() if self.front else [])}
        side = {h["label"]: h for h in (self.side.poll() if self.side else [])}
        return self.fuser.fuse(front, side)

    # ── The state machine ───────────────────────────────────────────────────────

    def step(self, hands: list[dict[str, Any]]) -> None:
        """Advance window control by one frame."""
        fists = self.fuser.confident(hands, "fist")
        pinches = self.fuser.confident(hands, "pinch")
        desired = "RESIZING" if len(pinches) >= 2 else ("MOVING" if fists else "IDLE")

        if desired != self.mode:
            self.mode = desired
            if desired == "MOVING":
                self._begin_move(fists)
            elif desired == "RESIZING":
                self._begin_resize(pinches)
            else:
                self.windows.release()
            return

        if self.mode == "MOVING" and fists:
            self._continue_move(fists)
        elif self.mode == "RESIZING" and len(pinches) >= 2:
            self._continue_resize(pinches)

    def _begin_move(self, fists: list[dict[str, Any]]) -> None:
        if not (self.windows.acquire() and self.windows.frame()):
            self.mode = "IDLE"
            return
        frame = self.windows.frame()
        assert frame is not None
        self._ref = {
            "hand": (fists[0]["x"], fists[0]["y"]),
            "target": [frame[0], frame[1]],
            "smoothed": [frame[0], frame[1]],
            "size": (frame[2], frame[3]),
            # Re-read every grab: displays get plugged and unplugged.
            "bounds": desktop_bounds(),
        }

    def _continue_move(self, fists: list[dict[str, Any]]) -> None:
        hx, hy = fists[0]["x"], fists[0]["y"]
        dx = hx - self._ref["hand"][0]
        dy = hy - self._ref["hand"][1]
        self._ref["hand"] = (hx, hy)
        if abs(dx) < MOVE_DEAD:
            dx = 0.0
        if abs(dy) < MOVE_DEAD:
            dy = 0.0

        bx, by, bw, bh = self._ref["bounds"]
        speed = math.hypot(dx, dy)
        gain = BASE_GAIN + (FAST_GAIN - BASE_GAIN) * min(1.0, speed / ACCEL_KNEE)
        self._ref["target"][0] += dx * bw * gain
        self._ref["target"][1] += dy * bh * gain

        # Keep a grabbable strip of the window on some display.
        width, _height = self._ref["size"]
        self._ref["target"][0] = min(
            max(self._ref["target"][0], bx - (width - EDGE_KEEP)), bx + bw - EDGE_KEEP
        )
        self._ref["target"][1] = min(max(self._ref["target"][1], by), by + bh - EDGE_KEEP)

        # Adaptive smoothing: chase hard when far behind so crossing a display
        # boundary does not crawl, settle gently when close.
        gap_x = self._ref["target"][0] - self._ref["smoothed"][0]
        gap_y = self._ref["target"][1] - self._ref["smoothed"][1]
        alpha = SMOOTH_MOVE + (SMOOTH_FAST - SMOOTH_MOVE) * min(
            1.0, math.hypot(gap_x, gap_y) / SMOOTH_KNEE
        )
        self._ref["smoothed"][0] += gap_x * alpha
        self._ref["smoothed"][1] += gap_y * alpha
        self.windows.set_position(*self._ref["smoothed"])

    def _begin_resize(self, pinches: list[dict[str, Any]]) -> None:
        if not (self.windows.acquire() and self.windows.frame()):
            self.mode = "IDLE"
            return
        frame = self.windows.frame()
        assert frame is not None
        gap = _pinch_gap(pinches[0], pinches[1]) + 1e-6
        self._ref = {
            "gap": gap,
            "smoothed_gap": gap,
            "size": (frame[2], frame[3]),
            "smoothed_size": [frame[2], frame[3]],
        }

    def _continue_resize(self, pinches: list[dict[str, Any]]) -> None:
        gap = _pinch_gap(pinches[0], pinches[1])
        self._ref["smoothed_gap"] += (gap - self._ref["smoothed_gap"]) * SMOOTH_SIZE
        raw = self._ref["smoothed_gap"] / self._ref["gap"]
        scale = 1.0 + (raw - 1.0) * RESIZE_DAMP
        target_w = max(MIN_WIDTH, self._ref["size"][0] * scale)
        target_h = max(MIN_HEIGHT, self._ref["size"][1] * scale)
        self._ref["smoothed_size"][0] += (target_w - self._ref["smoothed_size"][0]) * SMOOTH_SIZE
        self._ref["smoothed_size"][1] += (target_h - self._ref["smoothed_size"][1]) * SMOOTH_SIZE
        self.windows.set_size(*self._ref["smoothed_size"])

    # ── Cursor ──────────────────────────────────────────────────────────────────

    def step_pointer(self) -> None:
        """Advance cursor control, but only while no window gesture is active."""
        if self.pointer is None:
            self.pointer_mode = "off"
            return
        if self.mode != "IDLE":
            self.pointer.step(None, None, None)  # window gesture wins
            self.pointer_mode = "idle"
            return

        gesture, spread, point = _two_finger(self.front or self.side)
        result = self.pointer.step(gesture, spread, point)
        self.pointer_mode = result["mode"]

        if result["move"] is not None:
            cursor.move_to(*result["move"])
        if result["click"]:
            cursor.left_click()

    def close(self) -> None:
        for view in self.views.values():
            view.close()
        self.views.clear()
