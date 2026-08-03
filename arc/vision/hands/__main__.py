"""Run gesture control as its own process.

    python -m arc.vision.hands                 # windows + cursor
    python -m arc.vision.hands --no-mouse      # windows only
    python -m arc.vision.hands --single        # front camera only
    python -m arc.vision.hands --no-preview    # no window; useful when driven by a tool

Started by :mod:`arc.tools.camera`, but runnable directly for tuning, which is how the
thresholds get checked against a real hand rather than against a unit test.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from typing import Any

from arc.config import Config
from arc.log import get_logger
from arc.vision.hands.session import MODE_COLOR, PREVIEW_TITLE, GestureSession

_log = get_logger(__name__)


def _check_accessibility() -> None:
    """Say so loudly when the permission that makes any of this work is missing.

    Without it every window move and every click is silently ignored, which looks
    exactly like a tracking failure and sends you debugging the wrong thing.
    """
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:  # pragma: no cover - non-macOS
        return
    if not AXIsProcessTrusted():
        print(
            "\n[!] No Accessibility permission — windows will not move and clicks will\n"
            "    not land. Grant it to your terminal in System Settings >\n"
            "    Privacy & Security > Accessibility, then start this again.\n",
            file=sys.stderr,
        )


def _draw_preview(cv2: Any, session: GestureSession, hands: list[dict[str, Any]]) -> Any:
    """Build the preview image: front camera, skeletons, mode, and the side inset."""
    view = session.front or session.side
    if view is None or view.frame is None:
        return None

    frame = view.frame.copy()
    if view.tracker is not None:
        view.tracker.draw_on(frame, view.hands)
    height, width = frame.shape[:2]

    for hand in hands:
        px, py = int(hand["x"] * width), int(hand["y"] * height)
        tag = f"{hand['gesture'].upper()} {hand['conf']:.2f}"
        colour = (0, 230, 255) if hand["agreed"] else (150, 200, 200)
        cv2.putText(
            frame, tag, (px - 40, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA
        )

    cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
    if session.mode != "IDLE":
        shown = session.mode
        label = f"MODE: {shown}" + (f"  ->  {session.windows.name}" if session.windows.name else "")
    elif session.pointer_mode in ("mouse", "click"):
        shown = session.pointer_mode.upper()
        label = f"MODE: {shown}"
    else:
        shown, label = "IDLE", "MODE: IDLE"
    cv2.putText(
        frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, MODE_COLOR[shown], 2, cv2.LINE_AA
    )

    cameras = "2 CAM" if session.dual else "1 CAM"
    cv2.putText(
        frame,
        cameras,
        (width - 90, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 230, 255) if session.dual else (120, 120, 120),
        2,
        cv2.LINE_AA,
    )
    tip = "fist = move window | pinch(2 hands) = resize | 2 fingers = cursor | spread = click"
    cv2.putText(
        frame,
        tip,
        (12, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    display = cv2.resize(frame, (460, int(frame.shape[0] * 460 / frame.shape[1])))
    if session.side is not None and session.front is not None:
        display = _side_inset(cv2, display, session.side)
    return display


def _side_inset(cv2: Any, display: Any, view: Any, fraction: float = 0.34) -> Any:
    """Drop the side camera into the corner, skeleton and all.

    Shows the second camera is actually contributing rather than asking the user to
    trust a badge that says it is.
    """
    if view.frame is None:
        return display
    frame = view.frame.copy()
    if view.tracker is not None:
        view.tracker.draw_on(frame, view.hands)

    width = int(display.shape[1] * fraction)
    height = int(frame.shape[0] * width / frame.shape[1])
    x0, y0 = 6, display.shape[0] - height - 6
    if y0 < 0 or x0 + width > display.shape[1]:
        return display

    display[y0 : y0 + height, x0 : x0 + width] = cv2.resize(frame, (width, height))
    cv2.rectangle(display, (x0 - 1, y0 - 1), (x0 + width, y0 + height), (0, 230, 255), 1)
    cv2.putText(
        display,
        f"SIDE  hands:{len(view.hands)}",
        (x0 + 4, y0 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )
    return display


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arc.vision.hands")
    parser.add_argument("--single", action="store_true", help="front camera only")
    parser.add_argument("--no-mouse", action="store_true", help="windows only, no cursor")
    parser.add_argument("--no-preview", action="store_true", help="do not open a preview window")
    parser.add_argument("--camera", type=int, help="force a camera index for the front view")
    args = parser.parse_args(argv)

    _check_accessibility()

    config = dict(Config.load().section("camera"))
    cameras = [dict(c) for c in config.get("cameras", [])]
    if args.single:
        cameras = [c for c in cameras if c.get("role") == "front"]
    if args.camera is not None:
        for camera in cameras:
            if camera.get("role") == "front":
                camera["index"] = args.camera
    config["cameras"] = cameras

    session = GestureSession(config, control_mouse=not args.no_mouse)
    if not session.views:
        print(
            "No cameras opened. Check the `match` names under `camera.cameras` in "
            "config/default.yaml, and Camera access in System Settings.",
            file=sys.stderr,
        )
        session.close()
        return 1

    bx, by, bw, bh = session.bounds
    print(f"[hands] {'cross-camera' if session.dual else 'single-camera'} mode")
    print(f"[hands] desktop {bw}x{bh} at ({bx}, {by})")

    cv2 = None
    if not args.no_preview:
        import cv2 as _cv2

        cv2 = _cv2
        width, _height = screen_size_safe()
        cv2.namedWindow(PREVIEW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_TITLE, 460, 260)
        cv2.moveWindow(PREVIEW_TITLE, max(0, width - 480), 40)
        # Keep the preview on top: it is small, and it is the only feedback showing
        # whether a gesture is being seen at all.
        with contextlib.suppress(Exception):
            cv2.setWindowProperty(PREVIEW_TITLE, cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            hands = session.read()
            session.step(hands)
            session.step_pointer()

            if cv2 is not None:
                display = _draw_preview(cv2, session, hands)
                if display is not None:
                    cv2.imshow(PREVIEW_TITLE, display)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return 0


def screen_size_safe() -> tuple[int, int]:
    from arc.vision.hands.windows import screen_size

    try:
        return screen_size()
    except Exception:  # pragma: no cover - defensive
        return (1440, 900)


if __name__ == "__main__":
    sys.exit(main())
