"""The air-mouse decision logic, kept camera-free so it can be tested.

The two-finger pose — index and middle up, ring and pinky down — drives everything::

    fingers TOGETHER  ->  move the cursor
    fingers SPREAD    ->  left click, once, wherever the cursor already is

Telling those apart from a single spread number flickers at the boundary, so this uses
a hysteresis band plus edge detection: a click fires only on the transition *into* the
spread pose and re-arms only once the fingers come back together. Without that, holding
a peace sign machine-guns clicks.
"""

from __future__ import annotations

import time
from typing import Any


def map_to_screen(
    nx: float,
    ny: float,
    bounds: tuple[float, float, float, float],
    active: float = 0.12,
) -> tuple[float, float]:
    """Map a normalised hand point onto the desktop.

    ``active`` trims the outer margin of the frame before stretching what is left
    across the whole desktop. Hands rarely reach the edges of the camera's view, so
    mapping the full frame wastes most of the usable travel and makes the far corners
    unreachable.
    """
    dx, dy, dw, dh = bounds
    low, high = active, 1.0 - active
    fx = min(1.0, max(0.0, (nx - low) / (high - low)))
    fy = min(1.0, max(0.0, (ny - low) / (high - low)))
    return dx + fx * dw, dy + fy * dh


class PointerController:
    """Turns a stream of ``(gesture, spread, pointer)`` into cursor commands.

    :meth:`step` returns ``{"move": (x, y) | None, "click": bool, "mode": str}``.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        *,
        spread_click: float = 0.46,
        spread_mouse: float = 0.36,
        ema: float = 0.22,
        confirm: int = 2,
        click_cooldown: float = 0.35,
        active: float = 0.12,
    ) -> None:
        self.bounds = bounds
        self.spread_click = spread_click
        self.spread_mouse = spread_mouse
        self.ema = ema
        self.confirm = confirm
        self.click_cooldown = click_cooldown
        self.active = active
        self._smoothed: list[float] | None = None
        self._sub = "together"
        self._armed = True
        self._streak = 0
        self._last_click = -1e9

    def _substate(self, spread: float) -> str:
        if spread >= self.spread_click:
            return "spread"
        if spread <= self.spread_mouse:
            return "together"
        return self._sub  # inside the band: hold whatever we already decided

    def step(
        self,
        gesture: str | None,
        spread: float | None,
        pointer: tuple[float, float] | None,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        out: dict[str, Any] = {"move": None, "click": False, "mode": "idle"}

        if gesture != "two" or spread is None or pointer is None:
            self._streak = 0
            # Forget the smoothed target so re-entering the pose does not drag the
            # cursor from wherever the hand last was.
            self._smoothed = None
            return out

        self._streak += 1
        if self._streak < self.confirm:  # ignore a single flickered frame
            out["mode"] = "arming"
            return out

        self._sub = self._substate(spread)

        if self._sub == "spread":
            out["mode"] = "click"
            if self._armed and (now - self._last_click) >= self.click_cooldown:
                out["click"] = True
                self._armed = False
                self._last_click = now
            return out  # never move the cursor mid-click

        self._armed = True  # fingers reunited: re-arm
        tx, ty = map_to_screen(pointer[0], pointer[1], self.bounds, self.active)
        if self._smoothed is None:
            self._smoothed = [tx, ty]
        else:
            self._smoothed[0] += (tx - self._smoothed[0]) * self.ema
            self._smoothed[1] += (ty - self._smoothed[1]) * self.ema
        out["move"] = (self._smoothed[0], self._smoothed[1])
        out["mode"] = "mouse"
        return out
