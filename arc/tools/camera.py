"""Camera gesture control, as tools the agent can call.

The feature itself lives in :mod:`arc.vision.hands` and runs as its own process. These
are the handles: start it, stop it, ask whether it is on. Nothing more — this is a
switch, the equivalent of typing a command in a terminal to start gesture control and
another to stop it, with ARC doing the typing.

Starting is mutating even though nothing is typed or clicked at the moment of the call.
It opens the cameras and leaves a process that will move the pointer, and ``--dry-run``
promising not to touch the machine has to mean that.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from arc.errors import ToolError
from arc.log import get_logger
from arc.paths import log_dir
from arc.tools.registry import tool

_log = get_logger(__name__)

#: How the process is identified for stopping and status. Matching on the module path
#: is enough to be unambiguous — nothing else on the machine runs it.
_PROCESS_MATCH = "arc.vision.hands"

#: How long to watch for an immediate crash before reporting success. The failures
#: worth catching — a missing extra, a missing model, no camera at all — all happen in
#: the first second or two. Kept short because a voice turn waits on this call, and
#: every second here is a second of silence before ARC confirms.
_STARTUP_GRACE = 2.0

#: Where the gesture process's output goes, under ARC's usual log directory.
_LOG_NAME = "camera-gestures.log"

#: MediaPipe announces its graph, its GL context, and its delegates on every start.
#: None of it is a problem, and all of it would bury the line that is.
_NOISE = (
    "inference_feedback_manager",
    "XNNPACK",
    "Fiber init",
    "gl_context",
    "Custom gesture classifier",
    "landmark_projection_calculator",
)


def _log_tail(path: Path, lines: int = 12) -> str:
    """The interesting end of the gesture log, with MediaPipe's chatter dropped."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"(no output; see {path})"
    kept = [
        line for line in text.splitlines() if line.strip() and not any(n in line for n in _NOISE)
    ]
    return "\n".join(kept[-lines:]) or f"(no output; see {path})"


def _running() -> list[int]:
    """PIDs of any gesture-control process."""
    found = subprocess.run(
        ["/usr/bin/pgrep", "-f", _PROCESS_MATCH], capture_output=True, text=True, check=False
    )
    return [int(line) for line in found.stdout.split() if line.isdigit()]


@tool(category="camera", mutating=True)
def enable_camera_gestures(mouse: bool = True, preview: bool = True) -> str:
    """Turn on the camera gesture feature, so hands in front of the camera control
    windows and the mouse. Call this for "enable camera function", "turn on the camera
    feature", "turn on camera gestures", "enable hand tracking", "start gesture
    control", or "let me control the screen with my hands".

    Once on, with your hand in view of the camera:
      - a closed fist grabs the frontmost window and moves it
      - pinching with both hands resizes that window
      - two fingers held together moves the mouse cursor
      - spreading those two fingers into a peace sign left-clicks

    It stays on until turned off — say "turn off the camera feature", or press ESC in
    the preview window.

    Args:
        mouse: Control the cursor as well as windows. False for windows only.
        preview: Show the camera preview window with the tracked hand skeleton.
    """
    already = _running()
    if already:
        return f"camera gestures are already running (pid {already[0]})"

    # -u: unbuffered, so the log below is current rather than a stale buffer.
    command = [sys.executable, "-u", "-m", _PROCESS_MATCH]
    if not mouse:
        command.append("--no-mouse")
    if not preview:
        command.append("--no-preview")

    # A file, never a pipe. MediaPipe logs enough on startup to fill a pipe buffer that
    # nothing is draining, at which point the gesture process blocks forever; and a
    # pipe dies with whoever opened it, so gesture control would end when ARC restarts
    # rather than when asked to.
    log_path = log_dir() / _LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # Its own session, so a signal aimed at ARC does not take a feature the
                # user switched on down with it.
                start_new_session=True,
            )
    except OSError as exc:
        raise ToolError(f"could not start camera gestures: {exc}") from exc

    # Watch the first few seconds. A camera that will not open, a missing model, and a
    # missing extra all fail here, and reporting "started" over the top of that is
    # worse than waiting.
    deadline = time.monotonic() + _STARTUP_GRACE
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ToolError("camera gestures failed to start:\n" + _log_tail(log_path))
        time.sleep(0.2)

    _log.info("camera gestures started", extra={"pid": process.pid, "mouse": mouse})
    return (
        f"camera gestures on (pid {process.pid})\n"
        "fist = move window · both hands pinch = resize · "
        "two fingers together = cursor · spread = click\n"
        f"log: {log_path}"
    )


@tool(category="camera", mutating=True)
def disable_camera_gestures() -> str:
    """Turn the camera gesture feature off and release the cameras. Call this for
    "turn off the camera feature", "disable camera gestures", "stop hand tracking",
    "stop gesture control", or "close the camera".
    """
    pids = _running()
    if not pids:
        return "camera gestures are not running"

    subprocess.run(["/usr/bin/pkill", "-f", _PROCESS_MATCH], capture_output=True, check=False)
    # It releases the cameras on its way out; give it a moment so a status check
    # straight after this does not race the teardown and report it still running.
    for _ in range(20):
        if not _running():
            break
        time.sleep(0.1)

    _log.info("camera gestures stopped", extra={"pids": pids})
    return f"camera gestures off (stopped pid {', '.join(str(p) for p in pids)})"


@tool(category="camera")
def camera_gestures_status() -> str:
    """Report whether camera gesture control is currently running."""
    pids = _running()
    if not pids:
        return "camera gestures are off"
    return f"camera gestures are on (pid {', '.join(str(p) for p in pids)})"
