"""ARC as a real application window.

A native window rather than a browser tab. The UI is still HTML — it is the only
sensible way to draw a live point cloud without adding a GUI toolkit — but it is
rendered by ``WKWebView`` inside an ``NSWindow`` that ARC owns, so there is no address
bar, no tab strip, and nothing that looks like a web page.

``pyobjc-framework-WebKit`` is MIT and follows the precedent already set by
``arc/control/overlay.py`` and ``arc/vision/``: Apple frameworks used directly inside
the subsystem that needs them. A browser-embedding library like ``pywebview`` would
have done the same job and pulled a dependency tree to do it (§7).

The window runs on the main thread and owns the run loop, which is exactly what speech
recognition needs anyway — ``SFSpeechRecognitionTask`` delivers on the main queue, so
the app being the run loop owner is a feature rather than a constraint.
"""

from __future__ import annotations

import contextlib
import threading

from arc.errors import ArcError
from arc.log import get_logger

_log = get_logger(__name__)

#: Matches the UI's own background so there is no white flash before the page paints.
_BACKGROUND = (0x07 / 255, 0x0B / 255, 0x11 / 255, 1.0)


def available() -> bool:
    """Whether a native window can be opened on this machine."""
    try:
        import AppKit  # noqa: F401
        import WebKit  # noqa: F401
    except Exception:
        return False
    return True


def run(url: str, *, title: str = "ARC", width: int = 1100, height: int = 760) -> int:
    """Open the window and block until it is closed.

    Must be called on the main thread: AppKit refuses to build a window anywhere else,
    and the run loop this starts is the same one speech recognition needs pumped.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ArcError("the ARC window must be opened on the main thread")

    try:
        import AppKit
        import WebKit
        from Foundation import NSURL, NSURLRequest
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ArcError(
            "the app window needs pyobjc-framework-WebKit: pip install 'arc[app]'"
        ) from exc

    app = AppKit.NSApplication.sharedApplication()
    # Regular, so ARC gets a Dock icon and a menu bar and behaves like an application
    # rather than a floating panel.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)

    frame = AppKit.NSMakeRect(0, 0, width, height)
    style = (
        AppKit.NSWindowStyleMaskTitled
        | AppKit.NSWindowStyleMaskClosable
        | AppKit.NSWindowStyleMaskMiniaturizable
        | AppKit.NSWindowStyleMaskResizable
        | AppKit.NSWindowStyleMaskFullSizeContentView
    )
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, style, AppKit.NSBackingStoreBuffered, False
    )
    window.setTitle_(title)
    # The orb is the content; a title bar drawn over it reads as chrome around a page.
    window.setTitlebarAppearsTransparent_(True)
    window.setBackgroundColor_(
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(*_BACKGROUND)
    )
    window.center()

    config = WebKit.WKWebViewConfiguration.alloc().init()
    webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(frame, config)
    webview.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    # Suppress the white default so the dark UI does not flash on load.
    # Private key; harmless if Apple moves it, so failure is suppressed rather than
    # fatal — a white flash is not worth refusing to open the window over.
    with contextlib.suppress(Exception):
        webview.setValue_forKey_(False, "drawsBackground")

    webview.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(url)))
    window.setContentView_(webview)
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)

    _log.info("app window open", extra={"url": url})
    app.run()
    return 0
