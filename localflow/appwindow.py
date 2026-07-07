"""Native dashboard window - the Wispr Flow-style app window (macOS).

Wraps the local dashboard in a real NSWindow + WKWebView instead of sending
users to a browser tab. Rides the same main-thread AppKit pump as the pill
and menu bar; menu actions already run on the main thread, so the menu's
"Open Dashboard" can create and show this window directly.

Closing the window hides it (the daemon keeps running); reopening from the
menu brings it back instantly.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Optional

_DELEGATE_CLASS = None


def _delegate_class():
    """Window delegate: close button hides instead of destroying."""
    global _DELEGATE_CLASS
    if _DELEGATE_CLASS is not None:
        return _DELEGATE_CLASS

    from AppKit import NSObject

    class _LocalFlowWindowDelegate(NSObject):
        def windowShouldClose_(self, sender):  # noqa: N802
            sender.orderOut_(None)
            return False

    _DELEGATE_CLASS = _LocalFlowWindowDelegate
    return _DELEGATE_CLASS


class DashboardWindow:
    """Lazy NSWindow around the dashboard URL. Main-thread only."""

    def __init__(self, url: str, width: int = 1080, height: int = 720) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("DashboardWindow is macOS-only")
        if importlib.util.find_spec("WebKit") is None:
            raise RuntimeError("pyobjc-framework-WebKit not installed")
        self.url = url
        self.width = width
        self.height = height
        self._window = None
        self._webview = None
        self._delegate = None

    def _build(self) -> None:
        import AppKit
        import WebKit
        from Foundation import NSMakeRect, NSURL, NSURLRequest

        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
            | AppKit.NSWindowStyleMaskResizable
        )
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.width, self.height),
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setTitle_("LocalFlow")
        window.setReleasedWhenClosed_(False)
        window.center()
        self._delegate = _delegate_class().alloc().init()
        window.setDelegate_(self._delegate)

        config = WebKit.WKWebViewConfiguration.alloc().init()
        webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(
            window.contentView().bounds(), config
        )
        webview.setAutoresizingMask_(
            AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable
        )
        window.contentView().addSubview_(webview)
        webview.loadRequest_(
            NSURLRequest.requestWithURL_(NSURL.URLWithString_(self.url))
        )
        self._window, self._webview = window, webview

    def show(self) -> None:
        """Create on first use, then bring to front. Call on the main thread."""
        import AppKit

        if self._window is None:
            self._build()
        self._window.makeKeyAndOrderFront_(None)
        # Accessory apps aren't active by default; without this the window
        # appears but keystrokes keep going to the previous app.
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def close(self) -> None:
        if self._window is not None:
            self._window.orderOut_(None)
