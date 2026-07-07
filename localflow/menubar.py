"""macOS menu bar item - LocalFlow's face when running as an app.

An NSStatusItem showing a mic glyph (coral while recording), with a menu:
open the dashboard, toggle hands-free, quit. It rides the AppKit runloop
that the recording pill already owns (see overlay.RecordingOverlay.on_tick),
so status updates from worker threads are queued and applied on the main
thread each frame.
"""

from __future__ import annotations

import queue
import sys
import webbrowser
from typing import Callable, Optional

_TITLES = {
    "idle": "◎",         # ◎
    "recording": "◉",    # ◉
    "transcribing": "◌", # ◌
}

_TARGET_CLASS = None


def _target_class():
    """NSObject subclass for menu actions (PyObjC classes are one-shot)."""
    global _TARGET_CLASS
    if _TARGET_CLASS is not None:
        return _TARGET_CLASS

    from AppKit import NSObject

    class _LocalFlowMenuTarget(NSObject):
        callbacks: dict = None

        def openDashboard_(self, sender):  # noqa: N802
            cb = (self.callbacks or {}).get("dashboard")
            if cb:
                cb()

        def toggleHandsFree_(self, sender):  # noqa: N802
            cb = (self.callbacks or {}).get("hands_free")
            if cb:
                cb()

        def quitApp_(self, sender):  # noqa: N802
            cb = (self.callbacks or {}).get("quit")
            if cb:
                cb()

    _TARGET_CLASS = _LocalFlowMenuTarget
    return _TARGET_CLASS


class MacMenuBar:
    """Builds the status item; call attach(overlay) to receive UI ticks.

    Must be constructed on the main thread (AppKit). Raises RuntimeError
    off-macOS or when AppKit is unavailable - callers treat that as
    "no menu bar", nothing else breaks.
    """

    def __init__(
        self,
        version: str,
        dashboard_url: Optional[str],
        on_quit: Callable[[], None],
        on_toggle_hands_free: Optional[Callable[[], None]] = None,
        on_dashboard: Optional[Callable[[], None]] = None,
    ) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("menu bar is macOS-only")
        try:
            from AppKit import (
                NSMenu,
                NSMenuItem,
                NSStatusBar,
                NSVariableStatusItemLength,
            )
        except Exception as exc:  # pragma: no cover - no pyobjc
            raise RuntimeError(f"AppKit unavailable: {exc}")

        self._pending: "queue.Queue[str]" = queue.Queue()
        self._hands_free_on = False

        if on_dashboard is None and dashboard_url:
            on_dashboard = lambda: webbrowser.open(dashboard_url)  # noqa: E731
        self._target = _target_class().alloc().init()
        self._target.callbacks = {
            "dashboard": on_dashboard,
            "hands_free": on_toggle_hands_free,
            "quit": on_quit,
        }

        self._item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self._item.button().setTitle_(_TITLES["idle"])
        self._item.button().setToolTip_("LocalFlow - local voice dictation")

        menu = NSMenu.alloc().init()
        title = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"LocalFlow {version}", None, "")
        title.setEnabled_(False)
        menu.addItem_(title)
        menu.addItem_(NSMenuItem.separatorItem())

        if self._target.callbacks["dashboard"]:
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Open Dashboard", "openDashboard:", "")
            item.setTarget_(self._target)
            menu.addItem_(item)

        if on_toggle_hands_free:
            self._hf_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Hands-free Mode", "toggleHandsFree:", "")
            self._hf_item.setTarget_(self._target)
            menu.addItem_(self._hf_item)
        else:
            self._hf_item = None

        menu.addItem_(NSMenuItem.separatorItem())
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit LocalFlow", "quitApp:", "q")
        quit_item.setTarget_(self._target)
        menu.addItem_(quit_item)
        self._item.setMenu_(menu)

    # ------------------------------------------------------------- updates

    def set_status(self, status: str) -> None:
        """Thread-safe; applied on the next UI tick."""
        self._pending.put(status)

    def set_hands_free(self, on: bool) -> None:
        self._hands_free_on = on
        self._pending.put("__hf__")

    def attach(self, overlay) -> None:
        overlay.on_tick(self._drain)

    def _drain(self) -> None:
        try:
            while True:
                status = self._pending.get_nowait()
                if status == "__hf__":
                    if self._hf_item is not None:
                        self._hf_item.setState_(1 if self._hands_free_on else 0)
                elif status in _TITLES:
                    self._item.button().setTitle_(_TITLES[status])
        except queue.Empty:
            pass

    def remove(self) -> None:
        try:
            from AppKit import NSStatusBar

            NSStatusBar.systemStatusBar().removeStatusItem_(self._item)
        except Exception:
            pass
