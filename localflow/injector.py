"""Text injection - put transcribed text into whatever app has focus.

Backends, in order of preference on each platform:

* paste     - macOS: pbcopy + a synthesized Cmd+V (what Wispr Flow does;
              per-character typing is unreliable in Messages, Slack, etc.)
* type      - simulate keystrokes (pynput; xdotool on X11)
* clipboard - copy to clipboard and press Ctrl/Cmd+V (fast for long text)
* stdout    - print to stdout (piping into other tools, headless use)
* callback  - hand the text to a Python callable (embedding/testing)

"auto" picks paste on macOS, xdotool on X11, typing elsewhere.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from typing import Callable, List, Optional


class Injector:
    name = "base"

    def inject(self, text: str) -> None:
        raise NotImplementedError


class CallbackInjector(Injector):
    """Delivers text to a callable; the default in tests."""

    name = "callback"

    def __init__(self, callback: Optional[Callable[[str], None]] = None) -> None:
        self.received: List[str] = []
        self._callback = callback

    def inject(self, text: str) -> None:
        self.received.append(text)
        if self._callback:
            self._callback(text)


class StdoutInjector(Injector):
    name = "stdout"

    def inject(self, text: str) -> None:
        print(text, flush=True)


class TypeInjector(Injector):
    """Simulate typing via pynput (works on X11/Windows/macOS)."""

    name = "type"

    def __init__(self, interval: float = 0.0) -> None:
        from pynput.keyboard import Controller  # lazy import

        self._keyboard = Controller()
        self.interval = interval

    def inject(self, text: str) -> None:
        if self.interval <= 0:
            self._keyboard.type(text)
            return
        for char in text:
            self._keyboard.type(char)
            time.sleep(self.interval)


class XdotoolInjector(Injector):
    """X11 typing via xdotool - most reliable on Linux desktops."""

    name = "xdotool"

    def __init__(self) -> None:
        if not shutil.which("xdotool"):
            raise RuntimeError("xdotool not found")

    def inject(self, text: str) -> None:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "3", "--", text],
            check=True,
        )


class ClipboardInjector(Injector):
    """Copy to clipboard, press paste, optionally restore old clipboard."""

    name = "clipboard"

    def __init__(self, restore: bool = True) -> None:
        import pyperclip  # lazy; optional dependency

        self._clip = pyperclip
        self.restore = restore

    def inject(self, text: str) -> None:
        from pynput.keyboard import Controller, Key

        old = None
        if self.restore:
            try:
                old = self._clip.paste()
            except Exception:
                old = None
        self._clip.copy(text)
        kb = Controller()
        paste_mod = Key.cmd if sys.platform == "darwin" else Key.ctrl
        with kb.pressed(paste_mod):
            kb.press("v")
            kb.release("v")
        if self.restore and old is not None:
            time.sleep(0.15)  # let the paste land before restoring
            try:
                self._clip.copy(old)
            except Exception:
                pass


class MacPasteInjector(Injector):
    """macOS: copy via pbcopy, paste via a synthesized Cmd+V.

    Simulated per-character typing (pynput's CGEventKeyboardSetUnicodeString)
    drops or garbles text in many apps - Messages, Slack, Electron apps -
    and is slow for long dictations. Pasting is what Wispr Flow does and it
    works everywhere. Cmd+V is posted through Quartz with the command flag
    set explicitly, so a modifier still held from the hotkey can't corrupt it.
    """

    name = "paste"

    def __init__(self, restore: bool = True) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("MacPasteInjector is macOS-only")
        if not shutil.which("pbcopy"):
            raise RuntimeError("pbcopy not found")
        self.restore = restore
        # Set by the controller before inject(): pid of the app the user was
        # dictating into. If focus drifted (e.g. to our own recording pill),
        # inject() re-activates it so the paste lands in the right place.
        self.focus_pid = 0

    @staticmethod
    def _read_clipboard() -> Optional[str]:
        try:
            out = subprocess.run(["pbpaste"], capture_output=True, timeout=2)
            return out.stdout.decode("utf-8", "replace")
        except Exception:
            return None

    @staticmethod
    def _write_clipboard(text: str) -> None:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, timeout=2)

    @staticmethod
    def _press_cmd_v() -> None:
        try:
            import Quartz  # bundled with pynput's pyobjc dependency

            src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            v_key = 9  # kVK_ANSI_V
            for down in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(src, v_key, down)
                Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
                time.sleep(0.01)
            return
        except Exception:
            pass
        # Fallbacks: pynput, then AppleScript.
        try:
            from pynput.keyboard import Controller, Key

            kb = Controller()
            with kb.pressed(Key.cmd):
                kb.press("v")
                kb.release("v")
            return
        except Exception:
            pass
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke "v" using command down'],
            check=True, timeout=5,
        )

    def _ensure_target_focus(self) -> None:
        if not self.focus_pid:
            return
        try:
            from AppKit import (
                NSApplicationActivateIgnoringOtherApps,
                NSRunningApplication,
                NSWorkspace,
            )

            workspace = NSWorkspace.sharedWorkspace()
            front = workspace.frontmostApplication()
            if front is not None and int(front.processIdentifier()) == self.focus_pid:
                return
            target = NSRunningApplication.runningApplicationWithProcessIdentifier_(
                self.focus_pid
            )
            if target is None:
                return
            target.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            for _ in range(20):  # wait up to ~1s for the switch to land
                time.sleep(0.05)
                front = workspace.frontmostApplication()
                if front is not None and int(front.processIdentifier()) == self.focus_pid:
                    break
        except Exception:
            pass  # best effort - paste still goes to whatever is frontmost

    def press_return(self) -> None:
        """Press Enter in the target app (voice action: '... send it')."""
        self._ensure_target_focus()
        try:
            import Quartz

            src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
            for down in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(src, 36, down)  # kVK_Return
                Quartz.CGEventSetFlags(event, 0)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
                time.sleep(0.01)
        except Exception:
            from pynput.keyboard import Controller, Key

            kb = Controller()
            kb.press(Key.enter)
            kb.release(Key.enter)

    def inject(self, text: str) -> None:
        self._ensure_target_focus()
        old = self._read_clipboard() if self.restore else None
        self._write_clipboard(text)
        time.sleep(0.05)  # let the pasteboard settle before the keystroke
        self._press_cmd_v()
        if self.restore and old is not None:
            # Wait for the frontmost app to actually read the pasteboard;
            # restoring too early makes it paste the *old* clipboard.
            time.sleep(0.5)
            try:
                self._write_clipboard(old)
            except Exception:
                pass


def create_injector(method: str = "auto", type_interval: float = 0.0,
                    restore_clipboard: bool = True) -> Injector:
    if method == "none":
        return CallbackInjector()
    if method == "stdout":
        return StdoutInjector()
    if method in ("clipboard", "paste"):
        if sys.platform == "darwin":
            return MacPasteInjector(restore=restore_clipboard)
        return ClipboardInjector(restore=restore_clipboard)
    if method == "type":
        try:
            return TypeInjector(interval=type_interval)
        except Exception:
            return XdotoolInjector()
    # auto
    if sys.platform == "darwin":
        try:
            return MacPasteInjector(restore=restore_clipboard)
        except Exception:
            pass
    if sys.platform.startswith("linux") and shutil.which("xdotool"):
        try:
            return XdotoolInjector()
        except Exception:
            pass
    try:
        return TypeInjector(interval=type_interval)
    except Exception:
        return StdoutInjector()
