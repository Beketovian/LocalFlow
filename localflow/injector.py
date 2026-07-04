"""Text injection - put transcribed text into whatever app has focus.

Backends, in order of preference on each platform:

* type      - simulate keystrokes (pynput; xdotool on X11)
* clipboard - copy to clipboard and press Ctrl/Cmd+V (fast for long text)
* stdout    - print to stdout (piping into other tools, headless use)
* callback  - hand the text to a Python callable (embedding/testing)
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


def create_injector(method: str = "auto", type_interval: float = 0.0,
                    restore_clipboard: bool = True) -> Injector:
    if method == "none":
        return CallbackInjector()
    if method == "stdout":
        return StdoutInjector()
    if method == "clipboard":
        return ClipboardInjector(restore=restore_clipboard)
    if method == "type":
        try:
            return TypeInjector(interval=type_interval)
        except Exception:
            return XdotoolInjector()
    # auto
    if sys.platform.startswith("linux") and shutil.which("xdotool"):
        try:
            return XdotoolInjector()
        except Exception:
            pass
    try:
        return TypeInjector(interval=type_interval)
    except Exception:
        return StdoutInjector()
