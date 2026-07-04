"""Active-application awareness.

Wispr Flow adapts formatting to the app you're dictating into (casual in
Slack, formal in email, literal in terminals). LocalFlow does the same by
matching the focused window's title/class against configured AppProfiles.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

from .config import AppProfile


@dataclass
class WindowInfo:
    title: str = ""
    app: str = ""

    @property
    def haystack(self) -> str:
        return f"{self.app} {self.title}".lower()


class ActiveWindowProvider:
    """Best-effort focused-window lookup; degrades to 'unknown' gracefully."""

    def get(self) -> WindowInfo:
        try:
            if sys.platform == "darwin":
                return self._macos()
            if sys.platform.startswith("linux"):
                return self._linux()
            if os.name == "nt":
                return self._windows()
        except Exception:
            pass
        return WindowInfo()

    def _linux(self) -> WindowInfo:
        if shutil.which("xdotool"):
            title = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            cls = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowclassname"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            return WindowInfo(title=title, app=cls)
        return WindowInfo()

    def _macos(self) -> WindowInfo:
        script = (
            'tell application "System Events" to get name of first application '
            "process whose frontmost is true"
        )
        app = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=3
        ).stdout.strip()
        return WindowInfo(app=app)

    def _windows(self) -> WindowInfo:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return WindowInfo(title=buf.value)


def match_profile(window: WindowInfo, profiles: List[AppProfile]) -> Optional[AppProfile]:
    """Return the first profile whose match patterns hit the active window."""
    hay = window.haystack
    if not hay.strip():
        return None
    for profile in profiles:
        for pattern in profile.match:
            if pattern.lower() in hay:
                return profile
    return None
