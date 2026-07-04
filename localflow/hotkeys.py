"""Global hotkeys: push-to-talk, hands-free toggle, command mode.

Uses pynput (X11/Windows/macOS). Push-to-talk is hold-to-record: recording
starts when the full combo goes down and stops when any key of the combo is
released - matching Wispr Flow's hold-the-fn-key interaction.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Set


def _parse_combo(combo: str) -> Set[str]:
    """'<ctrl>+<space>' -> {'ctrl', 'space'} ; 'a' stays 'a'."""
    parts = [p.strip() for p in combo.split("+") if p.strip()]
    return {p.strip("<>").lower() for p in parts}


def _key_token(key) -> Optional[str]:
    """Normalize a pynput key event to a token comparable with _parse_combo."""
    try:
        from pynput.keyboard import Key, KeyCode
    except Exception:
        return None
    if isinstance(key, KeyCode):
        if key.char:
            return key.char.lower()
        return None
    if isinstance(key, Key):
        name = key.name.lower()
        # fold left/right variants: ctrl_l -> ctrl
        for base in ("ctrl", "alt", "shift", "cmd"):
            if name.startswith(base):
                return base
        return name
    return None


class HotkeyListener:
    def __init__(
        self,
        push_to_talk: str,
        toggle_dictation: str,
        command_mode: str,
        on_ptt_press: Callable[[], None],
        on_ptt_release: Callable[[], None],
        on_toggle: Callable[[], None],
        on_command: Callable[[], None],
    ) -> None:
        self.ptt = _parse_combo(push_to_talk)
        self.toggle = _parse_combo(toggle_dictation)
        self.command = _parse_combo(command_mode)
        self.on_ptt_press = on_ptt_press
        self.on_ptt_release = on_ptt_release
        self.on_toggle = on_toggle
        self.on_command = on_command
        self._down: Set[str] = set()
        self._ptt_active = False
        self._listener = None
        self._lock = threading.Lock()

    # The press/release handlers are separated from pynput so tests can drive
    # them with plain strings.

    def handle_press(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            self._down.add(token)
            down = set(self._down)
        if self.toggle and self.toggle <= down and self.toggle != self.ptt:
            self._down -= self.toggle - {t for t in self.toggle if t in self.ptt}
            self.on_toggle()
            return
        if self.command and self.command <= down and self.command != self.ptt:
            self.on_command()
            return
        if self.ptt and self.ptt <= down and not self._ptt_active:
            self._ptt_active = True
            self.on_ptt_press()

    def handle_release(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            self._down.discard(token)
        if self._ptt_active and token in self.ptt:
            self._ptt_active = False
            self.on_ptt_release()

    # ------------------------------------------------------------ real keys

    def start(self) -> None:
        from pynput import keyboard  # lazy import; needs a display server

        def on_press(key):
            self.handle_press(_key_token(key))

        def on_release(key):
            self.handle_release(_key_token(key))

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def join(self) -> None:
        if self._listener:
            self._listener.join()
