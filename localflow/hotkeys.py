"""Global hotkeys: push-to-talk, hands-free toggle, command mode.

Uses pynput (X11/Windows/macOS). Push-to-talk is hold-to-record: recording
starts when the full combo goes down and stops when any key of the combo is
released - matching Wispr Flow's hold-the-fn-key interaction.

The fn/globe key (Wispr Flow's default talk key) is invisible to pynput on
macOS - it never arrives as a key event, only as a modifier flag. A small
native listen-only CGEvent tap (_DarwinFnTap) watches for it and feeds the
same press/release path, so "<fn>" works in any combo on macOS.

Two rules keep the listener alive:

* Callbacks are exception-proofed - a raise would otherwise kill pynput's
  listener thread and silently disable all hotkeys.
* Callbacks must return fast. On macOS the listener is a CGEvent tap, and
  the OS *disables taps that block* (kCGEventTapDisabledByTimeout); doing
  seconds of transcription inside a callback bricks dictation. Callers put
  slow work on a worker thread (see cli.cmd_run).
"""

from __future__ import annotations

import sys
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


_FN_KEYCODE = 63  # kVK_Function


class _DarwinFnTap:
    """Listen-only CGEvent tap reporting fn/globe key presses (macOS).

    Watches flagsChanged events for keycode 63 and diffs the SecondaryFn
    flag. Runs its own CFRunLoop on a daemon thread. Requires the Input
    Monitoring permission the app already holds for pynput.
    """

    def __init__(self, on_change: Callable[[bool], None]) -> None:
        self._on_change = on_change  # True = fn down, False = fn up
        self._thread: Optional[threading.Thread] = None
        self._loop = None
        self._down = False

    def start(self) -> None:
        import Quartz

        def callback(proxy, etype, event, refcon):
            try:
                if etype == Quartz.kCGEventFlagsChanged:
                    keycode = Quartz.CGEventGetIntegerValueField(
                        event, Quartz.kCGKeyboardEventKeycode)
                    if keycode == _FN_KEYCODE:
                        down = bool(Quartz.CGEventGetFlags(event)
                                    & Quartz.kCGEventFlagMaskSecondaryFn)
                        if down != self._down:
                            self._down = down
                            self._on_change(down)
            except Exception:
                pass
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
            callback,
            None,
        )
        if tap is None:
            raise RuntimeError(
                "cannot create fn-key tap (Input Monitoring not granted?)")
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        ready = threading.Event()

        def run() -> None:
            self._loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(self._loop, source,
                                      Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(tap, True)
            ready.set()
            Quartz.CFRunLoopRun()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(timeout=2)

    def stop(self) -> None:
        import Quartz

        if self._loop is not None:
            Quartz.CFRunLoopStop(self._loop)
            self._loop = None


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
        self._fn_tap: Optional[_DarwinFnTap] = None
        self._lock = threading.Lock()

    # The press/release handlers are separated from pynput so tests can drive
    # them with plain strings.

    @staticmethod
    def _safe(callback: Callable[[], None]) -> None:
        """A raising callback must not take the whole listener down."""
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - keep hotkeys alive at any cost
            print(f"localflow: hotkey action failed: {exc!r}", file=sys.stderr)

    def handle_press(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            self._down.add(token)
            down = set(self._down)
        if self.toggle and self.toggle <= down and self.toggle != self.ptt:
            self._down -= self.toggle - {t for t in self.toggle if t in self.ptt}
            self._safe(self.on_toggle)
            return
        if self.command and self.command <= down and self.command != self.ptt:
            self._safe(self.on_command)
            return
        if self.ptt and self.ptt <= down and not self._ptt_active:
            self._ptt_active = True
            self._safe(self.on_ptt_press)

    def handle_release(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            self._down.discard(token)
        if self._ptt_active and token in self.ptt:
            self._ptt_active = False
            self._safe(self.on_ptt_release)

    # ------------------------------------------------------------ real keys

    def start(self) -> None:
        from pynput import keyboard  # lazy import; needs a display server

        def on_press(key):
            self.handle_press(_key_token(key))

        def on_release(key):
            self.handle_release(_key_token(key))

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

        # fn/globe support: only where a combo actually uses it (macOS).
        if sys.platform == "darwin" and "fn" in (self.ptt | self.toggle | self.command):
            try:
                self._fn_tap = _DarwinFnTap(
                    lambda down: self.handle_press("fn") if down
                    else self.handle_release("fn"))
                self._fn_tap.start()
            except Exception as exc:
                print(f"localflow: fn key unavailable: {exc}", file=sys.stderr)
                self._fn_tap = None

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
        if self._fn_tap:
            self._fn_tap.stop()
            self._fn_tap = None

    def join(self) -> None:
        if self._listener:
            self._listener.join()


# ------------------------------------------------------------ combo capture

_MODIFIER_ORDER = {"fn": 0, "ctrl": 1, "alt": 2, "shift": 3, "cmd": 4}


def format_combo(tokens: Set[str]) -> str:
    """{'ctrl','space'} -> '<ctrl>+<space>' (parse_combo round-trips)."""
    parts = sorted(tokens, key=lambda t: (_MODIFIER_ORDER.get(t, 9), t))
    return "+".join(f"<{t}>" if len(t) > 1 else t for t in parts)


def capture_combo(timeout: float = 8.0) -> Optional[str]:
    """Block until the user presses a key or combo; return it as a string.

    The combo is everything held down at the moment the first key is
    released - press Ctrl+Alt+D and you get '<ctrl>+<alt>+d'. Returns None
    on timeout. Callers must pause any active HotkeyListener first or the
    pressed keys will also trigger dictation.
    """
    from pynput import keyboard

    done = threading.Event()
    held: Set[str] = set()
    combo: Set[str] = set()

    def snapshot() -> None:
        combo.clear()
        combo.update(held)

    def on_press(key):
        token = _key_token(key)
        if token:
            held.add(token)
            snapshot()

    def on_release(key):
        done.set()
        return False  # one combo per capture: stop the listener

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    fn_tap = None
    if sys.platform == "darwin":
        def fn_change(down: bool) -> None:
            if down:
                held.add("fn")
                snapshot()
            else:
                done.set()

        try:
            fn_tap = _DarwinFnTap(fn_change)
            fn_tap.start()
        except Exception:
            fn_tap = None

    done.wait(timeout)
    listener.stop()
    if fn_tap:
        fn_tap.stop()
    return format_combo(combo) if combo else None
