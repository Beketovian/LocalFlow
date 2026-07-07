"""The floating recording pill - Wispr Flow's signature on-screen widget.

A small, frameless, always-on-top capsule centered at the bottom of the
screen. Hidden when idle; while recording it shows a live waveform that
follows your voice, then a pulsing "formatting" state while Whisper runs.

Pure tkinter + stdlib (imported lazily) so it works on any desktop Python
without extra dependencies, and this module stays importable headlessly.

Threading model
---------------
On Linux/Windows the Tk loop runs in a background thread:

    overlay = RecordingOverlay()
    overlay.start()                # spins up the Tk thread
    overlay.show("recording")      # pill slides in, waveform live
    overlay.set_level(0.42)        # feed mic RMS (0..1) ~30x/sec
    overlay.show("processing")     # waveform -> pulsing dots
    overlay.hide()                 # pill disappears
    overlay.stop()

macOS is different in two ways. First, AppKit only allows windows on the
*main* thread (anything else dies with "NSWindow should only be instantiated
on the main thread!"). Second - and critically for dictation - a Tk window
*activates the app* when shown, stealing keyboard focus from the app the
user is dictating into, which would make the paste land on the pill itself.
So on macOS the pill is a native non-activating NSPanel drawn via PyObjC
(exactly what Wispr Flow does), and the caller keeps the main thread:

    overlay = RecordingOverlay()
    if overlay.needs_main_thread:          # True on macOS
        overlay.init_main_thread()         # build the panel, non-blocking
        threading.Thread(target=my_work, daemon=True).start()
        overlay.run_forever()              # blocks until stop()
    else:
        overlay.start(); my_work()

All state-changing methods (show/hide/set_level/stop) are thread-safe in
both modes - they post to a queue drained by the render tick.
"""

from __future__ import annotations

import math
import queue
import random
import sys
import threading
import time
from typing import Optional

# Visual constants tuned to match Wispr Flow's pill
_PILL_W = 220
_PILL_H = 44
_MARGIN_BOTTOM = 48
_BG = "#16161A"          # near-black capsule
_BAR_COLOR = "#FFFFFF"   # waveform bars
_BAR_IDLE = "#6E6E78"    # bars at rest / processing dots
_ACCENT = "#FF6A3D"      # Wispr-style coral accent (recording dot)
_N_BARS = 24
_FPS = 30


_PILL_VIEW_CLASS = None


def _pill_view_class():
    """Define the NSView subclass lazily (PyObjC classes can only be created
    once per process, and only where AppKit exists)."""
    global _PILL_VIEW_CLASS
    if _PILL_VIEW_CLASS is not None:
        return _PILL_VIEW_CLASS

    from AppKit import NSBezierPath, NSColor, NSMakeRect, NSView

    def _rgb(spec: str, alpha: float = 1.0):
        r, g, b = (int(spec[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, alpha)

    class _PillView(NSView):
        overlay = None  # set right after alloc/init

        def isFlipped(self):  # noqa: N802 - y grows downward, like Tk
            return True

        def drawRect_(self, rect):  # noqa: N802
            overlay = self.overlay
            if overlay is None or overlay._status == "hidden":
                return
            bounds = self.bounds()
            w, h = bounds.size.width, bounds.size.height

            # capsule body
            _rgb(_BG).setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                bounds, h / 2, h / 2
            ).fill()

            if overlay._status == "recording":
                _rgb(_ACCENT).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(14, h / 2 - 4, 8, 8)
                ).fill()
                span = w - 44 - 16
                step = span / _N_BARS
                for i, level in enumerate(overlay._bars):
                    bx = 36 + i * step + step / 2
                    bh = max(3.0, level * (h - 16))
                    _rgb(_BAR_COLOR if level > 0.1 else _BAR_IDLE).setFill()
                    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        NSMakeRect(bx - 1.4, h / 2 - bh / 2, 2.8, bh), 1.4, 1.4
                    ).fill()
            elif overlay._status == "processing":
                for i in range(3):
                    pulse = 0.5 + 0.5 * math.sin(overlay._phase * 6.0 - i * 0.9)
                    radius = 3 + 2.2 * pulse
                    cx = w / 2 - 22 + i * 22
                    shade = 0x6E / 255 + (0xFF - 0x6E) / 255 * pulse
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        shade, shade, shade, 1.0
                    ).setFill()
                    NSBezierPath.bezierPathWithOvalInRect_(
                        NSMakeRect(cx - radius, h / 2 - radius, radius * 2, radius * 2)
                    ).fill()

    _PILL_VIEW_CLASS = _PillView
    return _PillView


class RecordingOverlay:
    """Recording pill; see module docstring for backends and threading."""

    def __init__(self, width: int = _PILL_W, height: int = _PILL_H) -> None:
        self.width = width
        self.height = height
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._running = False
        self._mode: Optional[str] = None  # "thread" | "main"
        self._root = None
        self._canvas = None
        # state owned by the Tk thread
        self._status = "hidden"  # hidden | recording | processing
        self._level = 0.0
        self._smooth = 0.0
        self._bars = [0.08] * _N_BARS
        self._phase = 0.0
        # macOS AppKit backend
        self._panel = None
        self._view = None
        self._quit = False
        # Called every frame on the main (UI) thread; lets other AppKit UI
        # (the menu bar item) update safely without its own runloop.
        self._tick_callbacks: list = []

    # ------------------------------------------------------------ public API

    @property
    def needs_main_thread(self) -> bool:
        """True where window creation is main-thread-only (macOS/AppKit)."""
        return sys.platform == "darwin"

    def start(self) -> bool:
        """Threaded mode: run Tk in a background thread (Linux/Windows).

        Returns False when tkinter/display is unavailable. Must not be used
        on macOS - use init_main_thread() + run_forever() there.
        """
        if self._running:
            return True
        if self.needs_main_thread:
            return False
        try:
            import tkinter  # noqa: F401
        except ImportError:
            return False
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        return self._running

    def init_main_thread(self) -> bool:
        """Main-thread mode: build the pill on the calling (main) thread.

        On macOS this is a native non-activating NSPanel (falls back to Tk
        when PyObjC is unavailable). Non-blocking; follow up with
        run_forever() to drive the UI. Returns False when no UI is possible.
        """
        if self._running:
            return True
        if sys.platform == "darwin" and self._build_appkit():
            self._mode = "main"
            self._running = True
            return True
        if not self._build():
            return False
        self._mode = "main"
        self._running = True
        self._tick()
        return True

    def run_forever(self) -> None:
        """Drive the pill on the current thread (main-thread mode).

        Blocks until stop() is called from another thread (or the window is
        destroyed). KeyboardInterrupt propagates to the caller.
        """
        if self._panel is not None:
            self._run_appkit()
            return
        if self._root is None:
            return
        try:
            self._root.mainloop()
        finally:
            self._running = False
            self._finalize()

    def show(self, status: str = "recording") -> None:
        self._events.put(("show", status))

    def hide(self) -> None:
        self._events.put(("hide", None))

    def set_level(self, level: float) -> None:
        self._events.put(("level", max(0.0, min(1.0, float(level)))))

    @property
    def native(self) -> bool:
        """True when the pill runs on the AppKit backend (macOS)."""
        return self._panel is not None

    def on_tick(self, callback) -> None:
        """Register a per-frame main-thread callback (AppKit mode)."""
        self._tick_callbacks.append(callback)

    def stop(self) -> None:
        self._events.put(("quit", None))
        if self._mode == "thread" and self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        elif self._mode == "main" and self._root is not None:
            # If the mainloop already exited (e.g. Ctrl+C), the queued quit
            # event will never be drained - tear down directly when we're on
            # the thread that owns Tk.
            if threading.current_thread() is threading.main_thread() and not self._running:
                self._destroy_root()
                self._finalize()

    # -------------------------------------------------- macOS AppKit backend

    def _build_appkit(self) -> bool:
        """Native pill: a borderless, non-activating, click-through NSPanel.

        Non-activating is the whole point - showing it must never move
        keyboard focus off the app receiving the dictation.
        """
        try:
            import AppKit
            from AppKit import (
                NSBackingStoreBuffered,
                NSColor,
                NSMakeRect,
                NSPanel,
                NSScreen,
                NSStatusWindowLevel,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSWindowStyleMaskBorderless,
                NSWindowStyleMaskNonactivatingPanel,
            )

            app = AppKit.NSApplication.sharedApplication()
            # Accessory: no Dock icon; the daemon lives in the terminal.
            app.setActivationPolicy_(1)  # NSApplicationActivationPolicyAccessory

            screens = NSScreen.screens()
            if not screens:
                return False
            frame = screens[0].frame()  # the menu-bar screen
            x = frame.origin.x + (frame.size.width - self.width) / 2
            y = frame.origin.y + _MARGIN_BOTTOM

            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, self.width, self.height),
                NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
                NSBackingStoreBuffered,
                False,
            )
            panel.setLevel_(NSStatusWindowLevel)
            panel.setOpaque_(False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setHasShadow_(True)
            panel.setIgnoresMouseEvents_(True)  # click-through
            panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            view = _pill_view_class().alloc().initWithFrame_(
                NSMakeRect(0, 0, self.width, self.height)
            )
            view.overlay = self
            panel.setContentView_(view)
            self._panel, self._view = panel, view
            return True
        except Exception:
            self._panel = self._view = None
            return False

    def _run_appkit(self) -> None:
        """30 fps loop pumping Cocoa events from Python.

        Keeping Python in charge (instead of NSApp.run()) means Ctrl+C and
        stop() behave exactly like the Tk backend. Events must be dequeued
        and dispatched explicitly - just spinning the runloop draws windows
        but never delivers clicks, leaving the menu bar item unresponsive.
        """
        import AppKit
        from Foundation import NSDate, NSDefaultRunLoopMode

        app = AppKit.NSApplication.sharedApplication()
        app.finishLaunching()
        try:
            while not self._pump_appkit():
                # Wait up to one frame for an event; dispatch everything
                # pending. A status-item click starts menu tracking inside
                # sendEvent_, which nests its own runloop until dismissed.
                deadline = NSDate.dateWithTimeIntervalSinceNow_(1.0 / _FPS)
                while True:
                    event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                        AppKit.NSEventMaskAny, deadline,
                        NSDefaultRunLoopMode, True,
                    )
                    if event is None:
                        break
                    app.sendEvent_(event)
                    deadline = NSDate.dateWithTimeIntervalSinceNow_(0.0)
        finally:
            self._running = False
            try:
                self._panel.orderOut_(None)
            except Exception:
                pass

    def _pump_appkit(self) -> bool:
        """Drain the event queue; returns True when asked to quit."""
        quit_now = False
        try:
            while True:
                kind, value = self._events.get_nowait()
                if kind == "show":
                    self._status = value
                    self._panel.orderFrontRegardless()  # never activates us
                elif kind == "hide":
                    self._status = "hidden"
                    self._panel.orderOut_(None)
                elif kind == "level":
                    self._level = value
                elif kind == "quit":
                    quit_now = True
        except queue.Empty:
            pass
        for callback in self._tick_callbacks:
            try:
                callback()
            except Exception:
                pass
        if not quit_now and self._status != "hidden":
            self._advance_state()
            self._view.setNeedsDisplay_(True)
        return quit_now

    # --------------------------------------------------------- shared state

    def _advance_state(self) -> None:
        """Advance waveform/pulse animation state by one frame (any backend)."""
        self._phase += 1.0 / _FPS
        # smooth the level so bars glide instead of jitter
        self._smooth += (self._level - self._smooth) * 0.35
        if self._status == "recording":
            # waveform: shift left, append newest level with natural variance
            jitter = 0.55 + 0.45 * random.random()
            sample = min(1.0, self._smooth * 1.15 * jitter + 0.04)
            self._bars = self._bars[1:] + [max(0.08, sample)]

    # ------------------------------------------------------------- Tk thread

    def _thread_main(self) -> None:
        if not self._build():
            self._ready.set()
            return
        self._mode = "thread"
        self._running = True
        self._ready.set()
        self._tick()
        self._root.mainloop()
        self._running = False
        self._finalize()

    def _build(self) -> bool:
        try:
            import tkinter as tk
        except ImportError:
            return False
        try:
            root = tk.Tk()
        except tk.TclError:
            return False

        root.withdraw()
        root.overrideredirect(True)  # frameless
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass

        # Transparency is platform-specific; fall back to solid corners that
        # match the pill when it isn't available (e.g. bare X11).
        transparent = _BG
        if sys.platform == "darwin":
            try:
                root.attributes("-transparent", True)
                transparent = "systemTransparent"
            except tk.TclError:
                pass
        elif sys.platform.startswith("win"):
            try:
                root.attributes("-transparentcolor", "#010101")
                transparent = "#010101"
            except tk.TclError:
                pass
        root.configure(bg=transparent)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - self.width) // 2
        y = screen_h - self.height - _MARGIN_BOTTOM
        root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        canvas = tk.Canvas(
            root, width=self.width, height=self.height,
            bg=transparent, highlightthickness=0, bd=0,
        )
        canvas.pack()
        self._root, self._canvas = root, canvas
        return True

    def _destroy_root(self) -> None:
        import tkinter as tk

        try:
            if self._root is not None:
                self._root.destroy()
        except tk.TclError:
            pass

    def _finalize(self) -> None:
        # Drop Tk references on the thread that owns them and collect: if the
        # Tcl interpreter is garbage-collected from another thread at exit,
        # Tcl aborts the whole process (Tcl_AsyncDelete).
        self._root = None
        self._canvas = None
        import gc

        gc.collect()

    # ------------------------------------------------------------- rendering

    def _tick(self) -> None:
        quit_now = False
        try:
            while True:
                kind, value = self._events.get_nowait()
                if kind == "show":
                    self._status = value
                    self._root.deiconify()
                    self._root.lift()
                elif kind == "hide":
                    self._status = "hidden"
                    self._root.withdraw()
                elif kind == "level":
                    self._level = value
                elif kind == "quit":
                    quit_now = True
        except queue.Empty:
            pass

        if quit_now:
            self._destroy_root()
            return

        if self._status != "hidden":
            self._draw()

        self._root.after(int(1000 / _FPS), self._tick)

    def _draw(self) -> None:
        c = self._canvas
        c.delete("all")
        w, h, r = self.width, self.height, self.height / 2

        # capsule body
        c.create_oval(0, 0, h, h, fill=_BG, outline="")
        c.create_oval(w - h, 0, w, h, fill=_BG, outline="")
        c.create_rectangle(r, 0, w - r, h, fill=_BG, outline="")

        self._advance_state()

        if self._status == "recording":
            # recording dot
            c.create_oval(14, h / 2 - 4, 22, h / 2 + 4, fill=_ACCENT, outline="")
            span = w - 44 - 16
            step = span / _N_BARS
            for i, level in enumerate(self._bars):
                bx = 36 + i * step + step / 2
                bh = max(3.0, level * (h - 16))
                color = _BAR_COLOR if level > 0.1 else _BAR_IDLE
                c.create_rectangle(
                    bx - 1.4, h / 2 - bh / 2, bx + 1.4, h / 2 + bh / 2,
                    fill=color, outline="",
                )
        elif self._status == "processing":
            # three pulsing dots while Whisper formats the text
            for i in range(3):
                pulse = 0.5 + 0.5 * math.sin(self._phase * 6.0 - i * 0.9)
                radius = 3 + 2.2 * pulse
                cx = w / 2 - 22 + i * 22
                shade = int(0x6E + (0xFF - 0x6E) * pulse)
                color = f"#{shade:02x}{shade:02x}{shade:02x}"
                c.create_oval(cx - radius, h / 2 - radius, cx + radius, h / 2 + radius,
                              fill=color, outline="")


def run_demo(seconds: float = 6.0) -> bool:
    """Drive the overlay through its states with fake audio (manual testing)."""
    overlay = RecordingOverlay()

    def script() -> None:
        overlay.show("recording")
        t0 = time.time()
        while time.time() - t0 < seconds * 0.7:
            overlay.set_level(0.5 + 0.5 * math.sin((time.time() - t0) * 4.0))
            time.sleep(1 / 30)
        overlay.show("processing")
        time.sleep(seconds * 0.3)
        overlay.hide()
        overlay.stop()

    if overlay.needs_main_thread:
        if not overlay.init_main_thread():
            print("no display / tkinter - overlay unavailable")
            return False
        threading.Thread(target=script, daemon=True).start()
        overlay.run_forever()
        return True
    if not overlay.start():
        print("no display / tkinter - overlay unavailable")
        return False
    script()
    return True


if __name__ == "__main__":
    run_demo()
