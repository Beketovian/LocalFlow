"""The floating recording pill - Wispr Flow's signature on-screen widget.

A small, frameless, always-on-top capsule centered at the bottom of the
screen. Hidden when idle; while recording it shows a live waveform that
follows your voice, then a pulsing "formatting" state while Whisper runs.

Pure tkinter + stdlib (imported lazily) so it works on any desktop Python
without extra dependencies, and this module stays importable headlessly.

    overlay = RecordingOverlay()
    overlay.start()                # spins up the Tk thread
    overlay.show("recording")      # pill slides in, waveform live
    overlay.set_level(0.42)        # feed mic RMS (0..1) ~30x/sec
    overlay.show("processing")     # waveform -> pulsing dots
    overlay.hide()                 # pill disappears
"""

from __future__ import annotations

import math
import queue
import random
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


class RecordingOverlay:
    """Thread-hosted tkinter overlay. All public methods are thread-safe."""

    def __init__(self, width: int = _PILL_W, height: int = _PILL_H) -> None:
        self.width = width
        self.height = height
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._running = False
        # state owned by the Tk thread
        self._status = "hidden"  # hidden | recording | processing
        self._level = 0.0
        self._smooth = 0.0
        self._bars = [0.08] * _N_BARS
        self._phase = 0.0

    # ------------------------------------------------------------ public API

    def start(self) -> bool:
        """Start the Tk thread. Returns False when no display is available."""
        if self._running:
            return True
        try:
            import tkinter  # noqa: F401
        except ImportError:
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        return self._running

    def show(self, status: str = "recording") -> None:
        self._events.put(("show", status))

    def hide(self) -> None:
        self._events.put(("hide", None))

    def set_level(self, level: float) -> None:
        self._events.put(("level", max(0.0, min(1.0, float(level)))))

    def stop(self) -> None:
        self._events.put(("quit", None))
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    # ------------------------------------------------------------- Tk thread

    def _run(self) -> None:
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError:
            self._ready.set()
            return

        self._running = True
        root.withdraw()
        root.overrideredirect(True)  # frameless
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        # Transparent background where supported (X11 needs a compositor;
        # fall back to a plain black window corner otherwise).
        transparent = "#010101"
        for attr, value in (("-transparentcolor", transparent), ("-transparent", True)):
            try:
                root.attributes(attr, value)
                break
            except tk.TclError:
                transparent = _BG  # no transparency: corners match the pill
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

        self._ready.set()
        self._tick()
        root.mainloop()
        self._running = False
        # Finalize Tcl in this thread: if the interpreter is collected from
        # another thread at exit, Tcl aborts the process (Tcl_AsyncDelete).
        self._root = None
        self._canvas = None
        del canvas, root
        import gc

        gc.collect()

    # ------------------------------------------------------------- rendering

    def _tick(self) -> None:
        import tkinter as tk

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
            try:
                self._root.destroy()
            except tk.TclError:
                pass
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

        self._phase += 1.0 / _FPS
        # smooth the level so bars glide instead of jitter
        self._smooth += (self._level - self._smooth) * 0.35

        if self._status == "recording":
            # recording dot
            c.create_oval(14, h / 2 - 4, 22, h / 2 + 4, fill=_ACCENT, outline="")
            # waveform: shift left, append newest level with natural variance
            jitter = 0.55 + 0.45 * random.random()
            sample = min(1.0, self._smooth * 1.15 * jitter + 0.04)
            self._bars = self._bars[1:] + [max(0.08, sample)]
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
    if not overlay.start():
        print("no display / tkinter - overlay unavailable")
        return False
    overlay.show("recording")
    t0 = time.time()
    while time.time() - t0 < seconds * 0.7:
        overlay.set_level(0.5 + 0.5 * math.sin((time.time() - t0) * 4.0))
        time.sleep(1 / 30)
    overlay.show("processing")
    time.sleep(seconds * 0.3)
    overlay.hide()
    overlay.stop()
    return True


if __name__ == "__main__":
    run_demo()
