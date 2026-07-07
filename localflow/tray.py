"""System tray icon (optional; needs pystray + Pillow and a desktop session)."""

from __future__ import annotations

import threading
import webbrowser
from typing import Optional

from .app import FlowController

_COLORS = {"idle": (122, 162, 255), "recording": (235, 87, 87), "transcribing": (242, 201, 76)}


def _make_icon(color):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 56, 56), fill=color + (255,))
    # small microphone silhouette
    draw.rounded_rectangle((26, 16, 38, 38), radius=6, fill=(255, 255, 255, 230))
    draw.arc((20, 26, 44, 46), start=0, end=180, fill=(255, 255, 255, 230), width=3)
    draw.line((32, 46, 32, 52), fill=(255, 255, 255, 230), width=3)
    return img


class TrayIcon:
    def __init__(self, controller: FlowController, dashboard_url: Optional[str] = None,
                 on_quit=None) -> None:
        self.controller = controller
        self.dashboard_url = dashboard_url
        self.on_quit = on_quit
        self._icon = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        import pystray

        menu = pystray.Menu(
            pystray.MenuItem(
                "Open dashboard",
                lambda: webbrowser.open(self.dashboard_url) if self.dashboard_url else None,
                default=True,
            ),
            pystray.MenuItem("Reset session context", lambda: self.controller.reset_session()),
            pystray.MenuItem("Quit", lambda: self._quit()),
        )
        self._icon = pystray.Icon("localflow", _make_icon(_COLORS["idle"]), "LocalFlow", menu)
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def set_status(self, status: str) -> None:
        if self._icon:
            self._icon.icon = _make_icon(_COLORS.get(status, _COLORS["idle"]))

    def _quit(self) -> None:
        if self._icon:
            self._icon.stop()
        if self.on_quit:
            self.on_quit()

    def stop(self) -> None:
        if self._icon:
            self._icon.stop()
            self._icon = None
