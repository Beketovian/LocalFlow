"""Tests for the dashboard UI, settings API, status events and the overlay."""

import json
import urllib.request

import numpy as np
import pytest

from localflow.app import FlowController
from localflow.audio import ArrayRecorder
from localflow.config import Config
from localflow.dashboard import DashboardServer
from localflow.engines.mock import MockEngine
from localflow.history import History
from localflow.injector import CallbackInjector
from localflow.sounds import SoundPlayer


def speechlike(seconds=1.0, rate=16000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def make_controller():
    config = Config()
    config.save_history = False
    return FlowController(
        config=config,
        engine=MockEngine(["hello"]),
        recorder=ArrayRecorder(speechlike()),
        injector=CallbackInjector(),
        history=History(":memory:"),
        sounds=SoundPlayer(enabled=False),
    )


class TestStatusEvents:
    def test_status_sequence(self):
        controller = make_controller()
        statuses = []
        controller.on_status(statuses.append)
        controller.start_recording()
        controller.stop_recording()
        assert statuses == ["recording", "transcribing", "idle"]

    def test_cancel_goes_idle(self):
        controller = make_controller()
        statuses = []
        controller.on_status(statuses.append)
        controller.start_recording()
        controller.cancel_recording()
        assert statuses == ["recording", "idle"]

    def test_listener_error_does_not_break_flow(self):
        controller = make_controller()
        controller.on_status(lambda s: 1 / 0)
        controller.start_recording()
        event = controller.stop_recording()
        assert event is not None


class TestDashboardUI:
    def setup_method(self):
        self.controller = make_controller()
        self.server = DashboardServer(self.controller, port=0)
        self.port = self.server.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def teardown_method(self):
        self.server.stop()

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as resp:
            return resp.status, resp.read()

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())

    def test_serves_flow_ui(self):
        status, body = self._get("/")
        assert status == 200
        page = body.decode()
        # the real Wispr-style app shell, not the fallback page
        for marker in ("page-home", "page-history", "page-dictionary",
                       "page-settings", "day streak", "Recent activity"):
            assert marker in page, f"missing UI marker: {marker}"

    def test_get_settings(self):
        status, body = self._get("/api/settings")
        cfg = json.loads(body)
        assert status == 200
        assert cfg["formatting"]["remove_fillers"] is True
        assert "push_to_talk" in cfg["hotkeys"]

    def test_patch_formatting_toggle(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.controller.config, "save",
                            lambda path=None: tmp_path / "c.json")
        _, cfg = self._post("/api/settings", {"formatting": {"remove_fillers": False}})
        assert cfg["formatting"]["remove_fillers"] is False
        assert self.controller.config.formatting.remove_fillers is False

    def test_patch_user_name_and_hotkey(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.controller.config, "save",
                            lambda path=None: tmp_path / "c.json")
        _, cfg = self._post("/api/settings", {
            "user_name": "Ada",
            "hotkeys": {"push_to_talk": "<f9>"},
        })
        assert cfg["user_name"] == "Ada"
        assert cfg["hotkeys"]["push_to_talk"] == "<f9>"

    def test_patch_rejects_unknown_and_wrong_types(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.controller.config, "save",
                            lambda path=None: tmp_path / "c.json")
        _, cfg = self._post("/api/settings", {
            "formatting": {"remove_fillers": "yes-please", "bogus_key": True},
            "not_a_section": {"x": 1},
        })
        # wrong type ignored, unknown keys ignored, config unharmed
        assert cfg["formatting"]["remove_fillers"] is True
        assert "not_a_section" not in cfg or cfg.get("not_a_section") is None

    def test_settings_persisted_via_save(self, tmp_path, monkeypatch):
        saved = []
        monkeypatch.setattr(self.controller.config, "save",
                            lambda path=None: saved.append(True))
        self._post("/api/settings", {"user_name": "Ada"})
        assert saved


class TestOverlay:
    def test_module_imports_headless(self):
        # must be importable (and start() must fail gracefully) with no display
        from localflow.overlay import RecordingOverlay

        overlay = RecordingOverlay()
        assert overlay._status == "hidden"

    def test_start_without_display(self, monkeypatch):
        pytest.importorskip("tkinter")
        monkeypatch.delenv("DISPLAY", raising=False)
        from localflow.overlay import RecordingOverlay

        overlay = RecordingOverlay()
        assert overlay.start() is False  # no display -> graceful False

    def test_level_clamped(self):
        from localflow.overlay import RecordingOverlay

        overlay = RecordingOverlay()
        overlay.set_level(7.5)
        kind, value = overlay._events.get_nowait()
        assert (kind, value) == ("level", 1.0)
        overlay.set_level(-3)
        kind, value = overlay._events.get_nowait()
        assert value == 0.0
