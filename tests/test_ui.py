"""Tests for the dashboard UI, settings API, status events and the overlay."""

import json
import sys
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

    def test_busy_port_falls_back_to_ephemeral(self):
        # a second server on the same port must come up anyway
        other = DashboardServer(make_controller(), port=self.port)
        try:
            port2 = other.start()
            assert port2 != self.port
            with urllib.request.urlopen(f"http://127.0.0.1:{port2}/api/state",
                                        timeout=5) as resp:
                assert resp.status == 200
        finally:
            other.stop()

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

    def test_start_refuses_background_thread_on_macos(self, monkeypatch):
        # AppKit kills the process if a window is made off the main thread;
        # threaded start() must refuse on darwin so callers take the
        # init_main_thread()/run_forever() path instead.
        import localflow.overlay as overlay_mod

        monkeypatch.setattr(overlay_mod.sys, "platform", "darwin")
        overlay = overlay_mod.RecordingOverlay()
        assert overlay.needs_main_thread is True
        assert overlay.start() is False
        assert overlay._thread is None  # never spawned a Tk thread

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="macOS uses AppKit (no DISPLAY) and always has a window server",
    )
    def test_init_main_thread_without_display(self, monkeypatch):
        pytest.importorskip("tkinter")
        monkeypatch.delenv("DISPLAY", raising=False)
        from localflow.overlay import RecordingOverlay

        overlay = RecordingOverlay()
        assert overlay.init_main_thread() is False

    def test_run_forever_without_init_returns(self):
        from localflow.overlay import RecordingOverlay

        RecordingOverlay().run_forever()  # no root -> immediate no-op

    def test_level_clamped(self):
        from localflow.overlay import RecordingOverlay

        overlay = RecordingOverlay()
        overlay.set_level(7.5)
        kind, value = overlay._events.get_nowait()
        assert (kind, value) == ("level", 1.0)
        overlay.set_level(-3)
        kind, value = overlay._events.get_nowait()
        assert value == 0.0


class TestHotkeyRecording:
    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.stop()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def test_record_endpoint_returns_combo(self):
        controller = make_controller()
        self.server = DashboardServer(
            controller, port=0, hotkey_recorder=lambda: "<ctrl>+<alt>+k")
        self.port = self.server.start()
        assert self._post("/api/hotkeys/record", {}) == {"combo": "<ctrl>+<alt>+k"}

    def test_record_endpoint_without_recorder(self):
        controller = make_controller()
        self.server = DashboardServer(controller, port=0)
        self.port = self.server.start()
        assert self._post("/api/hotkeys/record", {}) == {"combo": None}

    def test_hotkey_settings_patch_triggers_rebuild(self):
        rebuilds = []
        controller = make_controller()
        self.server = DashboardServer(
            controller, port=0, on_hotkeys_changed=lambda: rebuilds.append(1))
        self.port = self.server.start()
        self._post("/api/settings", {"hotkeys": {"push_to_talk": "<fn>"}})
        assert controller.config.hotkeys.push_to_talk == "<fn>"
        assert rebuilds == [1]
        # non-hotkey patches don't churn the listener
        self._post("/api/settings", {"formatting": {"remove_fillers": True}})
        assert rebuilds == [1]


class TestEngineSwitching:
    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.stop()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def _get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=5) as resp:
            return json.loads(resp.read())

    def test_engine_endpoint_reports_status(self, tmp_path):
        controller = make_controller()
        controller.config.data_dir = str(tmp_path)  # empty models dir
        self.server = DashboardServer(controller, port=0)
        self.server.engine_status = lambda: {"switching": True,
                                             "message": "downloading small... 42%"}
        self.port = self.server.start()
        s = self._get("/api/engine")
        assert s["switching"] is True
        assert "42%" in s["message"]
        assert s["downloaded"] == []

    def test_engine_endpoint_lists_downloaded(self, tmp_path):
        controller = make_controller()
        controller.config.data_dir = str(tmp_path)
        models = tmp_path / "models"
        models.mkdir(parents=True)
        (models / "ggml-small.bin").write_bytes(b"x")
        (models / "ggml-base.bin").write_bytes(b"x")
        self.server = DashboardServer(controller, port=0)
        self.port = self.server.start()
        assert self._get("/api/engine")["downloaded"] == ["base", "small"]

    def test_engine_patch_triggers_switch(self, tmp_path, monkeypatch):
        controller = make_controller()
        monkeypatch.setattr(controller.config, "save",
                            lambda path=None: tmp_path / "c.json")
        switches = []
        self.server = DashboardServer(controller, port=0)
        self.server.on_engine_changed = lambda: switches.append(
            controller.config.engine.model)
        self.port = self.server.start()
        self._post("/api/settings", {"engine": {"model": "large-v3"}})
        assert switches == ["large-v3"]
        self._post("/api/settings", {"formatting": {"remove_fillers": True}})
        assert switches == ["large-v3"]  # unrelated patches don't switch
