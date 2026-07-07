import json
import time
import urllib.error
import urllib.request

import numpy as np

from localflow.app import FlowController
from localflow.audio import ArrayRecorder
from localflow.config import Config
from localflow.dashboard import DashboardServer
from localflow.engines.mock import MockEngine
from localflow.history import History
from localflow.injector import CallbackInjector
from localflow.sounds import SoundPlayer
from localflow.streaming import StreamingPreview


def speechlike(seconds=1.0, rate=16000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


class TestStreamingPreview:
    def test_partials_emitted(self):
        engine = MockEngine(["partial one", "partial two"])
        recorder = ArrayRecorder(speechlike(2.0))
        recorder.start()
        partials = []
        preview = StreamingPreview(engine, recorder, partials.append, interval=0.05)
        preview.start()
        time.sleep(0.4)
        preview.stop()
        assert partials, "expected at least one partial"
        assert partials[0] == "partial one"
        # duplicates are suppressed
        assert len(partials) == len(set(partials))

    def test_short_audio_skipped(self):
        engine = MockEngine(["x"])
        recorder = ArrayRecorder(np.zeros(1000, dtype=np.float32))
        partials = []
        preview = StreamingPreview(engine, recorder, partials.append, interval=0.05)
        preview.start()
        time.sleep(0.2)
        preview.stop()
        assert partials == []
        assert engine.calls == []


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


class TestDashboard:
    def setup_method(self):
        self.controller = make_controller()
        self.server = DashboardServer(self.controller, port=0)  # ephemeral port
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

    def test_index_serves_html(self):
        status, body = self._get("/")
        assert status == 200
        assert b"LocalFlow" in body

    def test_stats_api(self):
        self.controller.history.add("", "three words here", duration=2.0)
        status, body = self._get("/api/stats")
        data = json.loads(body)
        assert status == 200
        assert data["total_words"] == 3

    def test_history_api_and_search(self):
        self.controller.history.add("", "alpha beta")
        self.controller.history.add("", "gamma delta")
        _, body = self._get("/api/history?q=gamma")
        entries = json.loads(body)["entries"]
        assert len(entries) == 1
        assert entries[0]["formatted_text"] == "gamma delta"

    def test_dictionary_roundtrip(self, tmp_path, monkeypatch):
        # keep config.save() away from the real home directory
        monkeypatch.setattr(
            self.controller.config, "save", lambda path=None: tmp_path / "c.json"
        )
        status, _ = self._post("/api/dictionary", {"add": "Kubernetes"})
        assert status == 200
        _, body = self._get("/api/dictionary")
        assert "Kubernetes" in json.loads(body)["words"]
        self._post("/api/dictionary", {"remove": "Kubernetes"})
        _, body = self._get("/api/dictionary")
        assert "Kubernetes" not in json.loads(body)["words"]

    def test_replacement_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            self.controller.config, "save", lambda path=None: tmp_path / "c.json"
        )
        self._post("/api/dictionary", {"replace_from": "eta", "replace_to": "ETA"})
        _, body = self._get("/api/dictionary")
        assert json.loads(body)["replacements"]["eta"] == "ETA"

    def test_history_delete(self):
        eid = self.controller.history.add("", "to be deleted")
        self._post("/api/history/delete", {"id": eid})
        assert self.controller.history.recent() == []

    def test_state_endpoint(self):
        _, body = self._get("/api/state")
        assert json.loads(body)["status"] == "idle"

    def test_404(self):
        try:
            self._get("/api/nonsense")
            raised = False
        except urllib.error.HTTPError as exc:
            raised = exc.code == 404
        assert raised
