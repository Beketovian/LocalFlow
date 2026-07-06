"""Tests for the local LLM layer: client, sanitization, controller wiring."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localflow.config import Config, LLMConfig
from localflow.llm import LLMClient, _sanitize


class FakeLLMServer:
    """Minimal OpenAI-compatible server: /models and /chat/completions."""

    def __init__(self, models=None, reply="cleaned text"):
        self.models = models or ["text-embedding-nomic", "google/gemma-4-26b-a4b"]
        self.reply = reply
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._send({"data": [{"id": m} for m in outer.models]})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                reply = outer.reply(payload) if callable(outer.reply) else outer.reply
                self._send({"choices": [{"message": {"content": reply}}]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def fake_server():
    server = FakeLLMServer()
    yield server
    server.stop()


def client_for(server, **overrides) -> LLMClient:
    overrides.setdefault("min_chars", 0)  # tests use short inputs
    cfg = LLMConfig(base_url=server.base_url, timeout=5.0, **overrides)
    return LLMClient(cfg)


class TestProbe:
    def test_detects_server_and_skips_non_chat_models(self, fake_server):
        client = client_for(fake_server)
        assert client.available is True
        assert client.model == "google/gemma-4-26b-a4b"  # embedding skipped
        assert client.base_url == fake_server.base_url

    def test_no_server(self):
        client = LLMClient(LLMConfig(base_url="http://127.0.0.1:9/v1"))
        assert client.available is False
        assert client.rewrite("hello") is None

    def test_auto_with_no_probe_urls_is_unavailable(self):
        client = LLMClient(LLMConfig(base_url="auto"))
        assert client.available is False

    def test_explicit_model_wins(self, fake_server):
        client = client_for(fake_server, model="openai/gpt-oss-20b")
        assert client.available is True
        assert client.model == "openai/gpt-oss-20b"


class TestRewrite:
    def test_basic(self, fake_server):
        client = client_for(fake_server)
        assert client.rewrite("um hello world") == "cleaned text"
        sent = fake_server.requests[0]
        assert sent["model"] == "google/gemma-4-26b-a4b"
        assert "um hello world" in sent["messages"][1]["content"]

    def test_tone_and_app_in_prompt(self, fake_server):
        client = client_for(fake_server)
        client.rewrite("hey there", tone="casual", app="Messages")
        sent = fake_server.requests[0]
        assert "casual" in sent["messages"][0]["content"]
        assert "Messages" in sent["messages"][0]["content"]
        assert sent["messages"][1]["content"] == "hey there"  # dictation only

    def test_rejects_runaway_output(self, fake_server):
        fake_server.reply = "way too long " * 100
        client = client_for(fake_server)
        assert client.rewrite("short input") is None

    def test_rejects_empty_output(self, fake_server):
        fake_server.reply = ""
        client = client_for(fake_server)
        assert client.rewrite("some text here") is None

    def test_skips_overlong_input(self, fake_server):
        client = client_for(fake_server, max_chars=10)
        assert client.rewrite("x" * 50) is None
        assert fake_server.requests == []

    def test_skips_tiny_input(self, fake_server):
        # "sounds good" isn't worth a model round-trip; rules handle it
        client = client_for(fake_server, min_chars=15)
        assert client.rewrite("sounds good") is None
        assert fake_server.requests == []

    def test_output_capped(self, fake_server):
        client = client_for(fake_server)
        client.rewrite("hello world um okay")
        assert fake_server.requests[0]["max_tokens"] >= 96

    def test_gpt_oss_gets_low_reasoning_effort(self, fake_server):
        client = client_for(fake_server, model="openai/gpt-oss-20b")
        client.rewrite("hello world um okay")
        assert fake_server.requests[0]["reasoning_effort"] == "low"

    def test_other_models_get_no_reasoning_param(self, fake_server):
        client = client_for(fake_server)  # auto -> gemma
        client.rewrite("hello world um okay")
        assert "reasoning_effort" not in fake_server.requests[0]


class TestEdit:
    def test_edit(self, fake_server):
        fake_server.reply = "HELLO"
        client = client_for(fake_server)
        assert client.edit("make it shout", "hello") == "HELLO"


class TestSanitize:
    def test_strips_think_tags(self):
        assert _sanitize("<think>hmm, ok</think>Hello there") == "Hello there"

    def test_strips_fences(self):
        assert _sanitize("```text\nHello\n```") == "Hello"

    def test_strips_wrapping_quotes(self):
        assert _sanitize('"Hello there."') == "Hello there."

    def test_keeps_interior_quotes(self):
        assert _sanitize('"quoted" and not') == '"quoted" and not'


class TestControllerIntegration:
    def make_controller(self, server, **llm_overrides):
        from localflow.app import FlowController
        from localflow.audio import ArrayRecorder
        from localflow.engines.mock import MockEngine
        from localflow.history import History
        from localflow.injector import CallbackInjector
        from localflow.sounds import SoundPlayer
        import numpy as np

        config = Config()
        config.save_history = False
        config.llm.base_url = server.base_url if server else "http://127.0.0.1:9/v1"
        config.llm.timeout = 5.0
        config.llm.min_chars = 0  # tests use short inputs
        for key, value in llm_overrides.items():
            setattr(config.llm, key, value)
        injector = CallbackInjector()
        controller = FlowController(
            config=config,
            engine=MockEngine(["um hello world"]),
            recorder=ArrayRecorder(np.zeros(16000, dtype=np.float32)),
            injector=injector,
            history=History(":memory:"),
            sounds=SoundPlayer(enabled=False),
        )
        return controller, injector

    def test_dictation_uses_llm(self, fake_server):
        fake_server.reply = "Hello, world!"
        controller, injector = self.make_controller(fake_server)
        controller.start_recording()
        event = controller.stop_recording()
        assert event.llm_used is True
        assert event.formatted_text == "Hello, world!"
        assert injector.received == ["Hello, world! "]

    def test_falls_back_to_rules_when_server_down(self):
        controller, injector = self.make_controller(None)
        controller.start_recording()
        event = controller.stop_recording()
        assert event.llm_used is False
        assert event.formatted_text == "Hello world"  # rule-based path

    def test_disabled_llm_never_called(self, fake_server):
        controller, _ = self.make_controller(fake_server, enabled=False)
        controller.start_recording()
        event = controller.stop_recording()
        assert event.llm_used is False
        assert fake_server.requests == []

    def test_command_mode_free_form_uses_llm(self, fake_server):
        fake_server.reply = "Bonjour le monde"
        controller, injector = self.make_controller(fake_server)
        edited = controller.run_command("translate to french", "hello world")
        assert edited == "Bonjour le monde"

    def test_command_mode_rules_still_win(self, fake_server):
        controller, _ = self.make_controller(fake_server)
        edited = controller.run_command("uppercase", "hello")
        assert edited == "HELLO"
        assert fake_server.requests == []  # rule matched; no LLM call


class TestConfigRoundtrip:
    def test_llm_section_persists(self, tmp_path):
        cfg = Config()
        cfg.llm.enabled = False
        cfg.llm.base_url = "http://127.0.0.1:1234/v1"
        cfg.llm.model = "openai/gpt-oss-20b"
        path = cfg.save(tmp_path / "config.json")
        loaded = Config.load(path)
        assert loaded.llm.enabled is False
        assert loaded.llm.base_url == "http://127.0.0.1:1234/v1"
        assert loaded.llm.model == "openai/gpt-oss-20b"
