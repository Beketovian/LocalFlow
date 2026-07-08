"""Tests for the embedded (in-process) LLM backend plumbing.

The real MLX engine is never loaded here - these cover model discovery and
backend routing, which is where the packaging bugs would live.
"""

from __future__ import annotations

from pathlib import Path

import localflow.llm as llm_mod
from localflow.config import LLMConfig
from localflow.llm import LLMClient
from localflow.llm_local import find_local_model


def make_mlx_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x")
    return d


class TestFindLocalModel:
    def test_explicit_path(self, tmp_path):
        model = make_mlx_dir(tmp_path, "some-model")
        assert find_local_model(tmp_path / "data", str(model)) == model

    def test_explicit_path_invalid(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert find_local_model(tmp_path / "data", str(tmp_path / "empty")) is None

    def test_own_dir_discovered(self, tmp_path):
        data = tmp_path / "data"
        model = make_mlx_dir(data / "models" / "llm", "qwen3-4b-instruct-2507")
        assert find_local_model(data) == model

    def test_non_model_dirs_skipped(self, tmp_path):
        data = tmp_path / "data"
        (data / "models" / "llm" / "junk").mkdir(parents=True)
        assert find_local_model(data) is None


class TestBackendRouting:
    def test_server_backend_never_probes_embedded(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "localflow.llm_local.find_local_model",
            lambda *a, **k: calls.append(1) or None,
        )
        client = LLMClient(LLMConfig(backend="server",
                                     base_url="http://127.0.0.1:9/v1"),
                           data_dir=tmp_path)
        assert client.available is False
        assert calls == []

    def test_embedded_backend_unavailable_without_weights(self, tmp_path):
        client = LLMClient(LLMConfig(backend="embedded"), data_dir=tmp_path)
        assert client.available is False
        assert client.mode is None

    def test_embedded_resolves_when_weights_exist(self, tmp_path, monkeypatch):
        # mlx_lm import stays lazy, so probing works even though the engine
        # would only load weights on first chat.
        monkeypatch.setattr(llm_mod.importlib.util, "find_spec",
                            lambda name: object())
        data = tmp_path
        make_mlx_dir(data / "models" / "llm", "test-model-4bit")
        client = LLMClient(LLMConfig(backend="embedded"), data_dir=data)
        assert client.available is True
        assert client.mode == "embedded"
        assert client.model == "test-model-4bit"
        assert client.models == ["test-model-4bit"]
        assert "MLX" in client.base_url

    def test_auto_prefers_embedded_over_running_server(self, tmp_path, monkeypatch):
        # A running LM Studio (base_url "auto" probes it) must NOT win over
        # local weights: dictation would hijack whatever model the user has
        # loaded there for other work.
        from tests.test_llm import FakeLLMServer

        monkeypatch.setattr(llm_mod.importlib.util, "find_spec",
                            lambda name: object())
        server = FakeLLMServer()
        try:
            make_mlx_dir(tmp_path / "models" / "llm", "local-model")
            monkeypatch.setattr(llm_mod, "_PROBE_URLS", (server.base_url,))
            client = LLMClient(LLMConfig(backend="auto"), data_dir=tmp_path)
            assert client.mode == "embedded"
            assert client.model == "local-model"
        finally:
            server.stop()

    def test_auto_with_explicit_base_url_prefers_that_server(self, tmp_path, monkeypatch):
        # Setting a base_url is a deliberate "use my server" - it wins even
        # when local weights exist.
        from tests.test_llm import FakeLLMServer

        monkeypatch.setattr(llm_mod.importlib.util, "find_spec",
                            lambda name: object())
        server = FakeLLMServer()
        try:
            make_mlx_dir(tmp_path / "models" / "llm", "local-model")
            client = LLMClient(
                LLMConfig(backend="auto", base_url=server.base_url),
                data_dir=tmp_path,
            )
            assert client.mode == "server"
        finally:
            server.stop()

    def test_auto_falls_back_to_server_without_weights(self, tmp_path):
        from tests.test_llm import FakeLLMServer

        server = FakeLLMServer()
        try:
            client = LLMClient(
                LLMConfig(backend="auto", base_url=server.base_url),
                data_dir=tmp_path,
            )
            assert client.mode == "server"
        finally:
            server.stop()
