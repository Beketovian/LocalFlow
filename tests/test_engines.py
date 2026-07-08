import numpy as np
import pytest

from localflow.config import Config
from localflow.engines import available_backends, create_engine
from localflow.engines.mock import MockEngine
from localflow.engines.registry import (
    bundled_models_dir,
    ggml_model_path,
    resolve_model_file,
)


class TestMockEngine:
    def test_canned_responses(self):
        engine = MockEngine(["one", "two"])
        a = np.zeros(16000, dtype=np.float32)
        assert engine.transcribe(a).text == "one"
        assert engine.transcribe(a).text == "two"
        assert engine.transcribe(a).text == "two"  # sticks at last

    def test_records_calls(self):
        engine = MockEngine()
        engine.transcribe(np.zeros(8000, dtype=np.float32), language="de", initial_prompt="Hi")
        assert engine.calls[0] == {"samples": 8000, "language": "de", "initial_prompt": "Hi"}


class TestRegistry:
    def test_mock_backend(self):
        config = Config()
        config.engine.backend = "mock"
        assert isinstance(create_engine(config), MockEngine)

    def test_unknown_backend_raises(self):
        config = Config()
        config.engine.backend = "quantum"
        with pytest.raises(ValueError):
            create_engine(config)

    def test_available_backends_returns_list(self):
        assert isinstance(available_backends(), list)

    def test_model_path_layout(self, tmp_path):
        config = Config()
        config.data_dir = str(tmp_path)
        path = ggml_model_path("tiny", config)
        assert path.name == "ggml-tiny.bin"
        assert str(tmp_path) in str(path)

    def test_whispercpp_with_explicit_model(self, model_path):
        pytest.importorskip("pywhispercpp")
        config = Config()
        config.engine.backend = "whisper.cpp"
        config.engine.model_path = str(model_path)
        engine = create_engine(config)
        assert engine.name == "whisper.cpp"


class TestBundledModels:
    """Standalone LocalFlow.app ships ggml weights in Resources/models;
    the stub advertises the dir via LOCALFLOW_RESOURCES."""

    def test_no_env_no_bundle(self, monkeypatch):
        monkeypatch.delenv("LOCALFLOW_RESOURCES", raising=False)
        assert bundled_models_dir() is None

    def test_bundle_dir_found(self, tmp_path, monkeypatch):
        (tmp_path / "models").mkdir()
        monkeypatch.setenv("LOCALFLOW_RESOURCES", str(tmp_path))
        assert bundled_models_dir() == tmp_path / "models"

    def test_resolve_prefers_data_dir_over_bundle(self, tmp_path, monkeypatch):
        config = Config()
        config.data_dir = str(tmp_path / "data")
        user_copy = ggml_model_path("base", config)
        user_copy.parent.mkdir(parents=True, exist_ok=True)
        user_copy.write_bytes(b"user")
        bundle = tmp_path / "resources" / "models"
        bundle.mkdir(parents=True)
        (bundle / "ggml-base.bin").write_bytes(b"bundled")
        monkeypatch.setenv("LOCALFLOW_RESOURCES", str(tmp_path / "resources"))
        assert resolve_model_file("base", config) == user_copy

    def test_resolve_falls_back_to_bundle(self, tmp_path, monkeypatch):
        config = Config()
        config.data_dir = str(tmp_path / "data")
        bundle = tmp_path / "resources" / "models"
        bundle.mkdir(parents=True)
        (bundle / "ggml-base.bin").write_bytes(b"bundled")
        monkeypatch.setenv("LOCALFLOW_RESOURCES", str(tmp_path / "resources"))
        assert resolve_model_file("base", config) == bundle / "ggml-base.bin"

    def test_resolve_missing_everywhere(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LOCALFLOW_RESOURCES", raising=False)
        config = Config()
        config.data_dir = str(tmp_path)
        assert resolve_model_file("base", config) is None


class TestNoiseAnnotations:
    def test_blank_audio_yields_empty(self):
        from localflow.engines.base import TranscriptionResult

        for noise in ("[BLANK_AUDIO]", "[ Silence ]", "(keyboard clacking)",
                      "*coughs*", "♪", "[BLANK_AUDIO] [BLANK_AUDIO]"):
            assert TranscriptionResult(noise).clean_text == "", noise

    def test_noise_stripped_from_real_speech(self):
        from localflow.engines.base import TranscriptionResult

        r = TranscriptionResult("Hello [BLANK_AUDIO] world (sighs)")
        assert r.clean_text == "Hello world"

    def test_legit_parentheses_kept(self):
        from localflow.engines.base import TranscriptionResult

        r = TranscriptionResult("I bought (organic) apples")
        assert r.clean_text == "I bought (organic) apples"
