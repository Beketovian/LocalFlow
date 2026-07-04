import numpy as np
import pytest

from localflow.config import Config
from localflow.engines import available_backends, create_engine
from localflow.engines.mock import MockEngine
from localflow.engines.registry import ggml_model_path


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
