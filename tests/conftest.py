from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from localflow.audio import load_wav

# A real ggml Whisper model for integration tests. Auto-discovered from the
# usual places; tests that need it skip cleanly when it's absent.
MODEL_CANDIDATES = [
    Path(__file__).parent / "models" / "ggml-tiny.bin",
    Path.home() / ".local" / "share" / "localflow" / "models" / "ggml-tiny.bin",
    Path("/tmp/claude-0/-home-user-Wispr-Flow-Clone/21b2b575-d287-5565-bfa0-95bbcac7482f/scratchpad/ggml-tiny.bin"),
]


def find_model() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists() and path.stat().st_size > 1_000_000:
            return path
    return None


def has_whispercpp() -> bool:
    try:
        import pywhispercpp  # noqa: F401

        return True
    except ImportError:
        return False


def has_espeak() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def synth_speech(text: str, out_path: Path, speed: int = 140) -> np.ndarray:
    """Generate spoken audio with espeak-ng and load it as 16 kHz float32."""
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    raw = out_path.with_suffix(".raw.wav")
    subprocess.run(
        [binary, "-v", "en-us", "-s", str(speed), "-w", str(raw), text],
        check=True,
        capture_output=True,
    )
    return load_wav(raw)


@pytest.fixture(autouse=True)
def _no_local_llm_autoprobe(monkeypatch):
    """Keep tests off any real LM Studio/Ollama running on this machine.

    LLM tests that want a server set an explicit base_url to a fake one."""
    monkeypatch.setattr("localflow.llm._PROBE_URLS", ())


@pytest.fixture(scope="session")
def model_path():
    path = find_model()
    if path is None:
        pytest.skip("no ggml whisper model available")
    return path


@pytest.fixture(scope="session")
def whisper_engine(model_path):
    if not has_whispercpp():
        pytest.skip("pywhispercpp not installed")
    from localflow.engines.whispercpp_engine import WhisperCppEngine

    return WhisperCppEngine(str(model_path), threads=4)


@pytest.fixture()
def speech(tmp_path):
    if not has_espeak():
        pytest.skip("espeak-ng not installed")

    def make(text: str, speed: int = 140) -> np.ndarray:
        return synth_speech(text, tmp_path / "speech.wav", speed=speed)

    return make
