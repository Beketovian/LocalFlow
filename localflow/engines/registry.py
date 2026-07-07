"""Engine selection and model management."""

from __future__ import annotations

import importlib.util
import urllib.request
from pathlib import Path
from typing import List, Optional

from ..config import Config, default_data_dir
from .base import STTEngine

# ggml model download sources for the whisper.cpp backend, tried in order.
GGML_SOURCES = [
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin",
    # Mirror that is reachable from some restricted networks:
    "https://raw.githubusercontent.com/Macoron/whisper.unity/master/Assets/StreamingAssets/Whisper/ggml-{name}.bin",
]


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> List[str]:
    found = []
    if _has("faster_whisper"):
        found.append("faster-whisper")
    if _has("pywhispercpp"):
        found.append("whisper.cpp")
    return found


def models_dir(config: Optional[Config] = None) -> Path:
    base = config.resolved_data_dir() if config else default_data_dir()
    d = base / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def validate_model_name(name: str) -> str:
    """Model names are URL/path components; reject anything decorative
    (a UI once leaked a checkmark suffix into the config)."""
    import re

    cleaned = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        raise ValueError(f"invalid model name: {name!r}")
    return cleaned


def ggml_model_path(name: str, config: Optional[Config] = None) -> Path:
    return models_dir(config) / f"ggml-{validate_model_name(name)}.bin"


def download_ggml_model(name: str, config: Optional[Config] = None, quiet: bool = False,
                        progress=None) -> Path:
    """Download a ggml model for whisper.cpp, trying each source in order.

    `progress` (optional) receives a completion fraction in [0, 1].
    """
    dest = ggml_model_path(name, config)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    hook = None
    if progress is not None:
        def hook(blocks, block_size, total):  # noqa: ANN001
            if total > 0:
                progress(min(1.0, blocks * block_size / total))
    last_error: Optional[Exception] = None
    for template in GGML_SOURCES:
        url = template.format(name=name)
        try:
            if not quiet:
                print(f"Downloading {url} ...")
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp, reporthook=hook)  # noqa: S310 - fixed hosts
            if tmp.stat().st_size < 1_000_000:
                tmp.unlink(missing_ok=True)
                raise IOError("downloaded file suspiciously small")
            tmp.rename(dest)
            return dest
        except Exception as exc:  # try next mirror
            last_error = exc
    raise RuntimeError(f"Could not download ggml model '{name}': {last_error}")


def create_engine(config: Config, progress=None) -> STTEngine:
    """Build the STT engine described by config.engine.

    backend="auto" prefers faster-whisper (accuracy) and falls back to
    whisper.cpp; a ready-to-use local ggml model beats a backend that would
    need a network download.
    """

    eng = config.engine
    backend = eng.backend

    if backend == "mock":
        from .mock import MockEngine

        return MockEngine()

    if backend == "auto":
        installed = available_backends()
        if not installed:
            raise RuntimeError(
                "No STT backend installed. Run: pip install localflow[fasterwhisper] "
                "or pip install localflow[whispercpp]"
            )
        # Prefer whisper.cpp when a ggml model is already on disk (offline-first)
        if (
            "whisper.cpp" in installed
            and (eng.model_path or ggml_model_path(eng.model, config).exists())
        ):
            backend = "whisper.cpp"
        else:
            backend = installed[0]

    if backend == "faster-whisper":
        from .faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(
            model=eng.model_path or eng.model,
            device=eng.device,
            compute_type=eng.compute_type,
            threads=eng.threads,
            beam_size=eng.beam_size,
        )

    if backend == "whisper.cpp":
        from .whispercpp_engine import WhisperCppEngine

        model_path = eng.model_path
        if not model_path:
            path = ggml_model_path(eng.model, config)
            if not path.exists():
                path = download_ggml_model(eng.model, config, progress=progress)
            model_path = str(path)
        return WhisperCppEngine(
            model_path=model_path,
            threads=eng.threads,
            beam_size=eng.beam_size,
        )

    raise ValueError(f"Unknown STT backend: {backend!r}")
