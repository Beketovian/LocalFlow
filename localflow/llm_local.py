"""In-process LLM engine - Apple MLX, no server required.

This is what lets LocalFlow ship as a single application: instead of talking
to LM Studio/Ollama over HTTP, the model runs inside the daemon process.
MLX is the same engine LM Studio uses on Apple Silicon, so speed is
identical (measured: ~0.3s per dictation cleanup with Qwen3-4B 4-bit).

The engine is lazy: nothing heavy is imported or loaded until the first
request, and a missing mlx-lm install just means "embedded backend
unavailable" rather than an import error at startup.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional

# The model the bake-off picked: fastest cleanup with full quality.
DEFAULT_REPO = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
DEFAULT_MODEL_NAME = "qwen3-4b-instruct-2507"

# Directories where LM Studio keeps MLX models; reused when present so
# users don't download the same weights twice.
_LMSTUDIO_MODEL_ROOTS = (
    Path.home() / ".lmstudio" / "models",
    Path.home() / ".cache" / "lm-studio" / "models",
)


def default_llm_dir(data_dir: Path) -> Path:
    return data_dir / "models" / "llm"


def _looks_like_mlx_model(path: Path) -> bool:
    return (path / "config.json").exists() and any(path.glob("*.safetensors"))


def find_local_model(data_dir: Path, model_path: Optional[str] = None,
                     model: str = "auto") -> Optional[Path]:
    """Locate MLX model weights on disk. Search order:

    1. an explicit configured path
    2. LocalFlow's own model dir (populated by `localflow llm download`)
    3. LM Studio's model dirs (matching the configured model name)
    """
    if model_path:
        p = Path(model_path).expanduser()
        if _looks_like_mlx_model(p):
            return p
        return None

    own = default_llm_dir(data_dir)
    if own.is_dir():
        for child in sorted(own.iterdir()):
            if child.is_dir() and _looks_like_mlx_model(child):
                return child

    wanted = None if model == "auto" else model.lower()
    for root in _LMSTUDIO_MODEL_ROOTS:
        if not root.is_dir():
            continue
        for publisher in sorted(root.iterdir()):
            if not publisher.is_dir():
                continue
            for child in sorted(publisher.iterdir()):
                if not (child.is_dir() and _looks_like_mlx_model(child)):
                    continue
                name = child.name.lower()
                if wanted is not None:
                    if wanted in name or name in wanted:
                        return child
                elif DEFAULT_MODEL_NAME.replace("-", "") in name.replace("-", ""):
                    return child
    return None


class EmbeddedEngine:
    """Thread-safe in-process chat engine over mlx-lm."""

    def __init__(self, model_dir: Path, temperature: float = 0.2) -> None:
        self.model_dir = Path(model_dir)
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()  # MLX generation is not concurrent-safe
        # KV cache reused across calls. The chat prompt starts with a large
        # constant prefix (system rules + few-shot examples); re-prefilling
        # it every dictation costs ~1s, reusing it makes cleanup ~0.4s.
        self._cache = None
        self._cached_tokens: list = []

    @property
    def name(self) -> str:
        return self.model_dir.name.lower()

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from mlx_lm import load  # heavy import, kept off startup

            self._model, self._tokenizer = load(str(self.model_dir))

    def chat(self, messages: list, max_tokens: int = 512) -> str:
        """Run one chat completion. Raises ValueError when the output hits
        the token cap (truncated text must never be pasted)."""
        self.ensure_loaded()
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import (
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )
        from mlx_lm.sample_utils import make_sampler

        with self._lock:
            tokens = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            # Reuse the KV cache for whatever prefix matches the last call
            # (usually everything up to the new dictation), trimming off the
            # part that diverged - including the previous generation.
            if self._cache is None or not can_trim_prompt_cache(self._cache):
                self._cache = make_prompt_cache(self._model)
                self._cached_tokens = []
            common = 0
            for cached, new in zip(self._cached_tokens, tokens):
                if cached != new:
                    break
                common += 1
            common = min(common, len(tokens) - 1)  # always feed >= 1 token
            if len(self._cached_tokens) > common:
                trim_prompt_cache(self._cache, len(self._cached_tokens) - common)
                self._cached_tokens = self._cached_tokens[:common]
            text = []
            generated = []
            last = None
            for resp in stream_generate(
                self._model,
                self._tokenizer,
                prompt=tokens[common:],
                max_tokens=max_tokens,
                sampler=make_sampler(temp=self.temperature),
                prompt_cache=self._cache,
            ):
                text.append(resp.text)
                generated.append(resp.token)
                last = resp
            self._cached_tokens = list(tokens) + generated
        if last is not None and last.finish_reason == "length":
            raise ValueError("llm response truncated")
        return "".join(text)


def download_default_model(data_dir: Path, progress=None) -> Path:
    """Fetch the default model from Hugging Face into LocalFlow's data dir.

    ~2.3 GB; called by `localflow llm download` and the dashboard's
    "Download built-in model" button. `progress` (optional) receives human
    strings, with an overall percentage while the weights stream in.
    """
    dest = default_llm_dir(data_dir) / DEFAULT_MODEL_NAME
    dest.mkdir(parents=True, exist_ok=True)
    # blobs=true adds per-file sizes, which is what makes a real % possible
    api = f"https://huggingface.co/api/models/{DEFAULT_REPO}?blobs=true"
    with urllib.request.urlopen(api, timeout=30) as resp:
        siblings = json.loads(resp.read())["siblings"]
    files = [(s["rfilename"], s.get("size") or 0) for s in siblings
             if not s["rfilename"].startswith(".") and s["rfilename"] != "README.md"]
    total = sum(size for _, size in files) or 1
    done = 0
    for name, size in files:
        target = dest / name
        url = f"https://huggingface.co/{DEFAULT_REPO}/resolve/main/{name}"
        have = target.stat().st_size if target.exists() else -1
        if have == size or (not size and have > 0):
            done += size
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
            file_done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                file_done += len(chunk)
                if progress and size > 1 << 22:  # only weights merit a %
                    progress(f"downloading {name}... "
                             f"{(done + file_done) / total:.0%}")
        tmp.rename(target)
        done += size
    if progress:
        progress(f"model ready: {dest}")
    return dest
