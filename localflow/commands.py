"""Command mode - speak an instruction, transform the selected text.

Wispr Flow's command mode sends the instruction to an LLM. LocalFlow ships a
local rule-based editor covering the common commands, plus an optional hook
for any OpenAI-compatible endpoint (e.g. a local Ollama) for free-form edits.
"""

from __future__ import annotations

import json
import re
import textwrap
import urllib.request
from typing import Callable, Dict, Optional

from .formatting import capitalize_sentences, remove_fillers

Transform = Callable[[str], str]


def _bulleted(text: str) -> str:
    parts = [p.strip() for p in re.split(r"[,\n;]+|\band\b", text) if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return "\n".join(f"- {p.rstrip('.').strip()}" for p in parts)


def _numbered(text: str) -> str:
    bullets = _bulleted(text).split("\n")
    return "\n".join(f"{i + 1}. {b[2:]}" for i, b in enumerate(bullets))


def _one_line(text: str) -> str:
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _snake(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return "_".join(w.lower() for w in words)


def _camel(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return text
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def _kebab(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return "-".join(w.lower() for w in words)


def _fix_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])(?=[A-Za-z])", r"\1 ", text)
    text = capitalize_sentences(text)
    if text and text[-1] not in ".!?…:\n":
        text += "."
    return text


def _shorten(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return text
    keep = max(1, len(sentences) // 2)
    return " ".join(sentences[:keep])


# instruction pattern -> transform
_RULES: Dict[str, Transform] = {
    r"\b(upper\s*case|all\s+caps|capitalize\s+everything)\b": str.upper,
    r"\b(lower\s*case)\b": str.lower,
    r"\btitle\s*case\b": lambda t: t.title(),
    r"\bsentence\s*case\b": capitalize_sentences,
    r"\b(bullet(ed)?\s*(list|points)?|list\s+of\s+bullets)\b": _bulleted,
    r"\bnumbered\s*list\b": _numbered,
    r"\b(one|single)\s+line\b": _one_line,
    r"\bsnake\s*case\b": _snake,
    r"\bcamel\s*case\b": _camel,
    r"\b(kebab|dash)\s*case\b": _kebab,
    r"\b(fix|correct|clean)\s*(up)?\s*(the)?\s*(punctuation|grammar|formatting)?\b": _fix_punctuation,
    r"\bremove\s+(the\s+)?fillers?\b": remove_fillers,
    r"\b(shorten|make\s+(it|this)\s+shorter|more\s+concise)\b": _shorten,
    r"\b(quote|wrap\s+in\s+quotes)\b": lambda t: f'"{t.strip()}"',
    r"\breverse\b": lambda t: t[::-1],
}


class CommandProcessor:
    def __init__(
        self,
        llm_url: Optional[str] = None,
        llm_model: str = "llama3.2",
        llm_api_key: Optional[str] = None,
    ) -> None:
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key

    def apply(self, instruction: str, text: str) -> Optional[str]:
        """Transform `text` per the spoken `instruction`.

        Returns None when the instruction isn't understood (callers can then
        surface an error rather than mangling the selection).
        """

        instruction = instruction.strip().lower()
        if not instruction or not text:
            return None
        for pattern, transform in _RULES.items():
            if re.search(pattern, instruction):
                return transform(text)
        if self.llm_url:
            return self._ask_llm(instruction, text)
        return None

    def known_commands(self) -> str:
        return textwrap.dedent(
            """\
            uppercase / lowercase / title case / sentence case
            bullet list / numbered list / one line
            snake case / camel case / kebab case
            fix punctuation / remove fillers / shorten
            wrap in quotes"""
        )

    def _ask_llm(self, instruction: str, text: str) -> Optional[str]:
        """Free-form edit via any OpenAI-compatible chat endpoint."""
        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You edit text. Apply the user's instruction to the text "
                        "and return ONLY the edited text, no commentary."
                    ),
                },
                {"role": "user", "content": f"Instruction: {instruction}\n\nText:\n{text}"},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            self.llm_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
