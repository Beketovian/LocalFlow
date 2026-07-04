"""Personal dictionary - names, jargon and snippets Whisper should get right.

Two mechanisms, mirroring Wispr Flow:

1. *Boosting*: dictionary entries are fed to Whisper as the initial prompt so
   the decoder is biased toward them ("Wispr", "kubectl", "Anthropic"...).
2. *Correction*: after transcription, tokens that are close misspellings of a
   dictionary entry are replaced with the canonical spelling.

Plus plain text replacements ("brb" -> "be right back", "eta" -> "ETA").
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, Iterable, List, Optional


class PersonalDictionary:
    def __init__(
        self,
        words: Optional[Iterable[str]] = None,
        replacements: Optional[Dict[str, str]] = None,
    ) -> None:
        self.words: List[str] = list(dict.fromkeys(words or []))  # keep order, dedupe
        self.replacements: Dict[str, str] = dict(replacements or {})

    # ------------------------------------------------------------- management

    def add(self, word: str) -> None:
        word = word.strip()
        if word and word not in self.words:
            self.words.append(word)

    def remove(self, word: str) -> bool:
        try:
            self.words.remove(word)
            return True
        except ValueError:
            return False

    def add_replacement(self, spoken: str, written: str) -> None:
        self.replacements[spoken.strip()] = written

    # --------------------------------------------------------------- boosting

    def initial_prompt(self, max_words: int = 60) -> Optional[str]:
        """Prompt bias for Whisper. Returns None when the dictionary is empty."""
        if not self.words:
            return None
        listed = ", ".join(self.words[:max_words])
        return f"Glossary: {listed}."

    # ------------------------------------------------------------- correction

    def correct(self, text: str, cutoff: float = 0.82) -> str:
        """Replace near-miss tokens with their canonical dictionary spelling.

        Only single-token entries are fuzzy-matched; multi-word entries are
        matched case-insensitively as phrases.
        """

        if not self.words:
            return self.apply_replacements(text)

        single = [w for w in self.words if " " not in w]
        multi = [w for w in self.words if " " in w]

        # Phrase entries: fix capitalization/spelling of the whole phrase
        for phrase in multi:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub(phrase, text)

        if single:
            lowered = {w.lower(): w for w in single}

            def fix_token(match: re.Match) -> str:
                token = match.group(0)
                low = token.lower()
                if low in lowered:  # exact hit, fix casing only
                    canonical = lowered[low]
                    # Preserve an all-caps or capitalized style the user typed?
                    # Canonical spelling wins: that's the point of a dictionary.
                    return canonical
                if len(token) < 4:
                    return token  # too short to fuzzy-match safely
                close = difflib.get_close_matches(low, lowered.keys(), n=1, cutoff=cutoff)
                if close:
                    return lowered[close[0]]
                return token

            text = re.sub(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]*", fix_token, text)

        return self.apply_replacements(text)

    def apply_replacements(self, text: str) -> str:
        for spoken, written in self.replacements.items():
            pattern = re.compile(rf"\b{re.escape(spoken)}\b", re.IGNORECASE)
            text = pattern.sub(written, text)
        return text
