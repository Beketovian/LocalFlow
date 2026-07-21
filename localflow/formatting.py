"""Transcript formatting - LocalFlow's equivalent of Wispr Flow's AI formatting.

Raw Whisper output is already punctuated, but dictation needs more:

* filler-word removal ("um", "uh", "er", ...)
* self-correction handling ("meet at five... no wait, six" -> "meet at six")
* spoken commands ("new line", "new paragraph", spoken punctuation)
* written-form numbers ("twenty three percent" -> "23%")
* spoken email addresses ("jo at gmail dot com" -> "jo@gmail.com")
* sentence capitalization and whitespace cleanup
* per-app tone via FormattingConfig overrides (see localflow.context)

Everything is deliberately rule-based and local; no cloud calls.
"""

from __future__ import annotations

import re
from dataclasses import replace as dc_replace
from typing import Dict, Optional

from .config import FormattingConfig

# --------------------------------------------------------------------------- fillers

# Standalone hesitation sounds. Kept conservative: words like "like"/"you know"
# are meaningful too often to strip safely without a language model.
_FILLERS = r"(?:um+|uh+|uhm+|erm+|er|ehm+|eh|hmm+|mmm+|mm-hmm|uh-huh)"
_FILLER_RE = re.compile(
    rf"(?:(?<=^)|(?<=[\s.,;:!?…—\"'\(\[])){_FILLERS}(?=$|[\s.,;:!?…—\"'\)\]])",
    re.IGNORECASE,
)


def remove_fillers(text: str) -> str:
    text = _FILLER_RE.sub("", text)
    # Tidy punctuation orphaned by the removal: " , like" / ", ," / leading commas
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])\s*(?:[,;:]\s*)+", r"\1 ", text)
    text = re.sub(r"(?:^|(?<=[.!?…]))\s*[,;:]\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


# ---------------------------------------------------------------- hallucinations

_HALLUCINATIONS = [
    r"thanks for watching",
    r"thank you for watching",
    r"i hope you enjoyed this video",
    r"i'll see you in the next one",
    r"subscribe to my channel",
]
_HALLUCINATION_RE = re.compile(
    r"\b(?:" + "|".join(_HALLUCINATIONS) + r")\b[,.!?]*\s*",
    re.IGNORECASE
)


def remove_hallucinations(text: str) -> str:
    text = _HALLUCINATION_RE.sub("", text)
    # Tidy orphaned punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])\s*(?:[,;:]\s*)+", r"\1 ", text)
    text = re.sub(r"(?:^|(?<=[.!?…]))\s*[,;:]\s*", " ", text)
    text = re.sub(r"\bbye\b\.?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


# ----------------------------------------------------------------- self-correction

# "…, no wait, X" / "… scratch that, X" / "… I mean X"
# Strategy: when a correction cue appears, drop the clause immediately before it.
_CORRECTION_CUES = [
    r"no,?\s+wait[,:]?",
    r"wait,?\s+no[,:]?",
    r"scratch\s+that[,:]?",
    r"actually[,:]?\s+no[,:]?",
    r"i\s+mean[,:]?",
    r"correction[,:]?",
    r"rather[,:]?",
    r"let\s+me\s+rephrase(?:\s+that)?[,:]?",
]
_CUE_RE = re.compile(
    r"[,;\s]*\b(?:" + "|".join(_CORRECTION_CUES) + r")\s+",
    re.IGNORECASE,
)
# Clause = last run of words after the previous clause boundary
_LAST_CLAUSE_RE = re.compile(r"[^,.;:!?\n]+$")


def apply_self_corrections(text: str) -> str:
    """Drop the clause preceding an explicit correction cue.

    "Send it to John, no wait, to Sarah." -> "Send it to Sarah."
    Only the immediately preceding clause is removed, so the risk of eating
    real content is small; cues in the middle of a clause keep the sentence
    prefix ("meet at five, I mean six" keeps "meet at ").
    """

    while True:
        match = _CUE_RE.search(text)
        if not match:
            return text
        before = text[: match.start()].rstrip()
        after = text[match.end():]

        clause = _LAST_CLAUSE_RE.search(before)
        if clause and re.search(r"\w", clause.group(0)):
            # Mid-sentence cue: drop the clause before it, keeping any shared
            # sentence prefix - if the corrected text repeats one of the
            # dropped clause's leading words, keep the part before it.
            prefix = before[: clause.start()]
            dropped = clause.group(0)
            kept = _common_stem(dropped, after)
        else:
            # Cue right after a sentence boundary ("... at five. No wait, six")
            # retracts the whole previous sentence.
            chunks = re.split(r"(?<=[.!?…])\s+", before)
            dropped = chunks[-1] if chunks else ""
            prefix = before[: len(before) - len(dropped)]
            kept = ""
        joined = prefix
        if joined and not joined.endswith((" ", "\n")) and (kept or after):
            joined += " "
        text = joined + kept + after


def _common_stem(dropped: str, after: str) -> str:
    """Return the part of the dropped clause to keep as a shared prefix.

    "meet at five" + "six o'clock" -> keeps "meet at " (no shared word,
    but "five" is dropped, and words before the last content word survive
    only when the correction is a single-word swap).
    """

    d_words = dropped.split()
    a_first = (after.split() or [""])[0].strip(".,;:!?").lower()
    if not d_words:
        return ""
    # Single-word swap: "five" -> "six"; keep everything but the last word
    # when the correction begins with a word of the same category (crude:
    # both are not function words) and the clause has more than one word.
    if len(d_words) > 1 and a_first:
        # If the correction re-states one of the dropped words, keep up to it:
        lowered = [w.strip(".,;:!?").lower() for w in d_words]
        if a_first in lowered:
            idx = lowered.index(a_first)
            return "".join(w + " " for w in d_words[:idx])
        function_words = {"a", "an", "the", "to", "at", "in", "on", "of", "for",
                          "with", "and", "or", "is", "are", "was", "it", "that"}
        if a_first not in function_words:
            return "".join(w + " " for w in d_words[:-1])
    return ""


# ----------------------------------------------------------------- spoken commands

# Structural commands are safe defaults; Wispr Flow honours these too.
_COMMANDS: Dict[str, str] = {
    "new line": "\n",
    "newline": "\n",
    "next line": "\n",
    "new paragraph": "\n\n",
    "next paragraph": "\n\n",
    "tab key": "\t",
    # Lists: "bullet point apples, bullet point bananas" -> "- apples\n- bananas"
    "bullet point": "\n- ",
    "new bullet": "\n- ",
    "next bullet": "\n- ",
}

# Spoken punctuation (opt-in; Whisper usually punctuates already)
_PUNCTUATION_WORDS: Dict[str, str] = {
    "period": ".",
    "full stop": ".",
    "comma": ",",
    "question mark": "?",
    "exclamation mark": "!",
    "exclamation point": "!",
    "colon": ":",
    "semicolon": ";",
    "semi colon": ";",
    "dash": " - ",
    "hyphen": "-",
    "ellipsis": "...",
    "open quote": '"',
    "close quote": '"',
    "open paren": "(",
    "close paren": ")",
    "open parenthesis": "(",
    "close parenthesis": ")",
    "at sign": "@",
    "ampersand": "&",
    "underscore": "_",
    "forward slash": "/",
    "backslash": "\\",
}


def _phrase_re(phrase: str) -> re.Pattern:
    inner = r"[\s]+".join(re.escape(w) for w in phrase.split())
    return re.compile(rf"[,.]?\s*\b{inner}\b[,.]?\s*", re.IGNORECASE)


_COMMAND_RES = [(_phrase_re(k), v) for k, v in sorted(_COMMANDS.items(), key=lambda kv: -len(kv[0]))]


def apply_spoken_commands(text: str) -> str:
    for pattern, repl in _COMMAND_RES:
        if repl.startswith("\n"):
            text = pattern.sub(lambda m, r=repl: r, text)
        else:
            text = pattern.sub(repl, text)
    # Whisper sometimes capitalizes after our inserted newlines oddly; strip
    # spaces around newlines introduced above ("- " markers are untouched:
    # the dash sits right after the newline, so nothing strips inside them).
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # A dictation that *starts* with "bullet point" shouldn't open blank.
    return text.lstrip("\n")


_PUNCT_RES = [(_phrase_re(k), v) for k, v in sorted(_PUNCTUATION_WORDS.items(), key=lambda kv: -len(kv[0]))]


def apply_spoken_punctuation(text: str) -> str:
    for pattern, repl in _PUNCT_RES:
        text = pattern.sub(lambda m, r=repl: r + " ", text)
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


# ------------------------------------------------------------------------ numbers

_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_NUMBER_WORD = re.compile(
    r"\b(?:(?:" + "|".join(list(_UNITS) + list(_TENS) + list(_SCALES)) +
    r")(?:[\s-]+(?:" + "|".join(list(_UNITS) + list(_TENS) + list(_SCALES) + ["and"]) + r"))*)\b",
    re.IGNORECASE,
)


def _words_to_number(words: str) -> Optional[int]:
    total, current = 0, 0
    tokens = re.split(r"[\s-]+", words.lower())
    seen_any = False
    for tok in tokens:
        if tok == "and":
            continue
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            current = max(current, 1) * 100
        elif tok in _SCALES:
            total += max(current, 1) * _SCALES[tok]
            current = 0
        else:
            return None
        seen_any = True
    return (total + current) if seen_any else None


def convert_numbers(text: str) -> str:
    """Convert multi-word spelled-out numbers to digits.

    Single-word small numbers ("one", "two", "ten") are left alone - "I have
    one idea" should not become "I have 1 idea". Multi-word numbers are nearly
    always meant as figures when dictating.
    """

    def repl(match: re.Match) -> str:
        phrase = match.group(0)
        if not re.search(r"[\s-]", phrase.strip()):
            return phrase  # single word - leave prose alone
        value = _words_to_number(phrase)
        if value is None:
            return phrase
        return str(value)

    text = _NUMBER_WORD.sub(repl, text)
    text = re.sub(r"\b(\d+)\s*percent\b", r"\1%", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)\s+dollars?\b", r"$\1", text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------------- emails

_EMAIL_RE = re.compile(
    r"\b([a-z0-9]+(?:[\s]+(?:dot|dash|underscore)[\s]+[a-z0-9]+)*)"
    r"[\s]+at[\s]+"
    r"([a-z0-9]+(?:[\s]+dot[\s]+[a-z0-9]+)+)\b",
    re.IGNORECASE,
)
_SEP = {"dot": ".", "dash": "-", "underscore": "_"}


def format_emails(text: str) -> str:
    def repl(match: re.Match) -> str:
        def join(part: str) -> str:
            tokens = re.split(r"\s+", part.strip())
            out = ""
            for tok in tokens:
                out += _SEP.get(tok.lower(), tok)
            return out

        return f"{join(match.group(1))}@{join(match.group(2))}"

    return _EMAIL_RE.sub(repl, text)


# ----------------------------------------------------------------- capitalization

_SENTENCE_START = re.compile(r"(^|[.!?…]\s+|\n)([a-zà-ÿ])")


def capitalize_sentences(text: str) -> str:
    text = _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\bi\b(?!\.[a-z])", "I", text)  # lookahead spares "i.e."
    text = re.sub(r"\bi'(m|ve|ll|d)\b", lambda m: "I'" + m.group(1), text)
    return text


# ---------------------------------------------------------------------- the pipe


def format_transcript(
    text: str,
    config: Optional[FormattingConfig] = None,
    overrides: Optional[Dict] = None,
) -> str:
    """Run the full formatting pipeline over a raw transcript."""

    cfg = config or FormattingConfig()
    if overrides:
        cfg = dc_replace(cfg, **{k: v for k, v in overrides.items() if hasattr(cfg, k)})

    if not cfg.enabled:
        return text.strip()

    if cfg.apply_self_corrections:
        text = apply_self_corrections(text)
    if cfg.remove_fillers:
        text = remove_fillers(text)
    text = remove_hallucinations(text)
    if cfg.spoken_commands:
        text = apply_spoken_commands(text)
    if cfg.spoken_punctuation:
        text = apply_spoken_punctuation(text)
    if cfg.convert_numbers:
        text = convert_numbers(text)
    if cfg.format_emails:
        text = format_emails(text)
    if cfg.collapse_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n +", "\n", text)
        text = text.strip()
    if cfg.capitalize_sentences:
        text = capitalize_sentences(text)
    if cfg.ensure_terminal_punctuation and text and text[-1] not in ".!?…:\n":
        text += "."
    return text


# ------------------------------------------------------------- voice actions

# "... send it" at the very end of a dictation -> paste, then press Enter.
# Deliberately strict: only explicit forms, only at the end, so sentences
# that merely *mention* sending never trigger ("what did he send" is safe).
_SEND_RE = re.compile(
    r"(?:^|[\s,;:.!?])(?:and\s+|then\s+)?send\s+(?:it|that|this|now|(?:the\s+)?message)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)


def detect_send_command(text: str) -> tuple:
    """Return (text_without_command, should_send)."""
    m = _SEND_RE.search(text)
    if not m:
        return text, False
    return text[: m.start()].rstrip(" ,;:.!?") or "", True


def smart_join(previous: str, new: str) -> str:
    """Join freshly dictated text onto previously injected text.

    Decides whether a space is needed and whether `new` should start lowercase
    (continuing a sentence) - used when the user dictates in several bursts.
    """

    if not previous:
        return new
    if not new:
        return ""
    prev_tail = previous.rstrip()
    if not prev_tail:
        return new
    if prev_tail[-1] in ".!?…\n":
        return new  # new sentence, keep capitalization
    # Continuing a sentence: lowercase the first letter unless it looks like
    # a proper noun kept capitalized mid-sentence by Whisper ("I", acronyms).
    first = new[:1]
    rest = new[1:]
    word = new.split()[0] if new.split() else ""
    if first.isupper() and word not in ("I", "I'm", "I'll", "I've", "I'd") and not word.isupper():
        new = first.lower() + rest
    return new
