"""Configuration for LocalFlow.

Config lives in a single JSON file (default: ~/.config/localflow/config.json)
and is represented in code as nested dataclasses so every module gets typed
access with sensible defaults.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "LocalFlow"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "localflow"


def default_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "LocalFlow"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "localflow"


@dataclass
class EngineConfig:
    """Which speech-to-text backend to use and how."""

    # "auto" picks the best available: faster-whisper, then whisper.cpp
    backend: str = "auto"  # auto | faster-whisper | whisper.cpp | mock
    # Model size/name: tiny, base, small, medium, large-v3, distil-*, or a path
    model: str = "base"
    # Explicit path to a model file/dir; overrides `model` when set
    model_path: Optional[str] = None
    # BCP-47-ish language code ("en", "de", ...) or "auto" for detection
    language: str = "auto"
    device: str = "cpu"  # cpu | cuda | auto
    compute_type: str = "int8"  # for faster-whisper: int8 | int8_float16 | float16 ...
    threads: int = 4
    # Beam size 1 = greedy (fastest); >1 trades latency for accuracy
    beam_size: int = 1


@dataclass
class HotkeyConfig:
    """Global hotkeys. Combos use pynput syntax."""

    # Hold to talk; release to transcribe and inject
    push_to_talk: str = "<ctrl>+<space>"
    # Toggle hands-free dictation (auto-stops on silence when enabled)
    toggle_dictation: str = "<ctrl>+<shift>+<space>"
    # Command mode: speak an instruction to transform selected text
    command_mode: str = "<ctrl>+<alt>+<space>"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    input_device: Optional[str] = None  # None = system default
    # Boost quiet speech (whispering) to a target RMS level
    normalize: bool = True
    target_rms: float = 0.06
    # Play soft feedback tones when recording starts/stops
    feedback_sounds: bool = True
    # Hands-free mode: stop after this many seconds of trailing silence
    silence_stop_after: float = 1.6
    # Energy threshold used by the simple VAD (fraction of full scale RMS)
    vad_threshold: float = 0.012
    max_recording_seconds: float = 120.0


@dataclass
class FormattingConfig:
    """The 'AI formatting' layer applied to raw transcripts."""

    enabled: bool = True
    remove_fillers: bool = True
    apply_self_corrections: bool = True
    spoken_commands: bool = True  # "new line", "new paragraph", ...
    spoken_punctuation: bool = False  # "period", "comma" -> symbols (off: Whisper already punctuates)
    capitalize_sentences: bool = True
    ensure_terminal_punctuation: bool = False
    convert_numbers: bool = True  # "twenty three" -> "23"
    format_emails: bool = True  # "foo at bar dot com" -> "foo@bar.com"
    collapse_whitespace: bool = True


@dataclass
class OutputConfig:
    # How to put text into the focused app: auto | type | clipboard | stdout | none
    method: str = "auto"
    # Delay between simulated keystrokes (seconds); 0 is fastest
    type_interval: float = 0.0
    # Restore previous clipboard contents after clipboard-paste injection
    restore_clipboard: bool = True
    trailing_space: bool = True


@dataclass
class AppProfile:
    """Per-application formatting overrides (Wispr Flow's 'context awareness').

    `match` patterns are matched case-insensitively against the active window's
    title and app/class name.
    """

    name: str = "default"
    match: List[str] = field(default_factory=list)
    tone: str = "auto"  # auto | casual | formal | code
    overrides: Dict[str, Any] = field(default_factory=dict)  # FormattingConfig fields


def default_profiles() -> List[AppProfile]:
    return [
        AppProfile(
            name="terminal",
            match=["terminal", "konsole", "gnome-terminal", "alacritty", "kitty",
                   "iterm", "xterm", "wezterm", "cmd.exe", "powershell"],
            tone="code",
            overrides={
                "capitalize_sentences": False,
                "ensure_terminal_punctuation": False,
                "convert_numbers": True,
            },
        ),
        AppProfile(
            name="code-editor",
            match=["visual studio code", "vscode", "code - ", "intellij", "pycharm",
                   "sublime", "neovim", "vim", "emacs", "zed"],
            tone="code",
            overrides={"ensure_terminal_punctuation": False},
        ),
        AppProfile(
            name="chat",
            match=["slack", "discord", "telegram", "whatsapp", "signal", "messages",
                   "imessage"],
            tone="casual",
            overrides={"ensure_terminal_punctuation": False},
        ),
        AppProfile(
            name="email",
            match=["gmail", "outlook", "thunderbird", "mail", "superhuman"],
            tone="formal",
            overrides={"ensure_terminal_punctuation": True},
        ),
        AppProfile(
            name="docs",
            match=["google docs", "word", "notion", "obsidian", "libreoffice"],
            tone="formal",
            overrides={"ensure_terminal_punctuation": True},
        ),
    ]


@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 5170


@dataclass
class Config:
    engine: EngineConfig = field(default_factory=EngineConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    profiles: List[AppProfile] = field(default_factory=default_profiles)
    # Personal dictionary: words/phrases Whisper should get right
    dictionary: List[str] = field(default_factory=list)
    # Text replacements applied after transcription, e.g. {"eta": "ETA"}
    replacements: Dict[str, str] = field(default_factory=dict)
    # Store history in SQLite (set False for fully ephemeral use)
    save_history: bool = True
    data_dir: Optional[str] = None
    # Shown in the dashboard greeting ("Good morning, ...")
    user_name: str = ""

    # ------------------------------------------------------------------ io

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir) if self.data_dir else default_data_dir()

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls()
        for section_name, section_cls in (
            ("engine", EngineConfig),
            ("hotkeys", HotkeyConfig),
            ("audio", AudioConfig),
            ("formatting", FormattingConfig),
            ("output", OutputConfig),
            ("dashboard", DashboardConfig),
        ):
            raw = data.get(section_name)
            if isinstance(raw, dict):
                section = getattr(cfg, section_name)
                for key, value in raw.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        if isinstance(data.get("profiles"), list):
            cfg.profiles = []
            for p in data["profiles"]:
                prof = AppProfile()
                for key, value in p.items():
                    if hasattr(prof, key):
                        setattr(prof, key, value)
                cfg.profiles.append(prof)
        for key in ("dictionary", "replacements", "save_history", "data_dir", "user_name"):
            if key in data:
                setattr(cfg, key, data[key])
        return cfg

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or default_config_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        path = path or default_config_dir() / "config.json"
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
        return cls()
