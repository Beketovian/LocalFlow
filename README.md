# LocalFlow

**A fully local, open-source voice dictation app — a [Wispr Flow](https://wisprflow.ai) clone powered by [OpenAI Whisper](https://github.com/openai/whisper).**

Hold a hotkey, speak, release: your words appear — cleaned up and formatted — in whatever app has focus. Everything runs on your machine. No cloud, no account, no audio ever leaves your computer.

![LocalFlow dashboard](docs/screenshots/ui_home.png)

While you dictate, the floating pill shows a live waveform, then pulses while Whisper formats your words:

![Recording pill - recording](docs/screenshots/pill-recording.png)
![Recording pill - processing](docs/screenshots/pill-processing.png)

## Features

| Wispr Flow | LocalFlow |
|---|---|
| Push-to-talk dictation into any app | ✅ hold a global hotkey, release to insert text at your cursor |
| Hands-free mode | ✅ toggle on; recording auto-stops on silence (built-in VAD) |
| AI formatting | ✅ two layers: a local rule engine (filler-word removal, self-corrections "at five — no wait, six" → "at six", spoken commands, spelled-out numbers → digits, spoken emails → `name@domain.com`, capitalization) **plus an optional local-LLM cleanup pass** — LocalFlow auto-detects LM Studio or Ollama and uses whatever chat model you have loaded; falls back to rules when no server is running |
| Personal dictionary | ✅ names/jargon are fed to Whisper as a bias prompt **and** fuzzy-corrected after transcription; plus text replacements ("omw" → "on my way") |
| Context awareness / app-aware tone | ✅ per-app profiles (terminal, code editor, chat, email, docs) adjust capitalization & punctuation automatically |
| Command mode (edit selected text by voice) | ✅ local commands (uppercase, bullet list, snake case, fix punctuation, shorten, …) + free-form edits ("translate to french", "make it friendlier") via the same local LLM |
| Whisper-quiet speech | ✅ automatic RMS gain normalization rescues very quiet audio |
| 100+ languages | ✅ multilingual Whisper models with auto-detection or a pinned language |
| Live preview while speaking | ✅ pseudo-streaming partial transcripts via background re-transcription |
| Floating recording pill with live waveform | ✅ frameless always-on-top capsule (tkinter): coral dot + waveform while recording, pulsing dots while formatting |
| Dashboard app (greeting, streak, WPM, time saved, activity) | ✅ the full Flow-style app: Home / History / Dictionary / Settings with live-saving toggles (`localflow ui`) |
| History, streaks, WPM stats | ✅ local SQLite + web dashboard (`http://127.0.0.1:5170`) |
| Privacy | ✅ **stronger**: 100% offline — Wispr Flow sends audio to the cloud, LocalFlow never does |

## Installation

```bash
# from a clone of this repo
pip install -e ".[whispercpp,desktop]"      # whisper.cpp engine (light, CPU-friendly)
# and/or
pip install -e ".[fasterwhisper,desktop]"   # faster-whisper engine (best accuracy)

# download a model + write default config
localflow setup --model base    # tiny | base | small | medium | large-v3

# check that everything is wired up
localflow doctor
```

Platform notes:

* **Linux (X11)**: install `xdotool` for the most reliable text injection; `portaudio19-dev` may be needed for `sounddevice`.
* **macOS**: grant Accessibility + Microphone permissions to your terminal (System Settings → Privacy & Security). Text is inserted by pasting (pbcopy + a synthesized ⌘V, like Wispr Flow) — simulated typing is unreliable in Messages/Slack/Electron apps.
* **Windows**: works out of the box with `pip install`.

## Usage

```bash
localflow run
```

* **Hold `Ctrl+Space`** — speak — release: text is typed into the focused app.
* **`Ctrl+Shift+Space`** — toggle hands-free dictation (stops on silence, repeats).
* **`Ctrl+Alt+Space`** — command mode: copy some text, press the hotkey, speak an instruction ("make this a bullet list"), press again.
* Dashboard with history, stats, dictionary editing: printed at startup (default `http://127.0.0.1:5170`).

All hotkeys are configurable:

```bash
localflow config set hotkeys.push_to_talk "<f9>"
```

### Other commands

```bash
localflow ui                             # open the dashboard app on its own
localflow transcribe memo.wav            # transcribe an audio file (add --json, --raw)
localflow listen                         # one hands-free dictation, prints the text
localflow dictionary add Wispr           # teach it names/jargon
localflow dictionary add brb "be right back"   # text replacement
localflow history --search "meeting"     # search past dictations
localflow stats                          # words, WPM, streak
localflow config show                    # full config as JSON
```

Config lives at `~/.config/localflow/config.json` (see `localflow/config.py` for every option — engine, hotkeys, audio/VAD, formatting toggles, per-app profiles, output method).

## Architecture

```
mic (sounddevice) ──► Recorder ──► VAD / RMS normalize ──► STT engine
                                                            (faster-whisper | whisper.cpp)
                                                                    │  ◄─ dictionary bias prompt
                                                                    ▼
   focused app ◄── Injector (⌘V paste / xdotool / pynput) ◄── Formatter ◄── dictionary correction
       ▲                                                        (rule engine + local LLM cleanup
       │                                                         via LM Studio/Ollama, app tone)
  HotkeyListener (pynput)                                               │
  Tray icon (pystray)          History (SQLite) ◄── FlowController ◄────┘
  Dashboard (localhost)  ◄────────┘
```

Every OS-dependent piece (microphone, hotkeys, window detection, typing) sits behind a small interface with a headless implementation, so the entire pipeline is unit-testable — and the STT layer is pluggable, so new engines are ~50 lines.

## Testing

```bash
pip install -e ".[dev]"
pytest                # 159 tests
```

The suite includes true end-to-end tests: speech is synthesized with `espeak-ng`, transcribed by a **real Whisper model** (whisper.cpp `ggml-tiny`), formatted, and injected — no mocks. Those tests auto-skip when `espeak-ng` or the model isn't available; everything else runs anywhere. Drop a model at `tests/models/ggml-tiny.bin` (or `~/.local/share/localflow/models/`) to enable them.

## Limitations vs. Wispr Flow

* The LLM cleanup pass adds latency proportional to your model's speed (a few seconds on Apple Silicon with a mid-size model). Turn it off in Settings → AI formatting, or pick a smaller/faster model, if you'd rather have instant rule-based-only insertion.
* "Streaming" preview re-transcribes the buffer periodically; Whisper isn't natively streaming.
* Wayland restricts global hotkeys and synthetic typing; X11/macOS/Windows are the happy paths.

## License

MIT
