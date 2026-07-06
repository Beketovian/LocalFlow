"""LocalFlow command line.

    localflow run                    start the dictation daemon (hotkeys + tray + dashboard)
    localflow transcribe FILE.wav    transcribe an audio file
    localflow listen                 one hands-free dictation from the mic, print it
    localflow setup [--model base]   download a Whisper model
    localflow dictionary ...         manage the personal dictionary
    localflow history [--search q]   show dictation history
    localflow stats                  dictation statistics
    localflow config ...             show config / set values / write defaults
    localflow doctor                 check what's installed and working
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .config import Config, default_config_dir


def _load_config(args) -> Config:
    path = Path(args.config) if getattr(args, "config", None) else None
    return Config.load(path)


def _config_path(args) -> Path:
    return Path(args.config) if getattr(args, "config", None) else default_config_dir() / "config.json"


# ------------------------------------------------------------------- commands


def cmd_transcribe(args) -> int:
    from .app import FlowController
    from .audio import load_wav
    from .injector import CallbackInjector

    config = _load_config(args)
    if args.language:
        config.engine.language = args.language
    if args.model:
        config.engine.model = args.model
    if args.raw:
        config.formatting.enabled = False
    controller = FlowController(config=config, injector=CallbackInjector())
    audio = load_wav(args.file)
    event = controller.dictate_array(audio)
    if args.json:
        print(json.dumps({
            "raw": event.raw_text,
            "formatted": event.formatted_text,
            "language": event.language,
            "audio_seconds": round(event.duration, 2),
            "elapsed_seconds": round(event.elapsed, 2),
        }, ensure_ascii=False, indent=2))
    else:
        print(event.formatted_text)
    controller.close()
    return 0


def cmd_listen(args) -> int:
    from .app import FlowController
    from .audio import MicrophoneRecorder
    from .injector import StdoutInjector

    config = _load_config(args)
    controller = FlowController(
        config=config,
        recorder=MicrophoneRecorder(
            sample_rate=config.audio.sample_rate,
            device=config.audio.input_device,
            max_seconds=config.audio.max_recording_seconds,
        ),
        injector=StdoutInjector(),
    )
    print("Listening... speak, then pause to finish.", file=sys.stderr)
    controller.run_hands_free_once()
    controller.close()
    return 0


def _warn_if_macos_untrusted() -> None:
    """On macOS, hotkeys and typing need Accessibility trust - say so clearly."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.util.find_library("ApplicationServices")
        appsvc = ctypes.cdll.LoadLibrary(lib)
        trusted = bool(appsvc.AXIsProcessTrusted())
    except Exception:
        return
    if not trusted:
        print(
            "\n  ⚠ macOS hasn't granted this process input-monitoring trust yet,\n"
            "    so the hotkeys and text insertion will NOT work until you:\n"
            "      1. Open System Settings → Privacy & Security → Accessibility\n"
            "         and add/enable your terminal app (and the venv's python if listed).\n"
            "      2. Do the same under Privacy & Security → Input Monitoring.\n"
            "      3. Fully quit and reopen the terminal, then run 'localflow run' again.\n"
            "    Also click Allow when macOS asks for Microphone access.\n"
        )


def _acquire_daemon_lock(config: Config):
    """One daemon at a time. Two instances double-transcribe, double-paste
    and fight over the hotkeys (seen in the wild with a Ctrl+Z'd daemon).

    Returns an open, flock-ed file to hold for the process lifetime, or None
    when another live daemon holds the lock.
    """
    path = config.resolved_data_dir() / "daemon.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        return handle  # Windows: no flock; stay permissive
    except OSError:
        handle.seek(0)
        pid = handle.read().strip() or "unknown pid"
        handle.close()
        print(f"Another 'localflow run' is already active ({pid}).\n"
              f"Stop it first (Ctrl+C in its terminal, or: kill {pid}).\n"
              f"Note: a daemon suspended with Ctrl+Z still counts - "
              f"resume it with 'fg' and Ctrl+C it.")
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def cmd_run(args) -> int:
    import queue
    import threading

    from .app import FlowController
    from .audio import MicrophoneRecorder, rms
    from .dashboard import DashboardServer
    from .hotkeys import HotkeyListener
    from .injector import create_injector
    from .overlay import RecordingOverlay

    config = _load_config(args)

    lock = _acquire_daemon_lock(config)
    if lock is None:
        return 1

    # The floating recording pill (Wispr Flow's on-screen widget). Feeds on
    # live mic levels; silently unavailable on headless systems. On macOS,
    # AppKit only allows windows on the main thread, so Tk is built here and
    # run_forever() below owns the main loop.
    overlay = RecordingOverlay()
    if overlay.needs_main_thread:
        overlay_ok = overlay.init_main_thread()
    else:
        overlay_ok = overlay.start()

    def on_chunk(chunk) -> None:
        if overlay_ok:
            overlay.set_level(min(1.0, rms(chunk) * 12.0))

    controller = FlowController(
        config=config,
        recorder=MicrophoneRecorder(
            sample_rate=config.audio.sample_rate,
            device=config.audio.input_device,
            max_seconds=config.audio.max_recording_seconds,
            on_chunk=on_chunk,
        ),
        injector=create_injector(
            config.output.method,
            config.output.type_interval,
            config.output.restore_clipboard,
        ),
    )

    if overlay_ok:
        def on_status(status: str) -> None:
            if status == "recording":
                overlay.show("recording")
            elif status == "transcribing":
                overlay.show("processing")
            else:
                overlay.hide()

        controller.on_status(on_status)
    print(f"LocalFlow {__version__} - local voice dictation")
    print(f"  engine: {config.engine.backend} / {config.engine.model}"
          f" (language: {config.engine.language})")
    print(f"  insert: {controller.injector.name}")

    if config.llm.enabled:
        if controller.llm.available:
            print(f"  ai formatting: {controller.llm.model} @ {controller.llm.base_url}")
            # load the model off the hot path so the first dictation is fast
            threading.Thread(target=controller.llm.warm_up, daemon=True).start()
        else:
            print("  ai formatting: no model available - using rule-based "
                  "formatting (run 'localflow llm download' for the built-in "
                  "model, or start LM Studio/Ollama)")

    if sys.platform == "darwin" and os.environ.get("LOCALFLOW_APP"):
        # Running as LocalFlow.app: use the system prompt that sends the
        # user straight to Privacy & Security -> Accessibility on first run.
        try:
            import ApplicationServices as AppServices

            AppServices.AXIsProcessTrustedWithOptions(
                {AppServices.kAXTrustedCheckOptionPrompt: True}
            )
        except Exception:
            pass

    dashboard = None
    if config.dashboard.enabled:
        try:
            dashboard = DashboardServer(controller, config.dashboard.host, config.dashboard.port)
            port = dashboard.start()
            note = "" if port == config.dashboard.port else \
                f"  (port {config.dashboard.port} was busy)"
            print(f"  dashboard: http://{config.dashboard.host}:{port}{note}")
        except OSError as exc:
            dashboard = None
            print(f"  dashboard: unavailable ({exc}); dictation still works")

    # warm the model up front so the first dictation is snappy
    print("  loading model...", end=" ", flush=True)
    controller.engine
    print("ready.")

    hands_free = {"on": False}

    # Hotkey callbacks run on pynput's macOS event tap, and the OS *disables*
    # taps that block (kCGEventTapDisabledByTimeout) - transcription takes
    # seconds, so doing it inline kills all hotkeys. A single worker thread
    # runs the real work; callbacks just enqueue, returning in microseconds.
    # One worker also preserves press/release ordering.
    actions: "queue.Queue" = queue.Queue()

    def action_worker() -> None:
        while True:
            fn = actions.get()
            if fn is None:
                return
            try:
                fn()
            except Exception as exc:
                print(f"  ! dictation failed: {exc!r}")

    threading.Thread(target=action_worker, daemon=True).start()

    def ptt_press() -> None:
        controller.start_recording()

    def _timing(event) -> str:
        parts = [f"stt {event.stt_seconds:.2f}s"]
        if event.llm_seconds:
            parts.append(f"llm {event.llm_seconds:.2f}s"
                         + ("" if event.llm_used else " (fallback)"))
        return f" [{', '.join(parts)}]"

    def ptt_release() -> None:
        event = controller.stop_recording()
        if event and args.verbose:
            print(f"  > {event.formatted_text}{_timing(event)}")

    menubar = {"bar": None}  # filled in below (macOS AppKit mode only)

    def toggle() -> None:
        hands_free["on"] = not hands_free["on"]
        controller.state.hands_free = hands_free["on"]
        print("hands-free:", "ON" if hands_free["on"] else "OFF")
        if menubar["bar"]:
            menubar["bar"].set_hands_free(hands_free["on"])

    def command_mode() -> None:
        # First press: start recording the spoken instruction. Second press:
        # transcribe it and apply to the copied text (copy your selection
        # first - reading live selections is platform-fragile).
        if controller.state.status == "idle":
            print("command mode: speak an instruction, press the hotkey again to apply")
            controller.start_recording(mode="command")
        elif controller.state.mode == "command":
            event = controller.stop_recording()
            if not event or not event.raw_text:
                return
            selection = ""
            try:
                import pyperclip

                selection = pyperclip.paste()
            except Exception:
                pass
            edited = controller.run_command(event.raw_text, selection)
            print(f"  command: {event.raw_text!r} -> "
                  f"{'applied' if edited is not None else 'not understood'}")

    listener = HotkeyListener(
        push_to_talk=config.hotkeys.push_to_talk,
        toggle_dictation=config.hotkeys.toggle_dictation,
        command_mode=config.hotkeys.command_mode,
        on_ptt_press=lambda: actions.put(ptt_press),
        on_ptt_release=lambda: actions.put(ptt_release),
        on_toggle=lambda: actions.put(toggle),
        on_command=lambda: actions.put(command_mode),
    )
    listener.start()
    print(f"  hold {config.hotkeys.push_to_talk} to dictate;"
          f" {config.hotkeys.toggle_dictation} toggles hands-free. Ctrl+C quits.")
    _warn_if_macos_untrusted()

    stop_event = threading.Event()

    tray = None
    if sys.platform == "darwin" and overlay_ok and overlay.native:
        # Native menu bar item riding the pill's AppKit runloop: status
        # glyph, dashboard link, hands-free toggle, quit. This is the app's
        # visible identity when launched as LocalFlow.app (no terminal).
        try:
            from .menubar import MacMenuBar

            url = f"http://{config.dashboard.host}:{dashboard.port}" if dashboard else None

            def quit_app() -> None:
                stop_event.set()
                overlay.stop()  # ends run_forever -> cmd_run's finally

            bar = MacMenuBar(
                version=__version__,
                dashboard_url=url,
                on_quit=quit_app,
                on_toggle_hands_free=lambda: actions.put(toggle),
            )
            bar.attach(overlay)
            controller.on_status(bar.set_status)
            menubar["bar"] = bar
        except Exception:
            pass  # no menu bar; dictation still works
    elif sys.platform != "darwin":
        # pystray's macOS backend is also main-thread-only, and the pill
        # already owns the main thread there - tray is non-mac only.
        try:
            from .tray import TrayIcon

            url = f"http://{config.dashboard.host}:{dashboard.port}" if dashboard else None
            tray = TrayIcon(controller, dashboard_url=url)
            tray.start()
            controller.on_dictation(lambda e: tray.set_status("idle"))
        except Exception:
            pass  # headless or pystray missing - fine

    if config.llm.enabled:
        # LM Studio JIT-unloads idle models; the next dictation would then
        # pay the full reload. A tiny request every 10 minutes keeps the
        # model resident for the whole session.
        def llm_keepalive() -> None:
            while not stop_event.wait(600):
                controller.llm.warm_up()

        threading.Thread(target=llm_keepalive, daemon=True).start()

    def daemon_loop() -> None:
        while not stop_event.is_set():
            if hands_free["on"] and controller.state.status == "idle":
                event = controller.run_hands_free_once()
                if event and args.verbose:
                    print(f"  > {event.formatted_text}{_timing(event)}")
            else:
                time.sleep(0.15)

    try:
        if overlay_ok and overlay.needs_main_thread:
            # macOS: pill runs the main thread, dictation work moves off it
            worker = threading.Thread(target=daemon_loop, daemon=True)
            worker.start()
            overlay.run_forever()
        else:
            daemon_loop()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        stop_event.set()
        actions.put(None)
        listener.stop()
        if dashboard:
            dashboard.stop()
        if tray:
            tray.stop()
        if menubar["bar"]:
            menubar["bar"].remove()
        if overlay_ok:
            overlay.stop()
        controller.close()
        lock.close()
    return 0


def cmd_ui(args) -> int:
    """Open the dashboard app on its own (no hotkeys/microphone)."""
    import webbrowser

    from .app import FlowController
    from .dashboard import DashboardServer
    from .injector import CallbackInjector

    config = _load_config(args)
    controller = FlowController(config=config, injector=CallbackInjector())
    server = DashboardServer(controller, config.dashboard.host, config.dashboard.port)
    port = server.start()
    url = f"http://{config.dashboard.host}:{port}"
    note = "" if port == config.dashboard.port else f"  (port {config.dashboard.port} was busy)"
    print(f"LocalFlow dashboard: {url}  (Ctrl+C to quit){note}")
    if not getattr(args, "no_browser", False):
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        controller.close()
    return 0


def cmd_setup(args) -> int:
    from .engines.registry import available_backends, download_ggml_model, ggml_model_path

    config = _load_config(args)
    backends = available_backends()
    print("Installed STT backends:", ", ".join(backends) or "none!")
    model = args.model or config.engine.model
    if "whisper.cpp" in backends:
        path = download_ggml_model(model, config)
        print(f"whisper.cpp model ready: {path}")
    elif "faster-whisper" in backends:
        print(f"Downloading faster-whisper '{model}' (via Hugging Face)...")
        from faster_whisper import WhisperModel

        WhisperModel(model, device="cpu", compute_type="int8")
        print("faster-whisper model cached.")
    else:
        print("Install a backend first: pip install 'localflow[whispercpp]'")
        return 1
    cfg_path = _config_path(args)
    if not cfg_path.exists():
        config.engine.model = model
        config.save(cfg_path)
        print(f"Wrote default config: {cfg_path}")
    return 0


def cmd_dictionary(args) -> int:
    config = _load_config(args)
    if args.action == "add":
        if args.written:  # replacement
            config.replacements[args.word] = args.written
            print(f"replacement: '{args.word}' -> '{args.written}'")
        elif args.word not in config.dictionary:
            config.dictionary.append(args.word)
            print(f"added: {args.word}")
    elif args.action == "remove":
        if args.word in config.dictionary:
            config.dictionary.remove(args.word)
            print(f"removed: {args.word}")
        config.replacements.pop(args.word, None)
    elif args.action == "list":
        print("Words:", ", ".join(config.dictionary) or "(none)")
        for k, v in config.replacements.items():
            print(f"  {k} -> {v}")
        return 0
    config.save(_config_path(args))
    return 0


def cmd_history(args) -> int:
    from .history import History

    config = _load_config(args)
    history = History(config.resolved_data_dir() / "history.db")
    entries = history.search(args.search) if args.search else history.recent(args.limit)
    for e in entries:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.timestamp))
        app = f" [{e.app}]" if e.app else ""
        print(f"{stamp}{app} {e.formatted_text}")
    history.close()
    return 0


def cmd_stats(args) -> int:
    from .history import History

    config = _load_config(args)
    history = History(config.resolved_data_dir() / "history.db")
    s = history.stats()
    print(f"Dictations:      {s.total_entries}")
    print(f"Words dictated:  {s.total_words}")
    print(f"Words today:     {s.words_today}")
    print(f"Audio minutes:   {s.total_audio_seconds / 60:.1f}")
    print(f"Average WPM:     {s.average_wpm}")
    print(f"Streak:          {s.streak_days} day(s)")
    history.close()
    return 0


def cmd_config(args) -> int:
    config = _load_config(args)
    if args.action == "show":
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    elif args.action == "init":
        path = config.save(_config_path(args))
        print(f"wrote {path}")
    elif args.action == "set":
        section_key, value = args.pair
        parts = section_key.split(".")
        target = config
        for part in parts[:-1]:
            target = getattr(target, part)
        current = getattr(target, parts[-1])
        if isinstance(current, bool):
            value = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(target, parts[-1], value)
        path = config.save(_config_path(args))
        print(f"{section_key} = {value!r} (saved to {path})")
    return 0


def cmd_llm(args) -> int:
    """Manage the built-in (embedded) LLM used for AI formatting."""
    from .llm import LLMClient
    from .llm_local import DEFAULT_REPO, download_default_model, find_local_model

    config = _load_config(args)
    data_dir = config.resolved_data_dir()
    if args.action == "download":
        print(f"Downloading {DEFAULT_REPO} (~2.3 GB)...")
        path = download_default_model(data_dir, progress=print)
        print(f"Done. LocalFlow will use it automatically: {path}")
        return 0
    # status
    client = LLMClient(config.llm, data_dir=data_dir)
    if client.available:
        print(f"backend: {client.mode}")
        print(f"model:   {client.model} @ {client.base_url}")
    else:
        print("no LLM available - AI formatting falls back to rules")
    local = find_local_model(data_dir, config.llm.model_path, config.llm.model)
    print(f"local weights: {local or 'none found'}")
    return 0


def cmd_doctor(args) -> int:
    import importlib.util

    from .engines.registry import available_backends, ggml_model_path

    config = _load_config(args)
    print(f"LocalFlow {__version__}")
    print(f"config: {_config_path(args)} "
          f"({'exists' if _config_path(args).exists() else 'defaults'})")
    backends = available_backends()
    print(f"STT backends: {', '.join(backends) or 'NONE - install one!'}")
    model_file = ggml_model_path(config.engine.model, config)
    print(f"ggml model '{config.engine.model}': "
          f"{'present' if model_file.exists() else 'not downloaded'} ({model_file})")
    for mod, why in (
        ("sounddevice", "microphone capture"),
        ("pynput", "global hotkeys + typing"),
        ("pystray", "tray icon"),
    ):
        ok = importlib.util.find_spec(mod) is not None
        print(f"{mod}: {'ok' if ok else f'missing ({why})'}")
    from .llm import LLMClient

    llm = LLMClient(config.llm)
    if llm.available:
        print(f"local LLM: {llm.model} @ {llm.base_url} "
              f"({len(llm.models)} model(s) available)")
    else:
        print("local LLM: not found (optional - start LM Studio or Ollama "
              "for AI formatting)")
    try:
        import sounddevice as sd

        devices = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
        print("input devices:", ", ".join(devices[:5]) or "none found")
    except Exception as exc:
        print(f"input devices: unavailable ({type(exc).__name__})")
    _warn_if_macos_untrusted()
    return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localflow",
        description="LocalFlow - fully local voice dictation powered by Whisper",
    )
    parser.add_argument("--version", action="version", version=f"localflow {__version__}")
    parser.add_argument("--config", help="path to config.json", default=None)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("run", help="start the dictation daemon")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("transcribe", help="transcribe an audio file")
    p.add_argument("file")
    p.add_argument("--language", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--raw", action="store_true", help="skip formatting")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("listen", help="one hands-free dictation, print the text")
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("ui", help="open the dashboard app (no dictation daemon)")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("setup", help="download a model and write default config")
    p.add_argument("--model", default=None, help="tiny, base, small, medium, large-v3 ...")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("dictionary", help="manage the personal dictionary")
    p.add_argument("action", choices=["add", "remove", "list"])
    p.add_argument("word", nargs="?", default="")
    p.add_argument("written", nargs="?", default=None,
                   help="written form (makes this a replacement)")
    p.set_defaults(func=cmd_dictionary)

    p = sub.add_parser("history", help="show recent dictations")
    p.add_argument("--search", default=None)
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("stats", help="dictation statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("config", help="show or edit configuration")
    p.add_argument("action", choices=["show", "init", "set"])
    p.add_argument("pair", nargs="*", metavar="KEY VALUE",
                   help="e.g. engine.model base")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("llm", help="manage the built-in AI-formatting model")
    p.add_argument("action", choices=["status", "download"], nargs="?",
                   default="status")
    p.set_defaults(func=cmd_llm)

    p = sub.add_parser("doctor", help="diagnose the installation")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if args.command == "config" and args.action == "set" and len(args.pair) != 2:
        parser.error("config set needs KEY VALUE")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
