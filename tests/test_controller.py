import numpy as np
import pytest

from localflow.app import FlowController
from localflow.audio import ArrayRecorder
from localflow.config import Config
from localflow.context import ActiveWindowProvider, WindowInfo
from localflow.engines.mock import MockEngine
from localflow.history import History
from localflow.injector import CallbackInjector
from localflow.sounds import SoundPlayer


def speechlike(seconds=1.0, rate=16000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


class FixedWindow(ActiveWindowProvider):
    def __init__(self, info):
        self.info = info

    def get(self):
        return self.info


def make_controller(responses=None, window=None, config=None):
    config = config or Config()
    config.save_history = False
    engine = MockEngine(responses or ["hello world"])
    injector = CallbackInjector()
    controller = FlowController(
        config=config,
        engine=engine,
        recorder=ArrayRecorder(speechlike()),
        injector=injector,
        history=History(":memory:"),
        window_provider=FixedWindow(window or WindowInfo(title="Doc", app="notepad")),
        sounds=SoundPlayer(enabled=False),
    )
    return controller, engine, injector


class TestDictationFlow:
    def test_full_cycle(self):
        controller, engine, injector = make_controller(["hello world"])
        assert controller.start_recording() is True
        assert controller.state.status == "recording"
        event = controller.stop_recording()
        assert controller.state.status == "idle"
        assert event.formatted_text == "Hello world"
        assert event.injected is True
        assert injector.received == ["Hello world "]
        assert len(engine.calls) == 1

    def test_double_start_rejected(self):
        controller, _, _ = make_controller()
        assert controller.start_recording() is True
        assert controller.start_recording() is False
        controller.stop_recording()

    def test_stop_without_start(self):
        controller, _, _ = make_controller()
        assert controller.stop_recording() is None

    def test_cancel(self):
        controller, engine, injector = make_controller()
        controller.start_recording()
        controller.cancel_recording()
        assert controller.state.status == "idle"
        assert engine.calls == []
        assert injector.received == []

    def test_history_recorded(self):
        controller, _, _ = make_controller(["some dictated text"])
        controller.start_recording()
        controller.stop_recording()
        entries = controller.history.recent()
        assert len(entries) == 1
        assert entries[0].formatted_text == "Some dictated text"
        assert entries[0].app == "notepad"

    def test_listener_notified(self):
        controller, _, _ = make_controller()
        seen = []
        controller.on_dictation(seen.append)
        controller.start_recording()
        controller.stop_recording()
        assert len(seen) == 1 and seen[0].injected

    def test_empty_transcript_not_injected(self):
        controller, _, injector = make_controller([""])
        controller.start_recording()
        event = controller.stop_recording()
        assert event.injected is False
        assert injector.received == []
        assert controller.history.recent() == []

    def test_smart_join_across_bursts(self):
        controller, _, injector = make_controller(["I went to the store", "And bought milk"])
        controller.start_recording()
        controller.stop_recording()
        controller.recorder.audio = speechlike()
        controller.start_recording()
        controller.stop_recording()
        assert injector.received[1].startswith("and bought milk")

    def test_dictionary_boost_passed_to_engine(self):
        config = Config()
        config.dictionary = ["Anthropic"]
        controller, engine, _ = make_controller(config=config)
        controller.start_recording()
        controller.stop_recording()
        assert "Anthropic" in engine.calls[0]["initial_prompt"]

    def test_dictionary_correction_applied(self):
        config = Config()
        config.dictionary = ["Wispr"]
        controller, _, injector = make_controller(["i tried wisper today"], config=config)
        controller.start_recording()
        controller.stop_recording()
        assert "Wispr" in injector.received[0]

    def test_replacements_applied(self):
        config = Config()
        config.replacements = {"be right back": "brb"}
        controller, _, injector = make_controller(["be right back"], config=config)
        controller.start_recording()
        controller.stop_recording()
        assert injector.received[0].strip() == "Brb"


class TestProfileAwareFormatting:
    def test_terminal_profile_disables_caps(self):
        window = WindowInfo(title="user@host: ~", app="gnome-terminal")
        controller, _, injector = make_controller(["list the files"], window=window)
        controller.start_recording()
        controller.stop_recording()
        assert injector.received[0].startswith("list")  # not capitalized

    def test_email_profile_adds_period(self):
        window = WindowInfo(title="Inbox - Gmail", app="firefox")
        controller, _, injector = make_controller(["thanks for the update"], window=window)
        controller.start_recording()
        controller.stop_recording()
        assert injector.received[0].strip().endswith(".")


class TestHandsFree:
    def test_hands_free_stops_on_silence(self):
        # speech then silence: run_hands_free_once should stop by itself
        audio = np.concatenate([speechlike(0.8), np.zeros(32000, dtype=np.float32)])
        controller, _, injector = make_controller(["auto stopped"])
        controller.recorder = ArrayRecorder(audio)
        controller.config.audio.silence_stop_after = 0.5
        event = controller.run_hands_free_once(poll_interval=0.02, max_seconds=3)
        assert event is not None
        assert event.mode == "hands-free"
        assert injector.received


class TestCommandMode:
    def test_run_command(self):
        controller, _, injector = make_controller()
        out = controller.run_command("make this uppercase", "hello world")
        assert out == "HELLO WORLD"
        assert injector.received[-1] == "HELLO WORLD"
        assert controller.history.recent()[0].mode == "command"

    def test_unknown_command_returns_none(self):
        controller, _, injector = make_controller()
        assert controller.run_command("frobnicate", "hello") is None
        assert injector.received == []

    def test_command_mode_recording_does_not_inject(self):
        controller, _, injector = make_controller(["uppercase please"])
        controller.start_recording(mode="command")
        event = controller.stop_recording()
        assert event.raw_text == "uppercase please"
        assert event.injected is False
        assert injector.received == []


class TestDictateArray:
    def test_one_shot(self):
        controller, _, _ = make_controller(["file transcription"])
        event = controller.dictate_array(speechlike())
        assert event.formatted_text == "File transcription"
        assert event.duration == pytest.approx(1.0, abs=0.01)


class TestStatusRecovery:
    def test_status_resets_when_recorder_raises(self):
        # If the recorder (or anything downstream) raises, status must return
        # to idle - otherwise the pill shows "processing" dots forever.
        controller, _, _ = make_controller()
        statuses = []
        controller.on_status(statuses.append)
        controller.start_recording()

        def explode():
            raise RuntimeError("mic went away")

        controller.recorder.stop = explode
        try:
            controller.stop_recording()
        except RuntimeError:
            pass
        assert controller.state.status == "idle"
        assert statuses[-1] == "idle"


class TestBlankAudio:
    def test_blank_audio_injects_nothing(self):
        # Whisper's [BLANK_AUDIO] for silent recordings must never be pasted
        # or saved to history.
        controller, _, injector = make_controller(["[BLANK_AUDIO]"])
        controller.start_recording()
        event = controller.stop_recording()
        assert event.formatted_text == ""
        assert event.injected is False
        assert injector.received == []
        assert controller.history.recent(10) == []


class SendCapableInjector(CallbackInjector):
    def __init__(self):
        super().__init__()
        self.returns_pressed = 0

    def press_return(self):
        self.returns_pressed += 1


class TestVoiceActions:
    def test_send_it_presses_return(self):
        controller, _, _ = make_controller(["See you at six, send it"])
        injector = SendCapableInjector()
        controller.injector = injector
        controller.start_recording()
        event = controller.stop_recording()
        assert event.formatted_text == "See you at six"
        assert injector.received == ["See you at six "]
        assert injector.returns_pressed == 1
        assert event.auto_sent is True

    def test_toggle_off_keeps_text(self):
        controller, _, _ = make_controller(["See you at six, send it"])
        controller.config.output.voice_send = False
        injector = SendCapableInjector()
        controller.injector = injector
        controller.start_recording()
        event = controller.stop_recording()
        assert "send it" in event.formatted_text.lower()
        assert injector.returns_pressed == 0
        assert event.auto_sent is False

    def test_plain_injector_no_send(self):
        controller, _, injector = make_controller(["hello there, send it"])
        controller.start_recording()
        event = controller.stop_recording()
        assert event.auto_sent is False  # injector lacks press_return
        assert injector.received == ["Hello there "]


class TestLazyEngineCreation:
    def test_concurrent_first_use_creates_one_engine(self):
        # The startup warm-up thread and an eager first dictation both hit
        # the lazy engine property; exactly one model must be created.
        import threading

        config = Config()
        config.engine.backend = "mock"
        config.save_history = False
        controller = FlowController(
            config=config,
            recorder=ArrayRecorder(speechlike()),
            injector=CallbackInjector(),
            history=History(":memory:"),
            sounds=SoundPlayer(enabled=False),
        )
        engines = []
        threads = [
            threading.Thread(target=lambda: engines.append(controller.ensure_engine()))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(engines) == 8
        assert len({id(e) for e in engines}) == 1
        assert controller.engine is engines[0]
