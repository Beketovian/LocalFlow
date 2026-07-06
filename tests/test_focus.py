"""Regression tests: the recording pill must not corrupt paste targeting.

The bug: showing the pill on macOS stole focus before the active window was
captured, so dictations were attributed to (and pasted into) our own Python
process instead of the user's app.
"""

from __future__ import annotations

import numpy as np

from localflow.app import FlowController
from localflow.audio import ArrayRecorder
from localflow.config import Config
from localflow.context import ActiveWindowProvider, WindowInfo
from localflow.engines.mock import MockEngine
from localflow.history import History
from localflow.injector import CallbackInjector
from localflow.sounds import SoundPlayer


class StealableWindowProvider(ActiveWindowProvider):
    """Reports the user's app until focus is 'stolen', then reports us."""

    def __init__(self):
        self.stolen = False

    def get(self) -> WindowInfo:
        if self.stolen:
            return WindowInfo(app="Python", pid=999)
        return WindowInfo(app="Messages", pid=42)


class FocusAwareInjector(CallbackInjector):
    """Callback injector that, like MacPasteInjector, exposes focus_pid."""

    def __init__(self):
        super().__init__()
        self.focus_pid = 0


def make_controller(provider, injector):
    config = Config()
    config.save_history = False
    config.llm.enabled = False
    return FlowController(
        config=config,
        engine=MockEngine(["hello there"]),
        recorder=ArrayRecorder(np.zeros(16000, dtype=np.float32)),
        injector=injector,
        history=History(":memory:"),
        window_provider=provider,
        sounds=SoundPlayer(enabled=False),
    )


class TestFocusCapture:
    def test_window_captured_before_status_listeners_fire(self):
        # The pill shows in reaction to the "recording" status change and can
        # steal focus; the target window must be captured before that.
        provider = StealableWindowProvider()
        controller = make_controller(provider, CallbackInjector())

        def steal_focus(status):
            if status == "recording":
                provider.stolen = True  # simulates the pill activating us

        controller.on_status(steal_focus)
        controller.start_recording()
        event = controller.stop_recording()
        assert event.app == "Messages"  # not "Python"

    def test_injector_receives_target_pid(self):
        provider = StealableWindowProvider()
        injector = FocusAwareInjector()
        controller = make_controller(provider, injector)
        controller.start_recording()
        controller.stop_recording()
        assert injector.focus_pid == 42

    def test_plain_injector_unharmed(self):
        # Injectors without focus_pid (stdout, xdotool, ...) must not grow one.
        provider = StealableWindowProvider()
        injector = CallbackInjector()
        controller = make_controller(provider, injector)
        controller.start_recording()
        controller.stop_recording()
        assert not hasattr(injector, "focus_pid")
        assert injector.received  # still injected
