from localflow.hotkeys import HotkeyListener, _parse_combo


def make_listener(events):
    return HotkeyListener(
        push_to_talk="<ctrl>+<space>",
        toggle_dictation="<ctrl>+<shift>+<space>",
        command_mode="<ctrl>+<alt>+<space>",
        on_ptt_press=lambda: events.append("press"),
        on_ptt_release=lambda: events.append("release"),
        on_toggle=lambda: events.append("toggle"),
        on_command=lambda: events.append("command"),
    )


class TestParsing:
    def test_parse_combo(self):
        assert _parse_combo("<ctrl>+<space>") == {"ctrl", "space"}
        assert _parse_combo("a") == {"a"}


class TestHotkeys:
    def test_push_to_talk_cycle(self):
        events = []
        listener = make_listener(events)
        listener.handle_press("ctrl")
        listener.handle_press("space")
        assert events == ["press"]
        listener.handle_release("space")
        assert events == ["press", "release"]

    def test_ptt_not_retriggered_while_held(self):
        events = []
        listener = make_listener(events)
        listener.handle_press("ctrl")
        listener.handle_press("space")
        listener.handle_press("space")  # key repeat
        assert events.count("press") == 1

    def test_toggle_fires(self):
        events = []
        listener = make_listener(events)
        listener.handle_press("ctrl")
        listener.handle_press("shift")
        listener.handle_press("space")
        assert "toggle" in events
        assert "press" not in events  # superset combo wins

    def test_command_fires(self):
        events = []
        listener = make_listener(events)
        listener.handle_press("ctrl")
        listener.handle_press("alt")
        listener.handle_press("space")
        assert "command" in events

    def test_unrelated_keys_ignored(self):
        events = []
        listener = make_listener(events)
        listener.handle_press("a")
        listener.handle_release("a")
        assert events == []

    def test_none_token_ignored(self):
        events = []
        listener = make_listener(events)
        listener.handle_press(None)
        listener.handle_release(None)
        assert events == []


class TestCallbackSafety:
    def test_raising_callback_does_not_propagate(self, capsys):
        # A raising callback would kill pynput's listener thread and silently
        # disable every hotkey; handle_* must swallow and report instead.
        def boom():
            raise RuntimeError("kaboom")

        listener = HotkeyListener(
            push_to_talk="<ctrl>+<space>",
            toggle_dictation="<ctrl>+<shift>+<space>",
            command_mode="<ctrl>+<alt>+<space>",
            on_ptt_press=boom,
            on_ptt_release=boom,
            on_toggle=boom,
            on_command=boom,
        )
        listener.handle_press("ctrl")
        listener.handle_press("space")   # ptt press -> boom, swallowed
        listener.handle_release("space")  # ptt release -> boom, swallowed
        assert "kaboom" in capsys.readouterr().err

    def test_ptt_state_survives_raising_callback(self):
        calls = []

        def bad_press():
            calls.append("press")
            raise RuntimeError("kaboom")

        listener = HotkeyListener(
            push_to_talk="<ctrl>+<space>",
            toggle_dictation="",
            command_mode="",
            on_ptt_press=bad_press,
            on_ptt_release=lambda: calls.append("release"),
            on_toggle=lambda: None,
            on_command=lambda: None,
        )
        listener.handle_press("ctrl")
        listener.handle_press("space")
        listener.handle_release("space")  # release still delivered
        assert calls == ["press", "release"]
