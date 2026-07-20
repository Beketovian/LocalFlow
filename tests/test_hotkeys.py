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


class TestFnKey:
    def test_fn_alone_as_push_to_talk(self):
        # Wispr Flow's default: hold fn to talk. The fn token arrives from
        # the native tap on macOS; the combo logic is platform-neutral.
        events = []
        listener = HotkeyListener(
            push_to_talk="<fn>",
            toggle_dictation="<ctrl>+<shift>+<space>",
            command_mode="<ctrl>+<alt>+<space>",
            on_ptt_press=lambda: events.append("press"),
            on_ptt_release=lambda: events.append("release"),
            on_toggle=lambda: events.append("toggle"),
            on_command=lambda: events.append("command"),
        )
        listener.handle_press("fn")
        assert events == ["press"]
        listener.handle_release("fn")
        assert events == ["press", "release"]

    def test_fn_combo(self):
        events = []
        listener = HotkeyListener(
            push_to_talk="<fn>+d",
            toggle_dictation="",
            command_mode="",
            on_ptt_press=lambda: events.append("press"),
            on_ptt_release=lambda: events.append("release"),
            on_toggle=lambda: None,
            on_command=lambda: None,
        )
        listener.handle_press("fn")
        assert events == []
        listener.handle_press("d")
        assert events == ["press"]
        listener.handle_release("d")
        assert events == ["press", "release"]


class TestFormatCombo:
    def test_round_trips_through_parse(self):
        from localflow.hotkeys import format_combo

        for tokens in ({"ctrl", "space"}, {"fn"}, {"ctrl", "alt", "d"}, {"f9"}):
            assert _parse_combo(format_combo(tokens)) == tokens

    def test_modifier_ordering(self):
        from localflow.hotkeys import format_combo

        assert format_combo({"d", "ctrl", "fn"}) == "<fn>+<ctrl>+d"


class TestSecondaryPtt:
    def make(self, events, alt="<f13>"):
        return HotkeyListener(
            push_to_talk="<fn>",
            push_to_talk_alt=alt,
            toggle_dictation="<ctrl>+<shift>+<space>",
            command_mode="<ctrl>+<alt>+<space>",
            on_ptt_press=lambda: events.append("press"),
            on_ptt_release=lambda: events.append("release"),
            on_toggle=lambda: events.append("toggle"),
            on_command=lambda: events.append("command"),
        )

    def test_primary_still_works(self):
        events = []
        listener = self.make(events)
        listener.handle_press("fn")
        listener.handle_release("fn")
        assert events == ["press", "release"]

    def test_secondary_also_works(self):
        events = []
        listener = self.make(events)
        listener.handle_press("f13")
        assert events == ["press"]
        listener.handle_release("f13")
        assert events == ["press", "release"]

    def test_release_matches_the_combo_that_started(self):
        # Holding f13, a stray fn release must not end the recording.
        events = []
        listener = self.make(events)
        listener.handle_press("f13")
        listener.handle_press("fn")
        listener.handle_release("fn")   # not the active combo
        assert events == ["press"]
        listener.handle_release("f13")
        assert events == ["press", "release"]

    def test_no_double_press_when_both_held(self):
        events = []
        listener = self.make(events)
        listener.handle_press("fn")
        listener.handle_press("f13")
        assert events == ["press"]  # second combo doesn't re-trigger

    def test_empty_alt_is_single_bind(self):
        events = []
        listener = self.make(events, alt="")
        assert listener.ptt_combos == [{"fn"}]
        listener.handle_press("f13")
        assert events == []


class TestParseTolerance:
    def test_stray_comma_and_space(self):
        assert _parse_combo("<fn>, ") == {"fn"}
        assert _parse_combo("<ctrl>+ <space>;") == {"ctrl", "space"}
        assert _parse_combo("") == set()
