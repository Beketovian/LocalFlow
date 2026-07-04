from localflow.config import AppProfile, Config, default_profiles
from localflow.context import WindowInfo, match_profile


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.engine.backend == "auto"
        assert cfg.formatting.remove_fillers is True
        assert cfg.hotkeys.push_to_talk

    def test_roundtrip(self, tmp_path):
        cfg = Config()
        cfg.engine.model = "small"
        cfg.dictionary = ["Wispr"]
        cfg.replacements = {"eta": "ETA"}
        cfg.formatting.spoken_punctuation = True
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = Config.load(path)
        assert loaded.engine.model == "small"
        assert loaded.dictionary == ["Wispr"]
        assert loaded.replacements == {"eta": "ETA"}
        assert loaded.formatting.spoken_punctuation is True

    def test_load_missing_file_gives_defaults(self, tmp_path):
        cfg = Config.load(tmp_path / "nope.json")
        assert cfg.engine.backend == "auto"

    def test_load_corrupt_file_gives_defaults(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert Config.load(path).engine.backend == "auto"

    def test_unknown_keys_ignored(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text('{"engine": {"model": "tiny", "bogus": 1}, "whatever": true}')
        cfg = Config.load(path)
        assert cfg.engine.model == "tiny"


class TestProfiles:
    def test_terminal_profile_matches(self):
        w = WindowInfo(title="user@host: ~", app="gnome-terminal-server")
        p = match_profile(w, default_profiles())
        assert p is not None and p.name == "terminal"
        assert p.overrides["capitalize_sentences"] is False

    def test_chat_profile_matches_title(self):
        w = WindowInfo(title="Slack - #general", app="slack")
        p = match_profile(w, default_profiles())
        assert p.name == "chat"

    def test_no_match(self):
        w = WindowInfo(title="Some Random App", app="randomapp")
        assert match_profile(w, default_profiles()) is None

    def test_unknown_window(self):
        assert match_profile(WindowInfo(), default_profiles()) is None

    def test_custom_profile(self):
        profiles = [AppProfile(name="x", match=["myapp"], overrides={"enabled": False})]
        p = match_profile(WindowInfo(app="MyApp v2"), profiles)
        assert p.name == "x"
