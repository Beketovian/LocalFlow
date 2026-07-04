import json

import pytest

from localflow.cli import build_parser, main


class TestParser:
    def test_help_runs(self, capsys):
        assert main([]) == 0
        assert "localflow" in capsys.readouterr().out

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            main(["--version"])


class TestConfigCommand:
    def test_show(self, tmp_path, capsys):
        cfg = tmp_path / "config.json"
        assert main(["--config", str(cfg), "config", "show"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["engine"]["backend"] == "auto"

    def test_init_and_set(self, tmp_path, capsys):
        cfg = tmp_path / "config.json"
        assert main(["--config", str(cfg), "config", "init"]) == 0
        assert cfg.exists()
        assert main(["--config", str(cfg), "config", "set", "engine.model", "tiny"]) == 0
        assert json.loads(cfg.read_text())["engine"]["model"] == "tiny"

    def test_set_bool_coercion(self, tmp_path):
        cfg = tmp_path / "config.json"
        main(["--config", str(cfg), "config", "set", "formatting.remove_fillers", "false"])
        assert json.loads(cfg.read_text())["formatting"]["remove_fillers"] is False


class TestDictionaryCommand:
    def test_add_list_remove(self, tmp_path, capsys):
        cfg = tmp_path / "config.json"
        main(["--config", str(cfg), "dictionary", "add", "Wispr"])
        main(["--config", str(cfg), "dictionary", "list"])
        assert "Wispr" in capsys.readouterr().out
        main(["--config", str(cfg), "dictionary", "remove", "Wispr"])
        assert "Wispr" not in json.loads(cfg.read_text())["dictionary"]

    def test_add_replacement(self, tmp_path):
        cfg = tmp_path / "config.json"
        main(["--config", str(cfg), "dictionary", "add", "brb", "be right back"])
        assert json.loads(cfg.read_text())["replacements"]["brb"] == "be right back"


class TestStatsHistory:
    def test_stats_empty(self, tmp_path, capsys, monkeypatch):
        cfg = tmp_path / "config.json"
        # point data dir at tmp so we don't touch the real home
        from localflow.config import Config

        c = Config()
        c.data_dir = str(tmp_path / "data")
        c.save(cfg)
        assert main(["--config", str(cfg), "stats"]) == 0
        assert "Words dictated:  0" in capsys.readouterr().out

    def test_history_empty(self, tmp_path, capsys):
        cfg = tmp_path / "config.json"
        from localflow.config import Config

        c = Config()
        c.data_dir = str(tmp_path / "data")
        c.save(cfg)
        assert main(["--config", str(cfg), "history"]) == 0


class TestDoctor:
    def test_doctor_runs(self, tmp_path, capsys):
        cfg = tmp_path / "config.json"
        from localflow.config import Config

        c = Config()
        c.data_dir = str(tmp_path / "data")
        c.save(cfg)
        assert main(["--config", str(cfg), "doctor"]) == 0
        out = capsys.readouterr().out
        assert "STT backends" in out
