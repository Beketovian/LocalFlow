import time

from localflow.history import History


class TestHistory:
    def test_add_and_recent(self):
        h = History(":memory:")
        h.add("raw one", "One formatted.", app="Slack", duration=2.0)
        h.add("raw two", "Two formatted.", app="Gmail", duration=3.0)
        entries = h.recent()
        assert len(entries) == 2
        assert entries[0].formatted_text == "Two formatted."  # newest first
        assert entries[1].app == "Slack"

    def test_search(self):
        h = History(":memory:")
        h.add("", "the quick brown fox")
        h.add("", "hello world")
        assert len(h.search("fox")) == 1
        assert h.search("fox")[0].formatted_text == "the quick brown fox"

    def test_delete_and_clear(self):
        h = History(":memory:")
        eid = h.add("", "delete me")
        assert h.delete(eid) is True
        assert h.delete(eid) is False
        h.add("", "a")
        h.clear()
        assert h.recent() == []

    def test_persistence(self, tmp_path):
        path = tmp_path / "hist.db"
        h1 = History(path)
        h1.add("", "persisted entry")
        h1.close()
        h2 = History(path)
        assert h2.recent()[0].formatted_text == "persisted entry"
        h2.close()

    def test_stats(self):
        h = History(":memory:")
        now = time.time()
        # 60 words over 60 seconds of audio -> 60 WPM
        words30 = " ".join(["word"] * 30)
        h.add("", words30, duration=30.0, timestamp=now)
        h.add("", words30, duration=30.0, timestamp=now - 86400)  # yesterday
        s = h.stats(now=now)
        assert s.total_entries == 2
        assert s.total_words == 60
        assert s.average_wpm == 60.0
        assert s.words_today == 30
        assert s.streak_days == 2

    def test_streak_broken(self):
        h = History(":memory:")
        now = time.time()
        h.add("", "a", timestamp=now)
        h.add("", "b", timestamp=now - 3 * 86400)  # gap
        assert h.stats(now=now).streak_days == 1

    def test_empty_stats(self):
        assert History(":memory:").stats().total_words == 0
