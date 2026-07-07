from localflow.dictionary import PersonalDictionary


class TestDictionary:
    def test_add_remove(self):
        d = PersonalDictionary()
        d.add("Wispr")
        d.add("Wispr")  # dedupe
        assert d.words == ["Wispr"]
        assert d.remove("Wispr") is True
        assert d.remove("Wispr") is False

    def test_initial_prompt(self):
        d = PersonalDictionary(words=["Anthropic", "kubectl"])
        prompt = d.initial_prompt()
        assert "Anthropic" in prompt and "kubectl" in prompt
        assert PersonalDictionary().initial_prompt() is None

    def test_exact_case_fix(self):
        d = PersonalDictionary(words=["GitHub"])
        assert d.correct("push it to github now") == "push it to GitHub now"

    def test_fuzzy_correction(self):
        d = PersonalDictionary(words=["Wispr"])
        assert d.correct("I love wisper flow") == "I love Wispr flow"

    def test_short_tokens_not_fuzzed(self):
        d = PersonalDictionary(words=["Nate"])
        # "not" is too close to "Nate" for careless fuzzing but too short to touch
        assert d.correct("it is not here") == "it is not here"

    def test_phrase_entries(self):
        d = PersonalDictionary(words=["Wispr Flow"])
        assert d.correct("i use wispr flow daily") == "i use Wispr Flow daily"

    def test_unrelated_text_untouched(self):
        d = PersonalDictionary(words=["Kubernetes"])
        text = "the quick brown fox jumps"
        assert d.correct(text) == text

    def test_replacements(self):
        d = PersonalDictionary(replacements={"brb": "be right back", "eta": "ETA"})
        assert d.apply_replacements("brb, what is the eta") == \
            "be right back, what is the ETA"

    def test_correct_applies_replacements_too(self):
        d = PersonalDictionary(words=["Slack"], replacements={"omw": "on my way"})
        assert d.correct("omw to slack") == "on my way to Slack"


class TestSuggestions:
    def test_mines_recurring_names(self):
        from localflow.dictionary import mine_suggestions

        texts = [
            "I talked to Beketov about LangChain today.",
            "Beketov said the LangChain docs are rough.",
            "Later Beketov pinged me again about LangChain.",
        ]
        out = dict(mine_suggestions(texts, known=[], min_count=2))
        assert out["LangChain"] == 3
        assert out["Beketov"] == 2  # one occurrence is sentence-initial

    def test_known_words_and_common_words_excluded(self):
        from localflow.dictionary import mine_suggestions

        texts = ["See you Monday with Anthropic.",
                 "Talked to Anthropic on Monday.",
                 "Anthropic again, this Monday."]
        out = dict(mine_suggestions(texts, known=["Anthropic"], min_count=2))
        assert "Anthropic" not in out
        assert "Monday" not in out
