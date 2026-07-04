from localflow.config import FormattingConfig
from localflow.formatting import (
    apply_self_corrections,
    apply_spoken_commands,
    apply_spoken_punctuation,
    capitalize_sentences,
    convert_numbers,
    format_emails,
    format_transcript,
    remove_fillers,
    smart_join,
)


class TestFillers:
    def test_removes_standalone_um_uh(self):
        assert remove_fillers("So um I think uh we should go") == "So I think we should go"

    def test_removes_fillers_with_commas(self):
        assert remove_fillers("Well, um, let's start.") == "Well, let's start."

    def test_keeps_words_containing_fillers(self):
        assert remove_fillers("The umbrella is under the summer sun") == \
            "The umbrella is under the summer sun"

    def test_filler_at_start(self):
        assert remove_fillers("Um, hello there") .strip() == "hello there"

    def test_hmm_and_erm(self):
        assert remove_fillers("Hmm, that's erm interesting") .strip() == "that's interesting"


class TestSelfCorrection:
    def test_scratch_that(self):
        out = apply_self_corrections("Send it to John, scratch that, send it to Sarah")
        assert "John" not in out
        assert "Sarah" in out

    def test_no_wait(self):
        out = apply_self_corrections("Meet me at five, no wait, six o'clock")
        assert "five" not in out
        assert "six" in out

    def test_i_mean(self):
        out = apply_self_corrections("Let's use Python, I mean TypeScript for this")
        assert "Python" not in out
        assert "TypeScript" in out

    def test_no_cue_no_change(self):
        text = "The meeting is at five and dinner is at eight."
        assert apply_self_corrections(text) == text

    def test_shared_prefix_kept(self):
        out = apply_self_corrections("Meet me at five, I mean six")
        assert out.startswith("Meet me at")

    def test_cue_after_sentence_boundary_retracts_sentence(self):
        out = apply_self_corrections("Send the report. Scratch that, email it instead.")
        assert "report" not in out
        assert out.strip().startswith("email it instead")

    def test_no_glued_words_after_correction(self):
        out = apply_self_corrections("The meeting is at 3.30. No wait, four o'clock.")
        assert ".f" not in out.replace(". f", "")  # words must not fuse
        assert "four o'clock" in out


class TestSpokenCommands:
    def test_new_line(self):
        assert apply_spoken_commands("first item new line second item") == \
            "first item\nsecond item"

    def test_new_paragraph(self):
        out = apply_spoken_commands("Intro done. New paragraph. Next topic")
        assert "\n\n" in out

    def test_punctuation_words(self):
        out = apply_spoken_punctuation("hello comma world period")
        assert out.strip() in ("hello, world.", "hello , world .".replace(" ,", ","))
        assert "," in out and "." in out


class TestNumbers:
    def test_two_word_number(self):
        assert convert_numbers("we saw twenty three birds") == "we saw 23 birds"

    def test_single_small_number_untouched(self):
        assert convert_numbers("I have one idea") == "I have one idea"

    def test_hundreds(self):
        assert convert_numbers("three hundred and forty two people") == "342 people"

    def test_thousands(self):
        assert convert_numbers("about twelve thousand five hundred") == "about 12500"

    def test_percent(self):
        assert convert_numbers("forty five percent done") == "45% done"

    def test_dollars(self):
        assert convert_numbers("costs twenty five dollars") == "costs $25"


class TestEmails:
    def test_simple_email(self):
        assert format_emails("send it to john at gmail dot com") == \
            "send it to john@gmail.com"

    def test_dotted_user(self):
        assert format_emails("jane dot doe at company dot co dot uk") == \
            "jane.doe@company.co.uk"

    def test_plain_at_not_converted(self):
        # "at" without a dotted domain should never become @
        assert "@" not in format_emails("meet me at noon")


class TestCapitalization:
    def test_sentence_starts(self):
        assert capitalize_sentences("hello. how are you? fine") == \
            "Hello. How are you? Fine"

    def test_standalone_i(self):
        assert capitalize_sentences("you and i went, i'm happy") == \
            "You and I went, I'm happy"


class TestPipeline:
    def test_full_pipeline(self):
        raw = "um so the meeting is at three thirty, no wait, four. new line remember to email bob at example dot com"
        out = format_transcript(raw, FormattingConfig())
        assert "um" not in out.lower().split()
        assert "three" not in out
        assert "\n" in out
        assert "bob@example.com" in out

    def test_disabled_returns_stripped(self):
        cfg = FormattingConfig(enabled=False)
        assert format_transcript("  hello um world  ", cfg) == "hello um world"

    def test_overrides(self):
        cfg = FormattingConfig(capitalize_sentences=True)
        out = format_transcript("hello world", cfg, overrides={"capitalize_sentences": False})
        assert out == "hello world"

    def test_terminal_punctuation(self):
        cfg = FormattingConfig(ensure_terminal_punctuation=True)
        assert format_transcript("hello world", cfg).endswith(".")


class TestSmartJoin:
    def test_continues_sentence_lowercase(self):
        assert smart_join("I went to the store", "And bought milk") == "and bought milk"

    def test_new_sentence_keeps_case(self):
        assert smart_join("Done here.", "Next thing") == "Next thing"

    def test_preserves_i(self):
        assert smart_join("you and", "I agree") == "I agree"

    def test_empty_previous(self):
        assert smart_join("", "Hello") == "Hello"
