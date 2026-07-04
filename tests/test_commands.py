from localflow.commands import CommandProcessor


class TestCommandMode:
    def setup_method(self):
        self.proc = CommandProcessor()

    def test_uppercase(self):
        assert self.proc.apply("make this uppercase", "hello") == "HELLO"

    def test_lowercase(self):
        assert self.proc.apply("lowercase please", "HELLO") == "hello"

    def test_title_case(self):
        assert self.proc.apply("title case", "hello world") == "Hello World"

    def test_bullet_list(self):
        out = self.proc.apply("make this a bullet list", "apples, bananas, cherries")
        assert out == "- apples\n- bananas\n- cherries"

    def test_bullets_from_and(self):
        out = self.proc.apply("bullet points", "eggs and milk and bread")
        assert out.count("- ") == 3

    def test_numbered_list(self):
        out = self.proc.apply("numbered list", "first, second")
        assert out.startswith("1. first")
        assert "2. second" in out

    def test_one_line(self):
        assert self.proc.apply("make it one line", "a\nb\nc") == "a b c"

    def test_snake_case(self):
        assert self.proc.apply("snake case", "My Variable Name") == "my_variable_name"

    def test_camel_case(self):
        assert self.proc.apply("camel case that", "my variable name") == "myVariableName"

    def test_kebab_case(self):
        assert self.proc.apply("kebab case", "My Blog Post") == "my-blog-post"

    def test_fix_punctuation(self):
        out = self.proc.apply("fix the punctuation", "hello world how are you")
        assert out[0].isupper()
        assert out.endswith(".")

    def test_remove_fillers(self):
        assert "um" not in self.proc.apply("remove fillers", "so um yeah")

    def test_shorten(self):
        long = "First sentence here. Second sentence here. Third one. Fourth one."
        out = self.proc.apply("make it shorter", long)
        assert len(out) < len(long)

    def test_quotes(self):
        assert self.proc.apply("wrap in quotes", "hello") == '"hello"'

    def test_unknown_instruction(self):
        assert self.proc.apply("translate to klingon", "hello") is None

    def test_empty_inputs(self):
        assert self.proc.apply("", "text") is None
        assert self.proc.apply("uppercase", "") is None
