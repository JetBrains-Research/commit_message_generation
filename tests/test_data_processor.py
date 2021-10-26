import pytest
from seq2seq_completion.data_utils import DataProcessor


@pytest.fixture
def default_data_processor():
    default_config = {
        "prompt_max_len": 200,
        "diff_tokenizer_name_or_path": "microsoft/codebert-base",
        "msg_tokenizer_name_or_path": "distilgpt2",
        "preprocessing": True,
    }
    return DataProcessor(**default_config)


@pytest.mark.parametrize(
    "test_input,expected_output",
    [
        (
            # modifying a file
            "diff --git a/some/path b/some/path\n"
            "--- a/some/path\n"
            "+++ b/some/path\n"
            "@@ -2,7 +2,6 @@\n"
            "-smth.old{}\n"
            "+smth.new()\n"
            "unchanged line\n"
            "\\ No newline at end of file\n",
            "some/path\n" "-smth.old{}\n" "+smth.new()",
        ),
        (
            # deleting a file
            "diff --git a/README.md b/README.md\n"
            "deleted file mode 100644\n"
            "--- a/README.md (revision xxx)\n"
            "+++ /dev/null (revision xxx)\n"
            "@@ -1,4 +0,0 @@\n"
            "-# Commit messages completion ~~(and generation)~~\n",
            "deleted file README.md\n" "-# Commit messages completion ~~(and generation)~~",
        ),
        (
            # creating new file
            "diff --git a/LICENSE b/LICENSE\n"
            "new file mode 100644\n"
            "--- /dev/null (date yyy)\n"
            "+++ b/LICENSE (date yyy)\n"
            "@@ -0,0 +1,201 @@\n"
            "+                                Apache License\n"
            "+                           Version 2.0, January 2004\n"
            "+                        http://www.apache.org/licenses/\n",
            "new file LICENSE\n"
            "+ Apache License\n"
            "+ Version 2.0, January 2004\n"
            "+ http://www.apache.org/licenses/",
        ),
        (
            # renaming a file
            "diff --git a/some/path b/some/path\n" "rename from a/some/path\n" "rename to b/some/path\n",
            "rename from a/some/path\n" "rename to b/some/path",
        ),
        (
            # modifying binary files
            "Binary files some/path/a and some/path/b differ\n",
            "Binary files some/path/a and some/path/b differ",
        ),
        (
            # trivial empty case
            "",
            "",
        ),
    ],
)
def test_preprocess_diff(default_data_processor, test_input, expected_output):
    assert default_data_processor.preprocess_diff(test_input) == expected_output
