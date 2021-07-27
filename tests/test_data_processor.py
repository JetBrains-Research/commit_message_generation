import pytest
from seq2seq_completion.data_utils import DataProcessor


@pytest.fixture
def default_data_processor():
    default_config = {
        "prompt_max_len": 200,
        "diff_tokenizer_name_or_path": "microsoft/codebert-base",
        "msg_tokenizer_name_or_path": "distilgpt2",
        "preprocessing": True,
        "nl_token": "<nl>",
    }
    return DataProcessor(**default_config)


@pytest.mark.parametrize(
    "test_input,expected_output",
    [
        (
            "<FILE> some/path <nl> - smth.old{} <nl> + smth.new() <nl> unchanged line",
            "some / path \n - smth . old { } \n + smth . new ( ) \n",
        ),
        (
            "Binary files some/path/a and some/path/b differ <nl>",
            "Binary files some / path / a and some / path / b differ \n",
        ),
        ("", ""),
    ],
)
def test_preprocess_diff(default_data_processor, test_input, expected_output):
    assert default_data_processor.preprocess_diff(test_input) == expected_output


@pytest.mark.parametrize(
    "test_input,expected_output",
    [
        (
            "docs: update.all-contributorsrc [skip ci] <nl> Authored-by: someone <user@mail.com> <nl>",
            "docs : update . all - contributorsrc [ skip ci ] \n Authored - by : someone < user @ mail " ". com > \n",
        ),
        ("", ""),
    ],
)
def test_preprocess_msg(default_data_processor, test_input, expected_output):
    assert default_data_processor.preprocess_msg(test_input) == expected_output
