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
            "some/path\n-smth.old{}\n+smth.new()\nunchanged line",
            "some/path\n-smth.old{}\n+smth.new()",
        ),
        (
            "Binary files some/path/a and some/path/b differ\n",
            "Binary files some/path/a and some/path/b differ",
        ),
        ("", ""),
    ],
)
def test_preprocess_diff(default_data_processor, test_input, expected_output):
    assert default_data_processor.preprocess_diff(test_input) == expected_output
