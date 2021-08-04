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


@pytest.mark.parametrize(
    "decoder_context,prefix,expected",
    [
        ("GPT-2 is generative la", " la", "GPT - 2 is generative"),
        ("history <nl> message with pref", " pref", "history \n message with"),
        ("whatever you want", None, "whatever you want"),
    ],
)
def test_correct_prefixes(default_data_processor, decoder_context, prefix, expected):
    tokenized_decoder_context = default_data_processor.prepare_decoder_input(
        decoder_context=decoder_context, prefix=prefix
    )
    assert (
        default_data_processor._msg_tokenizer.batch_decode(tokenized_decoder_context, skip_special_tokens=True)[0]
        == expected
    )


@pytest.mark.parametrize(
    "decoder_context,prefix",
    [
        ("prefix might appear somewhere but it should be the last word", " appear"),
        ("prefix might appear in history <nl> but it should be in the last message", " history"),
    ],
)
def test_wrong_prefixes(default_data_processor, decoder_context, prefix):
    with pytest.raises(ValueError):
        default_data_processor.prepare_decoder_input(decoder_context=decoder_context, prefix=prefix)
