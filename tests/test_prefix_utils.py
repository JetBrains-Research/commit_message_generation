import pytest
import torch
from transformers import AutoTokenizer
from seq2seq_completion.model import EncoderDecoder
from seq2seq_completion.model.prefix_utils import PrefixConstrainedLogitsProcessor

torch.manual_seed(42)


@pytest.fixture
def default_setting():
    model = EncoderDecoder(encoder_name_or_path="distilbert-base-uncased", decoder_name_or_path="distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    return model, tokenizer, {"num_beams": 4, "num_return_sequences": 1, "min_length": 1, "max_length": 1}


@pytest.mark.parametrize(
    "context,prefix,expected",
    [
        ("GPT-2 is a generative", " la", " language"),
        ("My twitter", " userna", " username"),
        ("Hello", " wor", " world"),
    ],
)
def test_with_and_without_prefix(default_setting, context, prefix, expected):
    model, tokenizer, generation_kwargs = default_setting
    tokenized_input = tokenizer(context, return_tensors="pt").input_ids
    generation_kwargs["min_length"] += tokenized_input.shape[1]
    generation_kwargs["max_length"] += tokenized_input.shape[1]

    results_with_prefix = model.generate(
        input_ids=tokenized_input, prefix=prefix, tokenizer=tokenizer, **generation_kwargs
    )
    sequences_with_prefix = tokenizer.batch_decode(results_with_prefix["sequences"][:, tokenized_input.shape[1] :])

    results_without_prefix = model.generate(input_ids=tokenized_input, **generation_kwargs)
    sequences_without_prefix = tokenizer.batch_decode(
        results_without_prefix["sequences"][:, tokenized_input.shape[1] :]
    )

    assert expected in sequences_with_prefix
    assert expected not in sequences_without_prefix


def test_prefix_allowed_tokens_fn(default_setting):
    _, tokenizer, generation_kwargs = default_setting

    context = "context"
    prefix = " prefix"

    logits_processor = PrefixConstrainedLogitsProcessor(
        prefix=prefix, tokenizer=tokenizer, num_beams=generation_kwargs["num_beams"]
    )

    results = logits_processor._prefix_allowed_tokens_fn(
        beam_id=0, sentence=tokenizer(context, return_tensors="pt").input_ids[0]
    )
    assert len(results) < len(tokenizer.get_vocab().keys())

    results = logits_processor._prefix_allowed_tokens_fn(
        beam_id=0, sentence=tokenizer(context + prefix, return_tensors="pt").input_ids[0]
    )
    assert len(results) == len(tokenizer.get_vocab().keys())
