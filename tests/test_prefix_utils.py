import pytest
import torch
from transformers import AutoTokenizer
from seq2seq_completion.model import EncoderDecoder
from seq2seq_completion.model.prefix_utils import PrefixAllowedTokens

torch.manual_seed(42)


@pytest.fixture
def default_setting():
    model = EncoderDecoder(encoder_name_or_path="distilbert-base-uncased", decoder_name_or_path="distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    return model, tokenizer, {"num_beams": 4, "num_return_sequences": 4}


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
    tokenized_context = tokenizer(context, return_tensors="pt").input_ids
    tokenized_context_w_prefix = tokenizer(context + prefix, return_tensors="pt").input_ids

    min_len = 5
    max_len = 5

    results_with_prefix = model.generate(
        input_ids=tokenized_context,
        prefix=prefix,
        tokenizer=tokenizer,
        min_length=min_len + tokenized_context.shape[1],
        max_length=max_len + tokenized_context.shape[1],
        **generation_kwargs
    )
    sequences_with_prefix = tokenizer.batch_decode(results_with_prefix["sequences"][:, tokenized_context.shape[1] :])

    results_without_prefix = model.generate(
        input_ids=tokenized_context_w_prefix,
        tokenizer=tokenizer,
        min_length=min_len + tokenized_context_w_prefix.shape[1],
        max_length=max_len + tokenized_context_w_prefix.shape[1],
        **generation_kwargs
    )
    sequences_without_prefix = tokenizer.batch_decode(
        results_without_prefix["sequences"][:, tokenized_context.shape[1] :]
    )

    assert any([seq.startswith(expected) for seq in sequences_with_prefix])
    assert not any([seq.startswith(expected) for seq in sequences_without_prefix])


@pytest.mark.parametrize(
    "context,prefix,generated",
    [
        ("GPT-2 is a generative", " la", ""),
        ("My twitter", " userna", ""),
        ("Hello", " wor", ""),
    ],
)
def test_no_prefix(default_setting, context, prefix, generated):
    _, tokenizer, generation_kwargs = default_setting

    beam_sentence = context + generated

    tokenized_context = tokenizer(context, return_tensors="pt").input_ids
    tokenized_beam_sentence = tokenizer(beam_sentence, return_tensors="pt").input_ids[0]
    prefix_fn = PrefixAllowedTokens(prefix=prefix, context_len=tokenized_context.shape[1], tokenizer=tokenizer)

    allowed_tokens = tokenizer.batch_decode(prefix_fn(0, sentence=tokenized_beam_sentence))
    for token in allowed_tokens:
        assert token.startswith(prefix) or prefix.startswith(token)


@pytest.mark.parametrize(
    "context,prefix,generated,remaining",
    [
        ("GPT-2 is a generative", " la", " l", "a"),
        ("My twitter", " userna", " user", "na"),
        ("Hello", " wor", " wo", "r"),
    ],
)
def test_prefix_part(default_setting, context, prefix, generated, remaining):
    _, tokenizer, generation_kwargs = default_setting

    beam_sentence = context + generated

    tokenized_context = tokenizer(context, return_tensors="pt").input_ids
    tokenized_beam_sentence = tokenizer(beam_sentence, return_tensors="pt").input_ids[0]
    prefix_fn = PrefixAllowedTokens(prefix=prefix, context_len=tokenized_context.shape[1], tokenizer=tokenizer)

    allowed_tokens = tokenizer.batch_decode(prefix_fn(0, sentence=tokenized_beam_sentence))
    for token in allowed_tokens:
        assert token.startswith(remaining) or remaining.startswith(token)


@pytest.mark.parametrize(
    "context,prefix,generated",
    [
        ("GPT-2 is a generative", " la", " language model"),
        ("My twitter", " userna", " username"),
        ("Hello", " wor", " wor"),
    ],
)
def test_whole_prefix(default_setting, context, prefix, generated):
    _, tokenizer, generation_kwargs = default_setting

    beam_sentence = context + generated

    tokenized_context = tokenizer(context, return_tensors="pt").input_ids
    tokenized_beam_sentence = tokenizer(beam_sentence, return_tensors="pt").input_ids[0]
    prefix_fn = PrefixAllowedTokens(prefix=prefix, context_len=tokenized_context.shape[1], tokenizer=tokenizer)

    allowed_tokens = tokenizer.batch_decode(prefix_fn(0, sentence=tokenized_beam_sentence))
    assert len(allowed_tokens) == len(tokenizer.get_vocab().keys())
