import pytest
import torch
from transformers import AutoTokenizer

from src.model import CMCModule
from src.utils import BatchTest, PrefixAllowedTokens

torch.manual_seed(42)


@pytest.fixture
def default_setting():
    model = CMCModule(
        model_configuration="encoder_decoder",
        diff_tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased"),
        msg_tokenizer=AutoTokenizer.from_pretrained("distilgpt2"),
        encoder_name_or_path="distilbert-base-uncased",
        decoder_name_or_path="distilgpt2",
        encoder_context_max_len=512,
        decoder_context_max_len=256,
        batch_size=1,
    )
    return model, {"num_beams": 4, "num_return_sequences": 4}


@pytest.mark.parametrize(
    "context,prefix,expected",
    [
        ("", "Firs", "First"),
        ("GPT-2 is a generative", " lan", " language"),
        ("My twitter", " userna", " username"),
    ],
)
def test_with_and_without_prefix_fn(default_setting, context, prefix, expected):
    model, generation_kwargs = default_setting

    if not context:
        context = model._msg_tokenizer.eos_token

    tokenized_context = model._msg_tokenizer(context, return_tensors="pt").input_ids
    tokenized_context_w_prefix = model._msg_tokenizer(context + prefix, return_tensors="pt").input_ids

    min_len = 2
    max_len = 2

    results_with_prefix_fn = model.generate(
        batch=BatchTest(
            encoder_input_ids=torch.zeros_like(tokenized_context),
            encoder_attention_mask=torch.zeros_like(tokenized_context),
            decoder_input_ids=tokenized_context,
            decoder_attention_mask=torch.ones_like(tokenized_context),
            prefixes=[prefix],
            targets=[],
            labels=None,
        ),
        min_length=min_len + tokenized_context.shape[1],
        max_length=max_len + tokenized_context.shape[1],
        **generation_kwargs
    )
    sequences_with_prefix_fn = model._msg_tokenizer.batch_decode(
        results_with_prefix_fn[:, tokenized_context.shape[1] :]
    )

    results_without_prefix_fn = model.generate(
        batch=BatchTest(
            encoder_input_ids=torch.zeros_like(tokenized_context_w_prefix),
            encoder_attention_mask=torch.zeros_like(tokenized_context_w_prefix),
            decoder_input_ids=tokenized_context_w_prefix,
            decoder_attention_mask=torch.ones_like(tokenized_context_w_prefix),
            prefixes=[],
            targets=[],
            labels=None,
        ),
        min_length=min_len + tokenized_context_w_prefix.shape[1],
        max_length=max_len + tokenized_context_w_prefix.shape[1],
        **generation_kwargs
    )
    sequences_without_prefix_fn = model._msg_tokenizer.batch_decode(
        results_without_prefix_fn[:, tokenized_context.shape[1] :]
    )

    assert any([seq.startswith(expected) for seq in sequences_with_prefix_fn])
    assert not any([seq.startswith(expected) for seq in sequences_without_prefix_fn])


@pytest.mark.parametrize(
    "context,prefix,generated",
    [
        ("GPT-2 is a generative", " la", ""),
        ("My twitter", " userna", ""),
        ("Hello", " wor", ""),
    ],
)
def test_generation_empty_prefix(default_setting, context, prefix, generated):
    model, generation_kwargs = default_setting

    beam_sentence = context + generated

    tokenized_beam_sentence = model._msg_tokenizer(beam_sentence, return_tensors="pt").input_ids[0]
    prefix_fn = PrefixAllowedTokens(prefix={0: ""}, context_len={0: 0}, tokenizer=model._msg_tokenizer)

    allowed_tokens = model._msg_tokenizer.batch_decode(prefix_fn(0, sentence=tokenized_beam_sentence))
    assert len(allowed_tokens) == len(model._msg_tokenizer.get_vocab().keys())


@pytest.mark.parametrize(
    "context,prefix,generated",
    [
        ("", "Firs", ""),
        ("GPT-2 is a generative", " la", ""),
        ("My twitter", " userna", ""),
        ("Hello", " wor", ""),
    ],
)
def test_generation_start(default_setting, context, prefix, generated):
    model, generation_kwargs = default_setting

    beam_sentence = context + generated

    tokenized_context = model._msg_tokenizer(context, return_tensors="pt").input_ids
    tokenized_beam_sentence = model._msg_tokenizer(beam_sentence, return_tensors="pt").input_ids[0]
    prefix_fn = PrefixAllowedTokens(
        prefix={0: prefix}, context_len={0: tokenized_context.shape[1]}, tokenizer=model._msg_tokenizer
    )

    allowed_tokens = model._msg_tokenizer.batch_decode(prefix_fn(0, sentence=tokenized_beam_sentence))
    for token in allowed_tokens:
        assert token.startswith(prefix) or prefix.startswith(token)


@pytest.mark.parametrize(
    "context,prefix,generated,remaining",
    [
        ("update to version", " 3.0", " 3", ".0"),
        ("", "GPT-2", "GPT", "-2"),
        ("GPT-2 is a generative", " la", " l", "a"),
        ("My twitter", " userna", " user", "na"),
        ("Hello", " wor", " wo", "r"),
    ],
)
def test_generation_prefix_part(default_setting, context, prefix, generated, remaining):
    model, generation_kwargs = default_setting

    beam_sentence = context + generated

    tokenized_context = model._msg_tokenizer(context, return_tensors="pt").input_ids
    tokenized_beam_sentence = model._msg_tokenizer(beam_sentence, return_tensors="pt").input_ids[0]
    prefix_fn = PrefixAllowedTokens(
        prefix={0: prefix}, context_len={0: tokenized_context.shape[1]}, tokenizer=model._msg_tokenizer
    )

    allowed_tokens = model._msg_tokenizer.batch_decode(prefix_fn(0, sentence=tokenized_beam_sentence))
    assert all(token.startswith(remaining) or remaining.startswith(token) for token in allowed_tokens)


@pytest.mark.parametrize(
    "context,prefix,generated",
    [
        ("GPT-2 is a generative", " la", " language model"),
        ("My twitter", " userna", " username"),
        ("Hello", " wor", " wor"),
    ],
)
def test_generation_whole_prefix(default_setting, context, prefix, generated):
    model, generation_kwargs = default_setting

    beam_sentence = context + generated

    tokenized_context = model._msg_tokenizer(context, return_tensors="pt").input_ids
    tokenized_beam_sentence = model._msg_tokenizer(beam_sentence, return_tensors="pt").input_ids[0]
    prefix_fn = PrefixAllowedTokens(
        prefix={0: prefix}, context_len={0: tokenized_context.shape[1]}, tokenizer=model._msg_tokenizer
    )

    allowed_tokens = model._msg_tokenizer.batch_decode(prefix_fn(0, sentence=tokenized_beam_sentence))
    assert len(allowed_tokens) == len(model._msg_tokenizer.get_vocab().keys())
