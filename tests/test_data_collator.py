import pytest
from transformers import AutoTokenizer

from src.data_utils.data_collator import DataCollator


@pytest.fixture
def default_tokenizers():
    encoder_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    decoder_tok = AutoTokenizer.from_pretrained("distilgpt2")
    decoder_tok.pad_token = decoder_tok.eos_token

    return encoder_tok, decoder_tok


@pytest.mark.parametrize(
    "input,target,expected_context,expected_prefix",
    [
        ("version ", "3.0.0", "version ", ""),
        ("version 3", ".0.0", "version", " 3"),
        ("Resi", "dence", "", "Resi"),
        ("long long sentence last word spl", "it", "long long sentence last word", " spl"),
    ],
)
def test_get_prefix(default_tokenizers, input, target, expected_context, expected_prefix):
    encoder_tok, decoder_tok = default_tokenizers
    message_ids = decoder_tok(input + target, padding=False, truncation=False).input_ids

    sep_tokens_id = decoder_tok.convert_tokens_to_ids("Ċ")

    data_collator = DataCollator(
        diff_tokenizer=encoder_tok,
        msg_tokenizer=decoder_tok,
        encoder_context_max_len=500,
        decoder_context_max_len=200,
        with_history=True,
        decoder_sep_tokens=[sep_tokens_id],
        context_ratio=0.1,
        generation=True,
    )

    res = data_collator._get_prefix(message_ids, context_len=len(input))

    assert decoder_tok.decode(res["msg_input_ids"]) == expected_context
    assert res["msg_prefix"] == expected_prefix


@pytest.mark.parametrize(
    "msgs,diffs,histories",
    [
        (
            ["Residence certainly elsewhere something she preferred cordially law."],
            ["- sample diff\n+ sample diff\n"],
            [[f"old message {i}" for i in range(200)]],
        ),
        (
            [
                "Residence certainly elsewhere something she preferred cordially law.",
                "The Roman Bridge (German: Römerbrücke) is an ancient structure in Trier, Germany, over the Moselle. It "
                "is the oldest standing bridge in the country. The nine bridge pillars date from the 2nd century AD.",
            ],
            ["- sample diff\n+ sample diff", "- sample diff\n+ sample diff"],
            [[f"old message {i}" for i in range(200)], []],
        ),
    ],
)
def test_generation_collator_with_history(msgs, diffs, histories, default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers
    inputs = []
    for diff, msg, history in zip(diffs, msgs, histories):
        diff_ids = encoder_tok(diff, padding=False, truncation=True).input_ids
        msgs_ids = decoder_tok(msg, padding=False, truncation=False).input_ids
        if history:
            history_ids = decoder_tok(history, padding=False, truncation=False).input_ids
        else:
            history_ids = []
        inputs.append({"diff_input_ids": diff_ids, "msg_input_ids": msgs_ids, "history_input_ids": history_ids})

    sep_tokens_id = decoder_tok.convert_tokens_to_ids("Ċ")

    for context_ratio in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:

        data_collator = DataCollator(
            diff_tokenizer=encoder_tok,
            msg_tokenizer=decoder_tok,
            encoder_context_max_len=500,
            decoder_context_max_len=200,
            with_history=True,
            decoder_sep_tokens=[sep_tokens_id],
            context_ratio=context_ratio,
            generation=True,
        )
        res = data_collator(inputs)

        assert res["diff_input_ids"].shape[1] <= 500
        assert res["msg_input_ids"].shape[1] <= 200

        for context, prefix, target, msg in zip(res["msg_input_ids"], res["msg_prefix"], res["msg_target"], msgs):
            if sep_tokens_id in context:
                last_nl_idx = (context == sep_tokens_id).nonzero(as_tuple=True)[0][-1].item()
                decoded_context = decoder_tok.decode(context[last_nl_idx + 1 :], skip_special_tokens=True)
            else:  # \n might not be present if history is empty
                decoded_context = decoder_tok.decode(context, skip_special_tokens=True)
            assert decoded_context + prefix + target == msg


@pytest.mark.parametrize(
    "msgs,diffs",
    [
        (["Residence certainly elsewhere something she preferred cordially law."], ["- sample diff\n+ sample diff"]),
        (
            [
                "Residence certainly elsewhere something she preferred cordially law.",
                "The Roman Bridge (German: Römerbrücke) is an ancient structure in Trier, Germany, over the Moselle. It "
                "is the oldest standing bridge in the country. The nine bridge pillars date from the 2nd century AD.",
            ],
            ["- sample diff\n+ sample diff", "- sample diff\n+ sample diff"],
        ),
    ],
)
def test_generation_collator_without_history(msgs, diffs, default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers
    inputs = []
    for diff, msg in zip(diffs, msgs):
        diff_ids = encoder_tok(diff, padding=False, truncation=True).input_ids
        msgs_ids = decoder_tok(msg, padding=False, truncation=False).input_ids
        inputs.append({"diff_input_ids": diff_ids, "msg_input_ids": msgs_ids, "history_input_ids": []})

    sep_tokens_id = decoder_tok.convert_tokens_to_ids("Ċ")

    for context_ratio in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:

        data_collator = DataCollator(
            diff_tokenizer=encoder_tok,
            msg_tokenizer=decoder_tok,
            encoder_context_max_len=500,
            decoder_context_max_len=200,
            with_history=False,
            decoder_sep_tokens=[sep_tokens_id],
            context_ratio=context_ratio,
            generation=True,
        )
        res = data_collator(inputs)

        assert res["diff_input_ids"].shape[1] <= 500
        assert res["msg_input_ids"].shape[1] <= 200

        for context, prefix, target, msg in zip(res["msg_input_ids"], res["msg_prefix"], res["msg_target"], msgs):
            decoded_context = decoder_tok.decode(context, skip_special_tokens=True)
            assert decoded_context + prefix + target == msg


@pytest.mark.parametrize(
    "msgs,diffs",
    [
        (["Residence certainly elsewhere something she preferred cordially law."], ["- sample diff\n+ sample diff"]),
        (
            [
                "Residence certainly elsewhere something she preferred cordially law.",
                "The Roman Bridge (German: Römerbrücke) is an ancient structure in Trier, Germany, over the Moselle. It "
                "is the oldest standing bridge in the country. The nine bridge pillars date from the 2nd century AD.",
            ],
            ["- sample diff\n+ sample diff", "- sample diff\n+ sample diff"],
        ),
    ],
)
def test_training_collator_without_history(msgs, diffs, default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers
    inputs = []
    for diff, msg in zip(diffs, msgs):
        diff_ids = encoder_tok(diff, padding=False, truncation=True).input_ids
        msgs_ids = decoder_tok(msg, padding=False, truncation=False).input_ids
        inputs.append({"diff_input_ids": diff_ids, "msg_input_ids": msgs_ids, "history_input_ids": []})

    sep_token_id = decoder_tok.convert_tokens_to_ids("Ċ")

    data_collator = DataCollator(
        diff_tokenizer=encoder_tok,
        msg_tokenizer=decoder_tok,
        encoder_context_max_len=500,
        decoder_context_max_len=200,
        with_history=False,
        decoder_sep_tokens=[sep_token_id],
        generation=False,
    )
    res = data_collator(inputs)

    assert res["diff_input_ids"].shape[1] <= 500
    assert res["msg_input_ids"].shape[1] <= 200

    assert res["diff_input_ids"][0, 0] == encoder_tok.bos_token_id
    assert res["diff_input_ids"][res["diff_attention_mask"] == 1][-1] == encoder_tok.eos_token_id

    assert res["msg_input_ids"][0, 0] == decoder_tok.eos_token_id
    assert res["msg_input_ids"][res["msg_attention_mask"] == 1][-1] == sep_token_id
