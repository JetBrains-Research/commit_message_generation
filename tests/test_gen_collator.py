import pytest
from src.dataset_utils import GenerationCollator
from transformers import AutoTokenizer


@pytest.fixture
def tokenizers():
    encoder_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    decoder_tok = AutoTokenizer.from_pretrained("distilgpt2")
    decoder_tok.pad_token = decoder_tok.eos_token

    return encoder_tok, decoder_tok


@pytest.mark.parametrize(
    "msgs,diffs,histories",
    [
        (
            ["Residence certainly elsewhere something she preferred cordially law."],
            ["- sample diff \n + sample diff"],
            [[f"old message {i}" for i in range(200)]],
        ),
        (
            [
                "Residence certainly elsewhere something she preferred cordially law.",
                "The Roman Bridge (German: Römerbrücke) is an ancient structure in Trier, Germany, over the Moselle. It "
                "is the oldest standing bridge in the country. The nine bridge pillars date from the 2nd century AD.",
            ],
            ["- sample diff \n + sample diff", "- sample diff \n + sample diff"],
            [[f"old message {i}" for i in range(200)], []],
        ),
    ],
)
def test_with_history(msgs, diffs, histories, tokenizers):
    encoder_tok, decoder_tok = tokenizers
    inputs = []
    for diff, msg, history in zip(diffs, msgs, histories):
        diff_ids = encoder_tok(diff, padding=False, truncation=True).input_ids
        msgs_ids = decoder_tok(msg, padding=False, truncation=False).input_ids
        if history:
            history_ids = decoder_tok(history, padding=False, truncation=False).input_ids
        else:
            history_ids = []
        inputs.append({"diff_input_ids": diff_ids, "msg_input_ids": msgs_ids, "history_input_ids": history_ids})

    data_collator = GenerationCollator(
        context_ratio=0.1, src_tokenizer=encoder_tok, trg_tokenizer=decoder_tok, max_len=200, with_history=True
    )
    res = data_collator(inputs)

    assert res["diff_input_ids"].shape[1] <= 500
    assert res["msg_input_ids"].shape[1] <= 200

    nl_token_id = decoder_tok(r"\n").input_ids[1]
    for context, prefix, target, msg in zip(res["msg_input_ids"], res["prefix"], res["target"], msgs):
        if nl_token_id in context:
            last_nl_idx = (context == nl_token_id).nonzero(as_tuple=True)[0][-1].item()
            decoded_context = decoder_tok.decode(context[last_nl_idx + 2 :], skip_special_tokens=True)
        else:  # \n might not be present if history is empty
            decoded_context = decoder_tok.decode(context, skip_special_tokens=True)
        decoded_target = decoder_tok.decode(target, skip_special_tokens=True)
        assert decoded_context + decoded_target == msg


@pytest.mark.parametrize(
    "msgs,diffs",
    [
        (["Residence certainly elsewhere something she preferred cordially law."], ["- sample diff \n + sample diff"]),
        (
            [
                "Residence certainly elsewhere something she preferred cordially law.",
                "The Roman Bridge (German: Römerbrücke) is an ancient structure in Trier, Germany, over the Moselle. It "
                "is the oldest standing bridge in the country. The nine bridge pillars date from the 2nd century AD.",
            ],
            ["- sample diff \n + sample diff", "- sample diff \n + sample diff"],
        ),
    ],
)
def test_without_history(msgs, diffs, tokenizers):
    encoder_tok, decoder_tok = tokenizers
    inputs = []
    for diff, msg in zip(diffs, msgs):
        diff_ids = encoder_tok(diff, padding=False, truncation=True).input_ids
        msgs_ids = decoder_tok(msg, padding=False, truncation=False).input_ids
        inputs.append({"diff_input_ids": diff_ids, "msg_input_ids": msgs_ids, "history_input_ids": []})

    data_collator = GenerationCollator(
        context_ratio=0.1, src_tokenizer=encoder_tok, trg_tokenizer=decoder_tok, max_len=200, with_history=False
    )
    res = data_collator(inputs)

    assert res["diff_input_ids"].shape[1] <= 500
    assert res["msg_input_ids"].shape[1] <= 200

    for context, prefix, target, msg in zip(res["msg_input_ids"], res["prefix"], res["target"], msgs):
        decoded_context = decoder_tok.decode(context, skip_special_tokens=True)
        decoded_target = decoder_tok.decode(target, skip_special_tokens=True)
        assert decoded_context + decoded_target == msg
