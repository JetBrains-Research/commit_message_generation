import pytest
import torch
from transformers import AutoTokenizer

from src.data_utils.data_collators.base_collator_utils import BaseCollatorUtils
from src.utils import SingleExample


@pytest.fixture(scope="module")
def default_tokenizers():
    encoder_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    return encoder_tok, encoder_tok


@pytest.fixture(scope="module")
def collator_diff(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers
    collator = BaseCollatorUtils(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        encoder_context_max_len=512,
        decoder_context_max_len=256,
        with_history=True,
        process_retrieved=False,
        encoder_input_type="diff",
        testing=False,
    )
    return collator


def test_diff_single_example(default_tokenizers, collator_diff):
    encoder_tok, decoder_tok = default_tokenizers

    diff_inputs = [SingleExample(diff_input_ids=[i for i in range(5, 105)], msg_input_ids=[], history_input_ids=[])]
    (encoder_input_ids, encoder_attention_mask), _, _ = collator_diff._process_encoder_input(diff_inputs)

    assert encoder_input_ids.shape == (1, 102)
    assert torch.all(
        encoder_input_ids
        == torch.tensor([encoder_tok.bos_token_id] + [i for i in range(5, 105)] + [encoder_tok.eos_token_id])
    )
    assert encoder_attention_mask.shape == (1, 102)
    assert torch.all(encoder_attention_mask == torch.tensor([1 for _ in range(100 + 2)]))


def test_diff_batch_pad_max_len(default_tokenizers, collator_diff):
    encoder_tok, decoder_tok = default_tokenizers

    diff_inputs = [
        SingleExample(diff_input_ids=[i for i in range(5, 105)], msg_input_ids=[], history_input_ids=[]),
        SingleExample(diff_input_ids=[i for i in range(5, 50)], msg_input_ids=[], history_input_ids=[]),
    ]
    (encoder_input_ids, encoder_attention_mask), _, _ = collator_diff._process_encoder_input(diff_inputs)

    assert encoder_input_ids.shape == (2, 102)
    assert torch.all(
        encoder_input_ids[0]
        == torch.tensor([encoder_tok.bos_token_id] + [i for i in range(5, 105)] + [encoder_tok.eos_token_id])
    )
    assert torch.all(
        encoder_input_ids[1]
        == torch.tensor(
            [encoder_tok.bos_token_id]
            + [i for i in range(5, 50)]
            + [encoder_tok.eos_token_id]
            + [encoder_tok.pad_token_id for _ in range(100 - 45)]
        )
    )
    assert encoder_attention_mask.shape == (2, 102)
    assert torch.all(encoder_attention_mask[0] == torch.tensor([1 for _ in range(100 + 2)]))
    assert torch.all(
        encoder_attention_mask[1] == torch.tensor([1 for _ in range(45 + 2)] + [0 for _ in range(100 - 45)])
    )


def test_diff_batch_truncate_max_len(default_tokenizers, collator_diff):
    encoder_tok, decoder_tok = default_tokenizers

    diff_inputs = [
        SingleExample(diff_input_ids=[i for i in range(5, 1024)], msg_input_ids=[], history_input_ids=[]),
        SingleExample(diff_input_ids=[i for i in range(5, 50)], msg_input_ids=[], history_input_ids=[]),
    ]
    (encoder_input_ids, encoder_attention_mask), _, _ = collator_diff._process_encoder_input(diff_inputs)

    assert encoder_input_ids.shape == (2, 512)
    assert torch.all(
        encoder_input_ids[0]
        == torch.tensor([encoder_tok.bos_token_id] + [i for i in range(5, 515)] + [encoder_tok.eos_token_id])
    )
    assert torch.all(
        encoder_input_ids[1]
        == torch.tensor(
            [encoder_tok.bos_token_id]
            + [i for i in range(5, 50)]
            + [encoder_tok.eos_token_id]
            + [encoder_tok.pad_token_id for _ in range(510 - 45)]
        )
    )
    assert encoder_attention_mask.shape == (2, 512)
    assert torch.all(encoder_attention_mask[0] == torch.tensor([1 for _ in range(512)]))
    assert torch.all(
        encoder_attention_mask[1] == torch.tensor([1 for _ in range(45 + 2)] + [0 for _ in range(510 - 45)])
    )


def test_history(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers

    collator_history = BaseCollatorUtils(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        encoder_context_max_len=8,
        encoder_input_type="history",
        testing=None,
        decoder_context_max_len=None,
        with_history=None,
        process_retrieved=False,
    )

    history_inputs = [
        SingleExample(diff_input_ids=[], msg_input_ids=[], history_input_ids=[[i] for i in range(5, 1024)]),
        SingleExample(diff_input_ids=[], msg_input_ids=[], history_input_ids=[[i] for i in range(1024, 1026)]),
    ]

    (encoder_input_ids, encoder_attention_mask), _, _ = collator_history._process_encoder_input(history_inputs)

    assert encoder_input_ids.shape == (2, 7)
    assert torch.all(
        encoder_input_ids[0]
        == torch.tensor(
            [decoder_tok.bos_token_id]
            + [1021, decoder_tok.sep_token_id, 1022, decoder_tok.sep_token_id, 1023]
            + [decoder_tok.eos_token_id]
        )
    )

    assert torch.all(
        encoder_input_ids[1]
        == torch.tensor(
            [encoder_tok.bos_token_id]
            + [1024, decoder_tok.sep_token_id, 1025]
            + [encoder_tok.eos_token_id]
            + [encoder_tok.pad_token_id for _ in range(2)]
        )
    )
    assert encoder_attention_mask.shape == (2, 7)
    assert torch.all(encoder_attention_mask[0] == torch.tensor([1 for _ in range(7)]))
    assert torch.all(encoder_attention_mask[1] == torch.tensor([1 for _ in range(5)] + [0 for _ in range(2)]))

    history_inputs = [
        SingleExample(
            diff_input_ids=[],
            msg_input_ids=[],
            history_input_ids=[
                decoder_tok("older message", add_special_tokens=False, padding=False, truncation=False).input_ids,
                decoder_tok("old message", add_special_tokens=False, padding=False, truncation=False).input_ids,
            ],
        ),
        SingleExample(
            diff_input_ids=[],
            msg_input_ids=[],
            history_input_ids=[
                decoder_tok("another old message", add_special_tokens=False, padding=False, truncation=False).input_ids
            ],
        ),
    ]

    collator_history = BaseCollatorUtils(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        encoder_context_max_len=256,
        encoder_input_type="history",
        testing=None,
        decoder_context_max_len=None,
        with_history=None,
        process_retrieved=False,
    )
    (encoder_input_ids, encoder_attention_mask), _, _ = collator_history._process_encoder_input(history_inputs)

    assert encoder_input_ids.shape == (
        2,
        len(history_inputs[0].history_input_ids[0]) + len(history_inputs[0].history_input_ids[1]) + 3,
    )
    assert (
        decoder_tok.decode(encoder_input_ids[0], skip_special_tokens=False)
        == f"{decoder_tok.bos_token}older message{decoder_tok.sep_token}old message{decoder_tok.eos_token}"
    )
