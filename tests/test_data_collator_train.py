import pytest
import torch
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
from transformers.models.encoder_decoder.modeling_encoder_decoder import (
    shift_tokens_right,
)

from src.data_utils.data_collators import DataCollatorTrain
from src.utils import SingleExample


@pytest.fixture(scope="session")
def default_tokenizers():
    encoder_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    decoder_tok = AutoTokenizer.from_pretrained("distilgpt2")
    decoder_tok.pad_token = decoder_tok.eos_token
    decoder_tok.sep_token = decoder_tok.eos_token

    return encoder_tok, decoder_tok


def test_shift_encoder_decoder(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers

    data_collator = DataCollatorTrain(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        decoder_context_max_len=None,
        with_history=None,
        shift_labels=None,
        encoder_input_type=None,
        encoder_context_max_len=None,
        testing=None,
        process_retrieved=False,
    )
    labels = [[decoder_tok.bos_token_id]] + [decoder_tok("some example").input_ids] + [[decoder_tok.eos_token_id]]
    ids_, labels_ = data_collator._shift_for_encoder_decoder(ids=labels, labels=labels)
    transformer_ids = shift_tokens_right(
        torch.tensor([[ex for sublist in labels for ex in sublist]], dtype=torch.int64),
        decoder_start_token_id=decoder_tok.bos_token_id,
        pad_token_id=decoder_tok.pad_token_id,
    )

    assert [[ex for sublist in ids_ for ex in sublist]] == transformer_ids.tolist()
    assert labels_ == labels


def test_shift_t5(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers

    t5_config = T5Config.from_pretrained("t5-small")

    data_collator = DataCollatorTrain(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        decoder_context_max_len=None,
        with_history=None,
        shift_labels=None,
        encoder_input_type=None,
        encoder_context_max_len=None,
        testing=None,
        process_retrieved=False,
        decoder_start_token_id=t5_config.decoder_start_token_id,
    )
    labels = [[decoder_tok.bos_token_id]] + [decoder_tok("some example").input_ids] + [[decoder_tok.eos_token_id]]
    ids_, labels_ = data_collator._shift_for_encoder_decoder(ids=labels, labels=labels)

    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    transformer_ids = t5_model._shift_right(
        torch.tensor([[ex for sublist in labels for ex in sublist]], dtype=torch.int64),
    )

    assert [[ex for sublist in ids_ for ex in sublist]] == transformer_ids.tolist()
    assert labels_ == labels


def test_decoder_input_without_history_no_shift(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers

    inputs = [
        SingleExample(diff_input_ids=[], msg_input_ids=[i for i in range(128)], history_input_ids=[]),
        SingleExample(diff_input_ids=[], msg_input_ids=[i for i in range(3)], history_input_ids=[]),
    ]
    for encoder_input_type in ["diff", "history"]:
        no_shift_data_collator = DataCollatorTrain(
            diff_bos_token_id=encoder_tok.bos_token_id,
            diff_eos_token_id=encoder_tok.eos_token_id,
            diff_pad_token_id=encoder_tok.pad_token_id,
            msg_bos_token_id=decoder_tok.bos_token_id,
            msg_eos_token_id=decoder_tok.eos_token_id,
            msg_pad_token_id=decoder_tok.pad_token_id,
            msg_sep_token_id=decoder_tok.sep_token_id,
            decoder_context_max_len=256,
            with_history=False
            if encoder_input_type == "diff"
            else True,  # with_history will be ignored when encoder input is history
            shift_labels=False,
            encoder_input_type=encoder_input_type,
            encoder_context_max_len=None,
            testing=None,
            process_retrieved=False,
        )
        decoder_input_ids, decoder_attention_mask, labels = no_shift_data_collator._process_decoder_input(inputs)

        assert decoder_input_ids.shape == (2, 130)
        assert torch.all(
            decoder_input_ids[0]
            == torch.tensor([decoder_tok.bos_token_id] + [i for i in range(128)] + [decoder_tok.eos_token_id])
        )
        assert torch.all(
            decoder_input_ids[1]
            == torch.tensor(
                [decoder_tok.bos_token_id]
                + [i for i in range(3)]
                + [decoder_tok.eos_token_id]
                + [decoder_tok.pad_token_id for _ in range(128 - 3)]
            )
        )
        assert decoder_attention_mask.shape == (2, 130)
        assert torch.all(decoder_attention_mask[0] == torch.tensor([1 for _ in range(128 + 2)]))
        assert torch.all(
            decoder_attention_mask[1] == torch.tensor([1 for _ in range(3 + 2)] + [0 for _ in range(128 - 3)])
        )

        assert labels.shape == (2, 130)
        assert torch.all(
            labels[0] == torch.tensor([decoder_tok.bos_token_id] + [i for i in range(128)] + [decoder_tok.eos_token_id])
        )
        assert torch.all(
            labels[1]
            == torch.tensor(
                [decoder_tok.bos_token_id]
                + [i for i in range(3)]
                + [decoder_tok.eos_token_id]
                + [-100 for _ in range(128 - 3)]
            )
        )


def test_decoder_input_without_history_shift(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers

    inputs = [
        SingleExample(diff_input_ids=[], msg_input_ids=[i for i in range(128)], history_input_ids=[]),
        SingleExample(diff_input_ids=[], msg_input_ids=[i for i in range(3)], history_input_ids=[]),
    ]

    for encoder_input_type in ["diff", "history"]:
        no_shift_data_collator = DataCollatorTrain(
            diff_bos_token_id=encoder_tok.bos_token_id,
            diff_eos_token_id=encoder_tok.eos_token_id,
            diff_pad_token_id=encoder_tok.pad_token_id,
            msg_bos_token_id=decoder_tok.bos_token_id,
            msg_eos_token_id=decoder_tok.eos_token_id,
            msg_pad_token_id=decoder_tok.pad_token_id,
            msg_sep_token_id=decoder_tok.sep_token_id,
            decoder_context_max_len=256,
            encoder_input_type=encoder_input_type,
            with_history=False
            if encoder_input_type == "diff"
            else True,  # with_history will be ignored when encoder input is history
            shift_labels=False,
            encoder_context_max_len=None,
            testing=None,
            process_retrieved=False,
        )
        decoder_input_ids, decoder_attention_mask, labels = no_shift_data_collator._process_decoder_input(inputs)

        shift_data_collator = DataCollatorTrain(
            diff_bos_token_id=encoder_tok.bos_token_id,
            diff_eos_token_id=encoder_tok.eos_token_id,
            diff_pad_token_id=encoder_tok.pad_token_id,
            msg_bos_token_id=decoder_tok.bos_token_id,
            msg_eos_token_id=decoder_tok.eos_token_id,
            msg_pad_token_id=decoder_tok.pad_token_id,
            msg_sep_token_id=decoder_tok.sep_token_id,
            decoder_context_max_len=256,
            encoder_input_type=encoder_input_type,
            with_history=False
            if encoder_input_type == "diff"
            else True,  # with_history will be ignored when encoder input is history
            shift_labels=True,
            encoder_context_max_len=None,
            testing=None,
            process_retrieved=False,
        )
        (
            shift_decoder_input_ids,
            shift_decoder_attention_mask,
            shift_labels,
        ) = shift_data_collator._process_decoder_input(inputs)

        assert torch.all(decoder_input_ids[:, :-1] == shift_decoder_input_ids[:, 1:])
        assert torch.all(decoder_attention_mask == shift_decoder_attention_mask)

        assert shift_labels.shape == (2, 130)
        assert torch.all(
            shift_labels[0]
            == torch.tensor([decoder_tok.bos_token_id] + [i for i in range(128)] + [decoder_tok.eos_token_id])
        )
        assert torch.all(
            shift_labels[1]
            == torch.tensor(
                [decoder_tok.bos_token_id]
                + [i for i in range(3)]
                + [decoder_tok.eos_token_id]
                + [-100 for _ in range(128 - 3)]
            )
        )


def test_decoder_input_with_history_no_shift(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers

    inputs = [
        SingleExample(
            diff_input_ids=[],
            msg_input_ids=[i for i in range(5, 255)],
            history_input_ids=[[i] for i in range(256, 512)],
        ),
        SingleExample(diff_input_ids=[], msg_input_ids=[i for i in range(5, 255)], history_input_ids=[]),
    ]

    data_collator = DataCollatorTrain(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        decoder_context_max_len=256,
        with_history=True,
        encoder_input_type="diff",
        shift_labels=False,
        encoder_context_max_len=None,
        testing=None,
        process_retrieved=False,
    )
    decoder_input_ids, decoder_attention_mask, labels = data_collator._process_decoder_input(inputs)

    assert decoder_input_ids.shape == (2, 256)
    assert torch.all(
        decoder_input_ids[0]
        == torch.tensor(
            [decoder_tok.bos_token_id]
            + [510, decoder_tok.sep_token_id, 511, decoder_tok.sep_token_id]
            + [i for i in range(5, 255)]
            + [decoder_tok.eos_token_id]
        )
    )
    assert torch.all(
        decoder_input_ids[1]
        == torch.tensor(
            [decoder_tok.bos_token_id]
            + [i for i in range(5, 255)]
            + [decoder_tok.eos_token_id]
            + [decoder_tok.pad_token_id for _ in range(4)]
        )
    )
    assert decoder_attention_mask.shape == (2, 256)
    assert torch.all(decoder_attention_mask[0] == torch.tensor([1 for _ in range(256)]))
    assert torch.all(decoder_attention_mask[1] == torch.tensor([1 for _ in range(256 - 4)] + [0 for _ in range(4)]))

    assert labels.shape == (2, 256)
    assert torch.all(
        labels[0]
        == torch.tensor(
            [decoder_tok.bos_token_id]
            + [-100 for _ in range(4)]
            + [i for i in range(5, 255)]
            + [decoder_tok.eos_token_id]
        )
    )
    assert torch.all(
        labels[1]
        == torch.tensor(
            [decoder_tok.bos_token_id]
            + [i for i in range(5, 255)]
            + [decoder_tok.eos_token_id]
            + [-100 for _ in range(4)]
        )
    )


def test_decoder_input_with_history_shift(default_tokenizers):
    encoder_tok, decoder_tok = default_tokenizers

    inputs = [
        SingleExample(
            diff_input_ids=[],
            msg_input_ids=[i for i in range(5, 255)],
            history_input_ids=[[i] for i in range(256, 512)],
        ),
        SingleExample(diff_input_ids=[], msg_input_ids=[i for i in range(5, 255)], history_input_ids=[]),
    ]

    no_shift_data_collator = DataCollatorTrain(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        decoder_context_max_len=256,
        encoder_input_type="diff",
        with_history=True,
        shift_labels=False,
        encoder_context_max_len=None,
        testing=None,
        process_retrieved=False,
    )
    decoder_input_ids, decoder_attention_mask, labels = no_shift_data_collator._process_decoder_input(inputs)

    shift_data_collator = DataCollatorTrain(
        diff_bos_token_id=encoder_tok.bos_token_id,
        diff_eos_token_id=encoder_tok.eos_token_id,
        diff_pad_token_id=encoder_tok.pad_token_id,
        msg_bos_token_id=decoder_tok.bos_token_id,
        msg_eos_token_id=decoder_tok.eos_token_id,
        msg_pad_token_id=decoder_tok.pad_token_id,
        msg_sep_token_id=decoder_tok.sep_token_id,
        decoder_context_max_len=256,
        encoder_input_type="diff",
        with_history=True,
        shift_labels=True,
        encoder_context_max_len=None,
        testing=None,
        process_retrieved=False,
    )
    shift_decoder_input_ids, shift_decoder_attention_mask, shift_labels = shift_data_collator._process_decoder_input(
        inputs
    )

    assert torch.all(decoder_input_ids[:, :-1] == shift_decoder_input_ids[:, 1:])
    assert torch.all(decoder_attention_mask == shift_decoder_attention_mask)

    assert shift_labels.shape == (2, 256)
    assert torch.all(
        shift_labels[0]
        == torch.tensor(
            [decoder_tok.bos_token_id]
            + [-100 for _ in range(4)]
            + [i for i in range(5, 255)]
            + [decoder_tok.eos_token_id]
        )
    )
    assert torch.all(
        shift_labels[1]
        == torch.tensor(
            [decoder_tok.bos_token_id]
            + [i for i in range(5, 255)]
            + [decoder_tok.eos_token_id]
            + [-100 for _ in range(4)]
        )
    )
