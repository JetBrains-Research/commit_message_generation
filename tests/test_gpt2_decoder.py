import pytest
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig  # type: ignore
from src.model import GPT2Decoder

torch.manual_seed(42)


@pytest.fixture
def default_encoder():
    encoder = AutoModel.from_pretrained("distilbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    return encoder, tokenizer


@pytest.fixture
def default_decoder():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    decoder_config = AutoConfig.from_pretrained("distilgpt2")
    decoder_config.pad_token_id = tokenizer.eos_token_id
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = GPT2Decoder.from_pretrained("distilgpt2", config=decoder_config)
    return decoder, tokenizer


@pytest.mark.parametrize(
    "encoder_input,decoder_input",
    [
        (
            "(CNN) -- An American woman died aboard a cruise ship that docked at Rio de "
            "Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, "
            "according to the state-run Brazilian news agency, Agencia Brasil.",
            "The elderly woman",
        )
    ],
)
def test_decoder_with_and_without_encoder(encoder_input, decoder_input, default_encoder, default_decoder):
    encoder, encoder_tokenizer = default_encoder
    encoder_input = encoder_tokenizer(encoder_input, padding=True, truncation=True, return_tensors="pt")
    encoder_outputs, encoder_attention_mask = encoder(**encoder_input), encoder_input.attention_mask

    decoder, decoder_tokenizer = default_decoder
    decoder_input = decoder_tokenizer(decoder_input, padding=True, truncation=True, return_tensors="pt")
    result_w_encoder = decoder.generate(
        input_ids=decoder_input.input_ids,
        attention_mask=decoder_input.attention_mask,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=encoder_attention_mask,
        max_length=50,
        min_length=1,
        num_beams=5,
        num_beam_hyps_to_keep=5,
        output_scores=True,
        return_dict_in_generate=True,
    )

    result_no_encoder = decoder.generate(
        input_ids=decoder_input.input_ids,
        attention_mask=decoder_input.attention_mask,
        max_length=50,
        min_length=1,
        num_beams=5,
        num_beam_hyps_to_keep=5,
        output_scores=True,
    )
    print("With encoder:")
    print(decoder_tokenizer.batch_decode(result_w_encoder.sequences, skip_special_tokens=True))
    print()
    print("Without encoder:")
    print(decoder_tokenizer.batch_decode(result_no_encoder.sequences, skip_special_tokens=True))
    assert not torch.equal(result_no_encoder.sequences_scores, result_w_encoder.sequences_scores)
