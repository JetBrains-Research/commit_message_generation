import torch
from typing import Optional, Dict
from .gpt2_decoder import GPT2Decoder
from transformers import AutoModel, AutoConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.file_utils import ModelOutput


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        encoder_name_or_path: Optional[str] = None,
        decoder_name_or_path: Optional[str] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[GPT2Decoder] = None,
    ):
        """
        Class for supporting optional use of encoder before generation.

        :param encoder_name_or_path: model name on huggingface hub or path to directory with pretrained weights
        :param decoder_name_or_path: model name on huggingface hub or path to directory with pretrained weights
        :param encoder: already initialized encoder model
        :param decoder: already initialized decoder model
        """
        super(EncoderDecoder, self).__init__()
        if encoder is None:
            if encoder_name_or_path is None:
                raise ValueError("You have to provide either `encoder` or `encoder_name_or_path`")
            encoder = AutoModel.from_pretrained(encoder_name_or_path)

        if decoder is None:
            if decoder_name_or_path is None:
                raise ValueError("You have to provide either `decoder` or `decoder_name_or_path`")
            decoder_config = AutoConfig.from_pretrained(decoder_name_or_path)
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            decoder = GPT2Decoder.from_pretrained(decoder_name_or_path, config=decoder_config)

        self.encoder = encoder
        self.decoder = decoder

    def generate(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        prefix: Optional[str] = None,
        **generation_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences conditioned on provided inputs via beam search.

        :param input_ids: input ids for decoder
        :param attention_mask: attention mask for decoder
        :param encoder_input_ids: input ids for encoder (optional, but you have to provide `encoder_outputs` otherwise)
        :param encoder_attention_mask: attention mask for encoder (optional)
        :param encoder_outputs: outputs of encoder (optional, but you have to provide `encoder_input_ids` otherwise)
        :param generation_kwargs: all other kwargs are passed to `GPT2Decoder.generate`
        :return: dictionary (with keys `sequences` and `scores`)
        """
        # run encoder on encoder_input_ids (if encoder_outputs are not defined)
        if encoder_outputs is None:
            if encoder_input_ids is not None and len(encoder_input_ids) != 0:
                encoder_outputs = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask)

        # pass encoder outputs to decoder
        return self.decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            prefix=prefix,
            tokenizer=tokenizer,
            **generation_kwargs
        )
