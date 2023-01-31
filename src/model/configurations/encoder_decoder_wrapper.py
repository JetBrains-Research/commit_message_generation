import logging
from typing import Optional

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from src.model.configurations import BaseModel
from src.utils import Batch, BatchTest, PrefixAllowedTokens, remove_layers_from_model


class EncoderDecoderWrapper(BaseModel):
    """This class serves as a wrapper of Transformer-based models for commit message completion task.

    More specifically, this class relies on EncoderDecoderModel from HuggingFace Transformers.

    It is possible to either use pretrained models for initializing encoder/decoder
    or initialize from scratch.

    Args:
        diff_tokenizer: Tokenizer for source sequences (diffs)
        msg_tokenizer: Tokenizer for target sequences (messages)
        encoder_context_max_len: Maximum allowed input sequence length for encoder, used for initializing from scratch.
        decoder_context_max_len: Maximum allowed input sequence length for decoder, used for initializing from scratch.
        encoder_name_or_path: Optional – name or path to pretrained checkpoint to initialize encoder.
        decoder_name_or_path: Optional – name or path to pretrained checkpoint to initialize decoder.
        num_layers_encoder: If `encoder_name_or_path` is None, encoder will be initialized
            from scratch with given number of layers; else, if given number of layers is less than number of layers in
            pretrained checkpoint, they will be uniformly picked.
        num_layers_decoder: If `decoder_name_or_path` is None, decoder will be initialized
            from scratch with given number of layers; else, if given number of layers is less than number of layers in
            pretrained checkpoint, they will be uniformly picked.
        encoder_model_type: Optional – if encoder is initialized from scratch, this specific model class will be used.
        decoder_model_type: Optional – if decoder is initialized from scratch, this specific model class will be used.
        tie_encoder_decoder: If set to `True`, encoder and decoder will share the same parameters
          (should be used with the same model classes and tokenizers).
        tie_word_embeddings: If set to `True`, encoder and decoder will share the same parameters for embedding layers
          (should be used with the same model classes and tokenizers).
    """

    def __init__(
        self,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        encoder_context_max_len: Optional[int] = None,
        decoder_context_max_len: Optional[int] = None,
        encoder_name_or_path: Optional[str] = None,
        decoder_name_or_path: Optional[str] = None,
        num_layers_encoder: Optional[int] = None,
        num_layers_decoder: Optional[int] = None,
        encoder_model_type: Optional[str] = None,
        decoder_model_type: Optional[str] = None,
        tie_encoder_decoder: Optional[bool] = None,
        tie_word_embeddings: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self._diff_tokenizer = diff_tokenizer
        self._msg_tokenizer = msg_tokenizer
        self.model = self._init_model(
            encoder_model_type=encoder_model_type,
            encoder_name_or_path=encoder_name_or_path,
            num_layers_encoder=num_layers_encoder,
            decoder_model_type=decoder_model_type,
            decoder_name_or_path=decoder_name_or_path,
            num_layers_decoder=num_layers_decoder,
            tie_encoder_decoder=tie_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            encoder_context_max_len=encoder_context_max_len,
            decoder_context_max_len=decoder_context_max_len,
        )

    def _init_model(
        self,
        encoder_context_max_len: Optional[int],
        encoder_model_type: Optional[str],
        encoder_name_or_path: Optional[str],
        num_layers_encoder: Optional[int],
        decoder_context_max_len: Optional[int],
        decoder_model_type: Optional[str],
        decoder_name_or_path: Optional[str],
        num_layers_decoder: Optional[int],
        tie_encoder_decoder: Optional[bool],
        tie_word_embeddings: Optional[bool],
    ):
        encoder = self._init_model_part(
            encoder_or_decoder="encoder",
            model_type=encoder_model_type,
            name_or_path=encoder_name_or_path,
            num_layers=num_layers_encoder,
            context_max_length=encoder_context_max_len,
        )
        decoder = self._init_model_part(
            encoder_or_decoder="decoder",
            model_type=decoder_model_type,
            name_or_path=decoder_name_or_path,
            num_layers=num_layers_decoder,
            context_max_length=decoder_context_max_len,
        )

        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder.config, decoder_config=decoder.config  # type: ignore[attr-defined]
        )
        if tie_encoder_decoder is not None:
            config.encoder.tie_encoder_decoder = tie_encoder_decoder
            config.decoder.tie_encoder_decoder = tie_encoder_decoder
            config.tie_encoder_decoder = tie_encoder_decoder
        if tie_word_embeddings is not None:
            config.tie_word_embeddings = tie_word_embeddings

        model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)
        return model

    def _init_model_part(
        self,
        encoder_or_decoder: str,
        context_max_length: Optional[int],
        model_type: Optional[str],
        num_layers: Optional[int],
        name_or_path: Optional[str],
    ) -> PreTrainedModel:
        """
        Initializes either encoder or decoder for further use in EncoderDecoderModel class.

        Args:
            encoder_or_decoder: Pass `encoder` to correctly initialize any model as seq2seq encoder.
              Pass `decoder` to correctly initialize any model as seq2seq decoder.
            model_type: Necessary for training from scratch. Corresponding model class will be used.
              Currently supported: `bert`, `roberta`, `gpt2`.
            num_layers: Optional – number of layers. If pretrained model is used and `num_layers` is less than
              actual number of layers in the model, `num_layers` layers will be picked uniformly. When empty,
              default number of layers will be used.
            name_or_path: Optional – name on HuggingFace Hub or path to pretrained model weights.

        Returns:
            initialized model for further use in EncoderDecoderModel class
        """
        # use pretrained model
        if name_or_path:
            if encoder_or_decoder == "encoder":
                model = AutoModel.from_pretrained(name_or_path)
                model.resize_token_embeddings(len(self._diff_tokenizer))  # type: ignore[arg-type]
            else:
                model = AutoModelForCausalLM.from_pretrained(name_or_path, is_decoder=True, add_cross_attention=True)
                model.resize_token_embeddings(len(self._msg_tokenizer))  # type: ignore[arg-type]
            # remove layers if necessary
            if num_layers is not None:
                if model.config.model_type in ["bert", "roberta", "gpt2"]:
                    if (
                        model.config.model_type in ["bert", "roberta"]
                        and num_layers < model.config.num_hidden_layers
                        or model.config.model_type == "gpt2"
                        and num_layers < model.config.n_layer
                    ):
                        model = remove_layers_from_model(model, num_layers)
                else:
                    logging.warning("Unknown model type, default number of layers is used")
            return model

        # use randomly initialized model
        if num_layers:
            if encoder_or_decoder == "encoder":
                vocab_size = self._diff_tokenizer.vocab_size  # type: ignore[attr-defined]
                bos_token_id = self._diff_tokenizer.bos_token_id  # type: ignore[attr-defined]
                eos_token_id = self._diff_tokenizer.eos_token_id  # type: ignore[attr-defined]
                pad_token_id = self._diff_tokenizer.pad_token_id  # type: ignore[attr-defined]
            else:
                vocab_size = self._msg_tokenizer.vocab_size  # type: ignore[attr-defined]
                bos_token_id = self._msg_tokenizer.bos_token_id  # type: ignore[attr-defined]
                eos_token_id = self._msg_tokenizer.eos_token_id  # type: ignore[attr-defined]
                pad_token_id = self._msg_tokenizer.pad_token_id  # type: ignore[attr-defined]

            assert bos_token_id is not None
            assert eos_token_id is not None
            assert pad_token_id is not None

            if model_type is None:
                raise ValueError(f'Provide one of "bert", "roberta", "gpt2" as `{encoder_or_decoder}_model_type`')
            if context_max_length is None:
                raise ValueError(f"Provide `{encoder_or_decoder}_context_max_len`")

            config = AutoConfig.for_model(
                model_type=model_type,
                vocab_size=vocab_size,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

            # set specified number of hidden layers and max allowed context length
            if config.model_type == "gpt2":
                config.n_layer = num_layers
                config.n_positions = context_max_length
            elif config.model_type in ["bert", "roberta"]:
                config.num_hidden_layers = num_layers
                config.type_vocab_size = 1
                config.max_position_embeddings = context_max_length
                if config.model_type == "roberta":  # related: https://github.com/facebookresearch/fairseq/issues/1187
                    config.max_position_embeddings += pad_token_id + 1
            else:
                logging.warning("Unknown model type, default number of layers is used")

            if encoder_or_decoder == "encoder":
                model = AutoModel.from_config(config=config)
            else:
                config.is_decoder = True
                config.add_cross_attention = True
                model = AutoModelForCausalLM.from_config(config=config)
            return model

        raise ValueError(
            f"Unable to initialize {encoder_or_decoder}. You have to specify either `num_layers` and `model_type` to train from scratch or `name_or_path` to load pretrained model"
        )

    def forward(self, batch: Batch):
        return self.model(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            labels=batch.labels,
        )

    def generate(self, batch: BatchTest, **generation_kwargs):
        prefix_fn = PrefixAllowedTokens(
            prefix={i: prefix for i, prefix in enumerate(batch.prefixes)},
            context_len={i: len(msg) for i, msg in enumerate(batch.decoder_input_ids)},
            tokenizer=self._msg_tokenizer,
        )

        return self.model.generate(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=self._msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
            bos_token_id=self._msg_tokenizer.bos_token_id,  # type: ignore[attr-defined]
            eos_token_id=self._msg_tokenizer.eos_token_id,  # type: ignore[attr-defined]
            **generation_kwargs,
        )

    def num_parameters(self, exclude_embeddings: bool):
        return self.model.num_parameters(exclude_embeddings=exclude_embeddings)
