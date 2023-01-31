from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class BaseModelConfig:
    """
    Basic model configuration.

    Args:
        configuration: What model architecture to use. Should be one of `decoder`, `encoder_decoder`, `seq2seq`, `race`.
        preprocessor_configuration: What diff processing strategy to use. Should be one of `default`, `codereviewer`, `race`.
        diff_tokenizer_name_or_path: Local path or name on HuggingFace Hub for diff tokenizer.
        msg_tokenizer_name_or_path: Local path or name on HuggingFace Hub for message tokenizer.
        encoder_context_max_length: Maximum allowed number of tokens for encoder context.
        decoder_context_max_length: Maximum allowed number of tokens for decoder context.
    """

    configuration: str = MISSING
    preprocessor_configuration: str = "default"
    diff_tokenizer_name_or_path: str = MISSING
    msg_tokenizer_name_or_path: str = MISSING
    encoder_context_max_len: int = MISSING
    decoder_context_max_len: int = MISSING


@dataclass
class BaseDecoderConfig(BaseModelConfig):
    """
    Base configuration for Transformer Decoder.
    """

    configuration: str = "decoder"
    decoder_name_or_path: str = MISSING


@dataclass
class BaseEncoderDecoderConfig(BaseModelConfig):
    """
    Base configuration for Transformer initialized with pretrained encoder/decoder.
    """

    configuration: str = "encoder_decoder"
    num_layers_encoder: Optional[int] = None
    encoder_model_type: Optional[str] = None
    encoder_name_or_path: Optional[str] = None
    num_layers_decoder: Optional[int] = None
    decoder_model_type: Optional[str] = None
    decoder_name_or_path: Optional[str] = None
    tie_encoder_decoder: bool = MISSING
    tie_word_embeddings: bool = MISSING


@dataclass
class BaseSeq2SeqConfig(BaseModelConfig):
    """
    Base configuration for pretrained seq2seq Transformer.
    """

    configuration: str = "seq2seq"
    name_or_path: str = MISSING


@dataclass
class BaseRACEConfig(BaseModelConfig):
    """
    Base configuration for RACE model.
    """

    configuration: str = "race"
    preprocessor_configuration: str = "race"
    name_or_path: str = MISSING
