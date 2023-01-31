from dataclasses import dataclass

from .base_configs import (
    BaseDecoderConfig,
    BaseEncoderDecoderConfig,
    BaseRACEConfig,
    BaseSeq2SeqConfig,
)


@dataclass
class DistilGPT2Config(BaseDecoderConfig):
    diff_tokenizer_name_or_path: str = "distilgpt2"
    msg_tokenizer_name_or_path: str = "distilgpt2"
    encoder_context_max_len: int = 512
    decoder_context_max_len: int = 512
    decoder_name_or_path: str = "distilgpt2"


@dataclass
class RandomTransformerConfig(BaseEncoderDecoderConfig):
    diff_tokenizer_name_or_path: str = "raw_data/multilang/byte_level"
    msg_tokenizer_name_or_path: str = "raw_data/multilang/byte_level"
    encoder_context_max_len: int = 512
    decoder_context_max_len: int = 256

    num_layers_encoder: int = 2
    encoder_model_type: str = "roberta"

    num_layers_decoder: int = 2
    decoder_model_type: str = "gpt2"

    tie_encoder_decoder: bool = False
    tie_word_embeddings: bool = False


@dataclass
class CodeT5Config(BaseSeq2SeqConfig):
    diff_tokenizer_name_or_path: str = "Salesforce/codet5-base"
    msg_tokenizer_name_or_path: str = "Salesforce/codet5-base"
    encoder_context_max_len: int = 512
    decoder_context_max_len: int = 512

    name_or_path: str = "Salesforce/codet5-base"


@dataclass
class CodeReviewerConfig(BaseSeq2SeqConfig):
    preprocessor_configuration: str = "codereviewer"
    diff_tokenizer_name_or_path: str = "microsoft/codereviewer"
    msg_tokenizer_name_or_path: str = "microsoft/codereviewer"
    encoder_context_max_len: int = 512
    decoder_context_max_len: int = 512

    name_or_path: str = "microsoft/codereviewer"


@dataclass
class RACEConfig(BaseRACEConfig):
    diff_tokenizer_name_or_path: str = "Salesforce/codet5-base"
    msg_tokenizer_name_or_path: str = "Salesforce/codet5-base"
    encoder_context_max_len: int = 512
    decoder_context_max_len: int = 512

    name_or_path: str = "Salesforce/codet5-base"
