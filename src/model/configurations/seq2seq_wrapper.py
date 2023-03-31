from typing import Any

from transformers import AutoModelForSeq2SeqLM

from src.model.configurations.base_model import BaseModel
from src.utils import Batch, BatchTest


class Seq2SeqWrapper(BaseModel):
    """This class serves as a wrapper of Transformer-based models for commit message completion task.

    More specifically, this class relies on pretrained seq2seq models from HuggingFace Transformers.

    Args:
        name_or_path: Name on HuggingFace hub or path to pretrained checkpoint.
        tokenizer: Tokenizer for the checkpoint (it's initialized earlier to add special tokens when necessary).
    """

    def __init__(self, tokenizer, name_or_path, **kwargs):
        super().__init__()
        self._tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
        self.model.resize_token_embeddings(len(self._tokenizer))

    def forward(self, batch: Batch) -> Any:
        return self.model(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            labels=batch.labels,
        )

    def generate(self, batch: BatchTest, **generation_kwargs) -> Any:
        return self.model.generate(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            **generation_kwargs,
        )

    def num_parameters(self, exclude_embeddings: bool):
        return self.model.num_parameters(exclude_embeddings=exclude_embeddings)
