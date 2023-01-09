from typing import Any

from transformers import AutoModelForSeq2SeqLM

from src.model.configurations.base_model import BaseModel
from src.utils import Batch, BatchTest, PrefixAllowedTokens


class Seq2SeqWrapper(BaseModel):
    """This class serves as a wrapper of Transformer-based models for commit message completion task.

    More specifically, this class relies on pretrained seq2seq models from HuggingFace Transformers.
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
        prefix_fn = PrefixAllowedTokens(
            prefix={i: prefix for i, prefix in enumerate(batch.prefixes)},
            context_len={i: len(msg) for i, msg in enumerate(batch.decoder_input_ids)},
            tokenizer=self._tokenizer,
        )

        return self.model.generate(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=self._tokenizer.pad_token_id,  # type: ignore[attr-defined]
            bos_token_id=self._tokenizer.bos_token_id,  # type: ignore[attr-defined]
            eos_token_id=self._tokenizer.eos_token_id,  # type: ignore[attr-defined]
            **generation_kwargs,
        )

    def num_parameters(self, exclude_embeddings: bool):
        return self.model.num_parameters(exclude_embeddings=exclude_embeddings)
