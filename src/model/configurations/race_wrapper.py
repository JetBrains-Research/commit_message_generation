from typing import Any

from src.model.configurations.base_model import BaseModel
from src.utils import Batch, BatchTest

from .utils.race import RACE


class RACEWrapper(BaseModel):
    """This class serves as a wrapper of RACE model for commit message completion task.

    Args:
        name_or_path: Name on HuggingFace hub or path to pretrained checkpoint.
        tokenizer: Tokenizer for the checkpoint (it's initialized earlier to add special tokens when necessary).
    """

    def __init__(self, tokenizer, name_or_path, **kwargs):
        super().__init__()
        self._tokenizer = tokenizer
        self.model = RACE.from_pretrained(name_or_path)
        self.model.resize_token_embeddings(len(self._tokenizer))

    def forward(self, batch: Batch) -> Any:
        return self.model(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            retrieved_diff_input_ids=batch.retrieved_diff_input_ids,
            retrieved_diff_attention_mask=batch.retrieved_diff_attention_mask,
            retrieved_msg_input_ids=batch.retrieved_msg_input_ids,
            retrieved_msg_attention_mask=batch.retrieved_msg_attention_mask,
            labels=batch.labels,
        )

    def generate(self, batch: BatchTest, **generation_kwargs) -> Any:
        return self.model.generate(
            input_ids=batch.encoder_input_ids,
            attention_mask=batch.encoder_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            retrieved_diff_input_ids=batch.retrieved_diff_input_ids,
            retrieved_diff_attention_mask=batch.retrieved_diff_attention_mask,
            retrieved_msg_input_ids=batch.retrieved_msg_input_ids,
            retrieved_msg_attention_mask=batch.retrieved_msg_attention_mask,
            **generation_kwargs,
        )

    def num_parameters(self, exclude_embeddings: bool):
        return self.model.num_parameters(exclude_embeddings=exclude_embeddings)
