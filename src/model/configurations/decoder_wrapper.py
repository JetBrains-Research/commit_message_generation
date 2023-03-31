from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from src.model.configurations.base_model import BaseModel
from src.utils import Batch, BatchTest


class DecoderWrapper(BaseModel):
    """This class serves as a GPT-2 wrapper for commit message completion task.

    Args:
        tokenizer: tokenizer for target sequences (messages)
        decoder_name_or_path: name or path for pretrained GPT-2 checkpoint
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        decoder_name_or_path: str,
        **kwargs,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(decoder_name_or_path)
        self.model.resize_token_embeddings(len(self._tokenizer))  # type: ignore[arg-type]

    def forward(self, batch: Batch):
        return self.model(
            input_ids=batch.decoder_input_ids, attention_mask=batch.decoder_attention_mask, labels=batch.labels
        )

    def generate(self, batch: BatchTest, **generation_kwargs):
        return self.model.generate(
            input_ids=batch.decoder_input_ids,
            attention_mask=batch.decoder_attention_mask,
            **generation_kwargs,
        )

    def num_parameters(self, exclude_embeddings: bool):
        return self.model.num_parameters(exclude_embeddings=exclude_embeddings)
