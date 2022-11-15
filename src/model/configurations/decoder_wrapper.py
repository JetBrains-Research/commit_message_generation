from transformers import AutoModelForCausalLM, GPT2Tokenizer, PreTrainedTokenizerFast

from src.model.configurations.base_model import BaseModel
from src.utils import Batch, BatchTest, PrefixAllowedTokens


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

    def forward(self, batch: Batch):
        return self.model(
            input_ids=batch.decoder_input_ids, attention_mask=batch.decoder_attention_mask, labels=batch.labels
        )

    def generate(self, batch: BatchTest, **generation_kwargs):
        prefix_fn = PrefixAllowedTokens(
            prefix={i: prefix for i, prefix in enumerate(batch.prefixes)},
            context_len={i: len(msg) for i, msg in enumerate(batch.decoder_input_ids)},
            tokenizer=self._tokenizer,
        )

        return self.model.generate(
            input_ids=batch.decoder_input_ids,
            attention_mask=batch.decoder_attention_mask,
            prefix_allowed_tokens_fn=prefix_fn,
            eos_token_id=self._tokenizer.eos_token_id,  # type: ignore[attr-defined]
            **generation_kwargs,
        )

    def num_parameters(self, exclude_embeddings: bool):
        return self.model.num_parameters(exclude_embeddings=exclude_embeddings)
