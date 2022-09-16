from transformers import AutoModelForCausalLM, GPT2Tokenizer, PreTrainedTokenizerFast

from src.model.configurations.base_model import BaseModel
from src.utils import Batch, BatchTest, PrefixAllowedTokens


class DecoderWrapper(BaseModel):
    """This class is used for training and evaluation of GPT-2-based model for
    commit message completion task.

    Args:
        decoder_name_or_path: name or path for pretrained GPT-2 checkpoint
        tokenizer: tokenizer for target sequences (messages)
        preds_artifact_name: an artifact name for saving model predictions as W&B artifact
        preds_artifact_type: an artifact type for saving model predictions as W&B artifact
        preds_table_name: a table name for saving model predictions as W&B artifact
        learning_rate: maximum learning rate
        num_epochs: total number of epochs (used to calculate total number of steps for LR scheduler)
        num_batches: total number of batches in one epoch (used to calculate total number of steps for LR scheduler)
        num_gpus: total number of GPUs (used to calculate total number of steps for LR scheduler)
        generation_kwargs: kwargs for transformers.generation_utils.GenerationMixin.generate
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
