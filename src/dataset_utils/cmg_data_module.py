import pytorch_lightning as pl
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import hydra
from omegaconf import DictConfig

from hydra.utils import to_absolute_path
from src.dataset_utils.cmg_dataset_w_history import CMGDatasetWithHistory
from src.dataset_utils.data_collators import NextTokenPredictionCollator, GenerationCollator


class CMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        history_max_len: int,
        actual_generation: bool,
        with_history: bool,
        context_ratio: float,
        encoder_name_or_path: str,
        decoder_name_or_path: str,
        local_rank: int,
        world_size: int,
        test_dataloader_conf: DictConfig,
        history_sep: str,
        **kwargs
    ):
        super().__init__()

        self.dataset_root = hydra.utils.to_absolute_path(dataset_root)
        self.history_max_len = history_max_len

        self.local_rank = local_rank
        self.world_size = world_size

        self.test_dataloader_conf = test_dataloader_conf

        if encoder_name_or_path.endswith(".json"):
            self._src_tokenizer = PreTrainedTokenizerFast(tokenizer_file=to_absolute_path(encoder_name_or_path))
            if self._src_tokenizer.pad_token is None:
                self._src_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            self._src_tokenizer = AutoTokenizer.from_pretrained(encoder_name_or_path, use_fast=True)

        if decoder_name_or_path.endswith(".json"):
            self._trg_tokenizer = PreTrainedTokenizerFast(tokenizer_file=to_absolute_path(decoder_name_or_path))
        else:
            self._trg_tokenizer = AutoTokenizer.from_pretrained(decoder_name_or_path, use_fast=True)
            # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
            # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
            if "gpt2" in decoder_name_or_path:
                self._trg_tokenizer.pad_token = self._trg_tokenizer.unk_token

            if history_sep == "\n":
                self._trg_tokenizer._sep = [self._trg_tokenizer.convert_tokens_to_ids("Ċ")]
            else:
                self._trg_tokenizer._sep = self._trg_tokenizer(history_sep).input_ids

        if actual_generation:
            self.data_collator = GenerationCollator(
                src_tokenizer=self._src_tokenizer,
                trg_tokenizer=self._trg_tokenizer,
                context_ratio=context_ratio,
                max_len=self.history_max_len,
                with_history=with_history,
            )
        else:
            self.data_collator = NextTokenPredictionCollator(
                src_tokenizer=self._src_tokenizer,
                trg_tokenizer=self._trg_tokenizer,
                max_len=self.history_max_len,
                with_history=with_history,
            )

        # datasets are initialized later
        self.test = None

    def setup(self, stage=None):
        # called on every GPU
        self.test = CMGDatasetWithHistory.load_data(
            self.dataset_root + "/test", rank=self.local_rank, world_size=self.world_size
        )

    def test_dataloader(self):
        return self.test.get_dataloader(**self.test_dataloader_conf, collate_fn=self.data_collator)
