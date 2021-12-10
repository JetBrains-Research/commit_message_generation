import pytorch_lightning as pl

from transformers import AutoTokenizer, PreTrainedTokenizerFast

import os
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from .cmg_dataset_w_history import CMGDatasetWithHistory
from .data_collators import DataCollatorWithHistory, DataCollatorWithHistoryGeneration, DataCollatorWithoutHistory


class CMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        training_data_root: str,
        marker_tests_root: str,
        history_max_len: int,
        encoder_name_or_path: str,
        decoder_name_or_path: str,
        local_rank: int,
        world_size: int,
        with_history: bool,
        sep_tokens: str,
        train_dataloader_conf: DictConfig,
        val_dataloader_conf: DictConfig,
        test_dataloader_conf: DictConfig,
        marker_tests_dataloader_conf: DictConfig,
        testing: bool = False,
    ):
        super().__init__()

        self.training_data_root = os.path.join(hydra.utils.to_absolute_path(dataset_root), training_data_root)
        self.marker_tests_root = os.path.join(hydra.utils.to_absolute_path(dataset_root), marker_tests_root)
        self.history_max_len = history_max_len

        self.local_rank = local_rank
        self.world_size = world_size

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf
        self.marker_tests_dataloader_conf = marker_tests_dataloader_conf

        if encoder_name_or_path.endswith(".json"):
            self._src_tokenizer = PreTrainedTokenizerFast(tokenizer_file=to_absolute_path(encoder_name_or_path))
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

        if "gpt2" in decoder_name_or_path and sep_tokens == "\n":
            sep_tokens = [self._trg_tokenizer.convert_tokens_to_ids("Ċ")]
        else:
            sep_tokens = self._trg_tokenizer(sep_tokens).input_ids

        if with_history:
            self.data_collator = DataCollatorWithHistory(
                src_tokenizer=self._src_tokenizer,
                trg_tokenizer=self._trg_tokenizer,
                max_len=self.history_max_len,
                testing=testing,
                sep_tokens=sep_tokens,
            )
        else:
            self.data_collator = DataCollatorWithoutHistory(
                src_tokenizer=self._src_tokenizer,
                trg_tokenizer=self._trg_tokenizer,
                max_len=self.history_max_len,
                testing=testing,
                sep_tokens=sep_tokens,
            )
        self.data_collator_gen = DataCollatorWithHistoryGeneration(
            src_tokenizer=self._src_tokenizer,
            trg_tokenizer=self._trg_tokenizer,
            max_len=self.history_max_len,
            sep_tokens=sep_tokens,
        )

        # datasets are initialized later
        self.train = None
        self.val = None
        self.marker_tests = None
        self.test = None

    def setup(self, stage=None):
        # called on every GPU
        if stage == "fit" or stage is None:
            self.train = CMGDatasetWithHistory.load_data(
                self.training_data_root + "/train", rank=self.local_rank, world_size=self.world_size
            )

            self.val = CMGDatasetWithHistory.load_data(
                self.training_data_root + "/val", rank=self.local_rank, world_size=self.world_size
            )
            self.marker_tests = CMGDatasetWithHistory.load_data(
                self.marker_tests_root + "/test", rank=self.local_rank, world_size=self.world_size
            )
        if stage == "test" or stage is None:
            self.test = CMGDatasetWithHistory.load_data(
                self.training_data_root + "/test", rank=self.local_rank, world_size=self.world_size
            )

    def train_dataloader(self):
        return self.train.get_dataloader(**self.train_dataloader_conf, collate_fn=self.data_collator)

    def val_dataloader(self):
        return [
            self.val.get_dataloader(**self.val_dataloader_conf, collate_fn=self.data_collator),
            self.marker_tests.get_dataloader(**self.marker_tests_dataloader_conf, collate_fn=self.data_collator_gen),
        ]

    def test_dataloader(self):
        return self.test.get_dataloader(**self.test_dataloader_conf, collate_fn=self.data_collator)
