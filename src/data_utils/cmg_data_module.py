import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .cmg_dataset_w_history import CMGDatasetWithHistory
from .data_collator import DataCollator


class CMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        training_data_root: str,
        marker_tests_root: str,
        decoder_context_max_len: int,
        diff_tokenizer_name_or_path: str,
        msg_tokenizer_name_or_path: str,
        training_with_history: bool,
        generation_with_history: bool,
        sep_tokens: str,
        train_dataloader_conf: DictConfig,
        val_dataloader_conf: DictConfig,
        test_dataloader_conf: DictConfig,
        marker_tests_dataloader_conf: DictConfig,
        local_rank: int,
        world_size: int,
        context_ratio: Optional[float] = None,
        testing: bool = False,
    ):
        super().__init__()

        self.training_data_root = os.path.join(hydra.utils.to_absolute_path(dataset_root), training_data_root)
        self.marker_tests_root = os.path.join(hydra.utils.to_absolute_path(dataset_root), marker_tests_root)

        self.local_rank = local_rank
        self.world_size = world_size

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf
        self.marker_tests_dataloader_conf = marker_tests_dataloader_conf

        if diff_tokenizer_name_or_path.endswith(".json"):
            self._diff_tokenizer = PreTrainedTokenizerFast(tokenizer_file=to_absolute_path(diff_tokenizer_name_or_path))
            if self._diff_tokenizer.pad_token is None:
                self._diff_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            self._diff_tokenizer = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path, use_fast=True)

        if msg_tokenizer_name_or_path.endswith(".json"):
            self._msg_tokenizer = PreTrainedTokenizerFast(tokenizer_file=to_absolute_path(msg_tokenizer_name_or_path))
        else:
            self._msg_tokenizer = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path, use_fast=True)
            # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
            # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
            if "gpt2" in msg_tokenizer_name_or_path:
                self._msg_tokenizer.pad_token = self._msg_tokenizer.unk_token

        if "gpt2" in msg_tokenizer_name_or_path and sep_tokens == "\n":
            sep_tokens_ids = [self._msg_tokenizer.convert_tokens_to_ids("Ċ")]
        else:
            sep_tokens_ids = self._msg_tokenizer(sep_tokens).input_ids

        self.data_collator_train = DataCollator(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            max_len=decoder_context_max_len,
            with_history=training_with_history,
            sep_tokens=sep_tokens_ids,
            generation=False,
            testing=testing,
        )
        self.data_collator_mt = DataCollator(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            max_len=decoder_context_max_len,
            with_history=False,
            sep_tokens=sep_tokens_ids,
            generation=True,
            context_ratio=0.0,
        )
        self.data_collator_test = DataCollator(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            max_len=decoder_context_max_len,
            with_history=generation_with_history,
            sep_tokens=sep_tokens_ids,
            context_ratio=context_ratio,
            generation=True,
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
        return self.train.get_dataloader(**self.train_dataloader_conf, collate_fn=self.data_collator_train)

    def val_dataloader(self):
        return [
            self.val.get_dataloader(**self.val_dataloader_conf, collate_fn=self.data_collator_train),
            self.marker_tests.get_dataloader(**self.marker_tests_dataloader_conf, collate_fn=self.data_collator_mt),
        ]

    def test_dataloader(self):
        return self.test.get_dataloader(**self.test_dataloader_conf, collate_fn=self.data_collator_test)