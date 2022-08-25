import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer

from .cmc_dataset_w_history import CMCDatasetWithHistory
from .data_collator import DataCollator


class CMCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        diffs_folder: str,
        msgs_folder: str,
        encoder_context_max_len: int,
        decoder_context_max_len: int,
        diff_tokenizer_name_or_path: str,
        msg_tokenizer_name_or_path: str,
        local_rank: int,
        world_size: int,
        train_with_history: bool,
        generate_with_history: bool,
        use_mtests: Optional[bool] = False,
        marker_tests_root: Optional[str] = None,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
        test_dataloader_conf: Optional[DictConfig] = None,
        marker_tests_dataloader_conf: Optional[DictConfig] = None,
        decoder_sep_tokens: Optional[str] = None,
        context_ratio: Optional[float] = None,
        testing: bool = False,
    ):
        super().__init__()

        dataset_root = hydra.utils.to_absolute_path(dataset_root)
        self.diffs_path = os.path.join(dataset_root, "diffs", diffs_folder)
        self.msgs_path = os.path.join(dataset_root, "msgs", msgs_folder)
        self.marker_tests_root = None
        if use_mtests and marker_tests_root:
            self.marker_tests_root = hydra.utils.to_absolute_path(marker_tests_root)

        self.local_rank = local_rank
        self.world_size = world_size

        self.train_len: int

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf
        self.marker_tests_dataloader_conf = marker_tests_dataloader_conf

        if not msg_tokenizer_name_or_path:
            raise ValueError("Please make sure to set message tokenizer")
        try:
            self._msg_tokenizer = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path)
        except ValueError:
            self._msg_tokenizer = AutoTokenizer.from_pretrained(
                hydra.utils.to_absolute_path(msg_tokenizer_name_or_path)
            )
        if "gpt2" in msg_tokenizer_name_or_path:
            # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
            # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
            self._msg_tokenizer.pad_token = self._msg_tokenizer.unk_token

        if "gpt2" in msg_tokenizer_name_or_path and decoder_sep_tokens == "\n":
            sep_tokens_ids = [self._msg_tokenizer.convert_tokens_to_ids("Ċ")]
        elif decoder_sep_tokens:
            sep_tokens_ids = self._msg_tokenizer(decoder_sep_tokens, add_special_tokens=False).input_ids
        elif self._msg_tokenizer.sep_token_id:
            sep_tokens_ids = [self._msg_tokenizer.sep_token_id]
        else:
            sep_tokens_ids = [self._msg_tokenizer.eos_token_id]

        if not diff_tokenizer_name_or_path:
            self._diff_tokenizer = self._msg_tokenizer
            logging.warning("Diff tokenizer is not set, using message tokenizer")
        elif diff_tokenizer_name_or_path == msg_tokenizer_name_or_path:
            self._diff_tokenizer = self._msg_tokenizer
        else:
            try:
                self._diff_tokenizer = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path)
            except ValueError:
                self._diff_tokenizer = AutoTokenizer.from_pretrained(
                    hydra.utils.to_absolute_path(diff_tokenizer_name_or_path)
                )

        self.data_collator_train = DataCollator(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            encoder_context_max_len=encoder_context_max_len,
            decoder_context_max_len=decoder_context_max_len,
            with_history=train_with_history,
            decoder_sep_tokens=sep_tokens_ids,
            generation=False,
            testing=testing,
        )
        self.data_collator_mt = DataCollator(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            encoder_context_max_len=encoder_context_max_len,
            decoder_context_max_len=decoder_context_max_len,
            with_history=False,
            decoder_sep_tokens=sep_tokens_ids,
            generation=True,
            context_ratio=0.0,
        )
        self.data_collator_test = DataCollator(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            encoder_context_max_len=encoder_context_max_len,
            decoder_context_max_len=decoder_context_max_len,
            with_history=generate_with_history,
            decoder_sep_tokens=sep_tokens_ids,
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
            self.train = CMCDatasetWithHistory.load_data(
                diffs_path=self.diffs_path,
                msgs_path=self.msgs_path,
                part="train",
                rank=self.local_rank,
                world_size=self.world_size,
            )
            self.train_len = self.train._len
            self.val = CMCDatasetWithHistory.load_data(
                diffs_path=self.diffs_path,
                msgs_path=self.msgs_path,
                part="val",
                rank=self.local_rank,
                world_size=self.world_size,
            )

            self.marker_tests = None
            if self.marker_tests_root:
                self.marker_tests = CMCDatasetWithHistory.load_data(
                    diffs_path=self.diffs_path,
                    msgs_path=self.msgs_path,
                    part="mtests",
                    rank=self.local_rank,
                    world_size=self.world_size,
                )
        if stage == "test" or stage is None:
            self.test = CMCDatasetWithHistory.load_data(
                diffs_path=self.diffs_path,
                msgs_path=self.msgs_path,
                part="test",
                rank=self.local_rank,
                world_size=self.world_size,
            )
        if stage == "sweep":
            # when tuning hyperparameters, run test_conf logic but on validation set
            self.test = CMCDatasetWithHistory.load_data(
                diffs_path=self.diffs_path,
                msgs_path=self.msgs_path,
                part="val",
                rank=self.local_rank,
                world_size=self.world_size,
            )

    def train_dataloader(self):
        return self.train.get_dataloader(**self.train_dataloader_conf, collate_fn=self.data_collator_train)

    def val_dataloader(self):
        if self.marker_tests:
            return [
                self.val.get_dataloader(**self.val_dataloader_conf, collate_fn=self.data_collator_train),
                self.marker_tests.get_dataloader(**self.marker_tests_dataloader_conf, collate_fn=self.data_collator_mt),
            ]

        return self.val.get_dataloader(**self.val_dataloader_conf, collate_fn=self.data_collator_train)

    def test_dataloader(self):
        return self.test.get_dataloader(**self.test_dataloader_conf, collate_fn=self.data_collator_test)
