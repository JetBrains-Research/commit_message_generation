import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer

from .cmc_dataset_w_history import CMCDatasetWithHistory
from .data_collator import DataCollatorTest, DataCollatorTrain


class CMCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        diffs_folder: str,
        msgs_folder: str,
        encoder_input_type: str,
        encoder_context_max_len: int,
        decoder_context_max_len: int,
        diff_tokenizer_name_or_path: str,
        msg_tokenizer_name_or_path: str,
        local_rank: int,
        world_size: int,
        train_with_history: bool,
        generate_with_history: bool,
        shift_labels: bool,
        context_ratio: float,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
        test_dataloader_conf: Optional[DictConfig] = None,
        testing: bool = False,
    ):
        super().__init__()

        dataset_root = hydra.utils.to_absolute_path(dataset_root)
        self.diffs_path = os.path.join(dataset_root, diffs_folder, "diffs", str(encoder_context_max_len))
        self.msgs_path = os.path.join(dataset_root, msgs_folder, "msgs")

        self.local_rank = local_rank
        self.world_size = world_size

        self.train_len: int

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf

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
            self._msg_tokenizer.sep_token = self._msg_tokenizer.unk_token

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

        self.data_collator_train = DataCollatorTrain(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            encoder_input_type=encoder_input_type,
            encoder_context_max_len=encoder_context_max_len,
            decoder_context_max_len=decoder_context_max_len,
            with_history=train_with_history,
            shift_labels=shift_labels,
            testing=testing,
        )
        self.data_collator_test = DataCollatorTest(
            diff_tokenizer=self._diff_tokenizer,
            msg_tokenizer=self._msg_tokenizer,
            encoder_input_type=encoder_input_type,
            encoder_context_max_len=encoder_context_max_len,
            decoder_context_max_len=decoder_context_max_len,
            with_history=generate_with_history,
            context_ratio=context_ratio,
            testing=testing,
        )

        # datasets are initialized later
        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage=None):
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

        if stage == "test" or stage is None:
            self.test = CMCDatasetWithHistory.load_data(
                diffs_path=self.diffs_path,
                msgs_path=self.msgs_path,
                part="test",
                rank=self.local_rank,
                world_size=self.world_size,
            )
        if stage == "sweep":
            # when tuning hyperparameters, run test logic but on validation
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
        return self.val.get_dataloader(**self.val_dataloader_conf, collate_fn=self.data_collator_train)

    def test_dataloader(self):
        return self.test.get_dataloader(**self.test_dataloader_conf, collate_fn=self.data_collator_test)
