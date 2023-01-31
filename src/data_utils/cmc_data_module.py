import logging
import os
from copy import deepcopy
from typing import Optional

import hydra
import pytorch_lightning as pl
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from conf import BaseModelConfig, DatasetConfig, InputConfig
from src.data_utils.preprocessors import (
    BasePreprocessor,
    CodeReviewerPreprocessor,
    DefaultPreprocessor,
    RACEPreprocessor,
)

from .cmc_dataset_w_history import CMCDatasetWithHistory
from .data_collators import DataCollatorTest, DataCollatorTrain


class CMCDataModule(pl.LightningDataModule):
    """ """

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        model_cfg: BaseModelConfig,
        input_cfg: InputConfig,
        local_rank: int,
        world_size: int,
        shift_labels: bool,
        process_retrieved: bool,
    ):
        super().__init__()

        self._dataset_root = hydra.utils.to_absolute_path(dataset_cfg.dataset_root)

        if not model_cfg.msg_tokenizer_name_or_path:
            msgs_tok_dir = ""
        elif "/" in model_cfg.msg_tokenizer_name_or_path:
            msgs_tok_dir = model_cfg.msg_tokenizer_name_or_path.split("/")[-1]
        else:
            msgs_tok_dir = model_cfg.msg_tokenizer_name_or_path

        if not model_cfg.diff_tokenizer_name_or_path:
            diffs_tok_dir = msgs_tok_dir
        elif "/" in model_cfg.diff_tokenizer_name_or_path:
            diffs_tok_dir = model_cfg.diff_tokenizer_name_or_path.split("/")[-1]
        else:
            diffs_tok_dir = model_cfg.diff_tokenizer_name_or_path

        self._data_path = os.path.join(
            self._dataset_root,
            "-".join(
                [
                    diffs_tok_dir,
                    str(model_cfg.encoder_context_max_len),
                    msgs_tok_dir,
                    model_cfg.preprocessor_configuration,
                ]
            ),
        )
        os.makedirs(self._data_path, exist_ok=True)

        self._local_rank = local_rank
        self._world_size = world_size

        self._encoder_context_max_len = model_cfg.encoder_context_max_len
        self._line_sep = dataset_cfg.line_sep
        self._use_cache = dataset_cfg.use_cache
        self._process_retrieved = process_retrieved

        self.diff_tokenizer, self.msg_tokenizer = self._load_tokenizers(
            msg_tokenizer_name_or_path=model_cfg.msg_tokenizer_name_or_path,
            diff_tokenizer_name_or_path=model_cfg.diff_tokenizer_name_or_path,
            configuration=model_cfg.preprocessor_configuration,
        )

        self._preprocessor: BasePreprocessor
        if model_cfg.preprocessor_configuration == "default":
            self._preprocessor = DefaultPreprocessor(
                diff_tokenizer=self.diff_tokenizer,
                msg_tokenizer=self.msg_tokenizer,
                chunksize=dataset_cfg.preprocessor_chunksize,
            )
        elif model_cfg.preprocessor_configuration == "race":
            self._preprocessor = RACEPreprocessor(
                diff_tokenizer=self.diff_tokenizer,
                msg_tokenizer=self.msg_tokenizer,
                chunksize=dataset_cfg.preprocessor_chunksize,
            )
        elif model_cfg.preprocessor_configuration == "codereviewer":
            self._preprocessor = CodeReviewerPreprocessor(
                diff_tokenizer=self.diff_tokenizer,
                msg_tokenizer=self.msg_tokenizer,
                chunksize=dataset_cfg.preprocessor_chunksize,
            )
        else:
            raise ValueError("Current preprocessor configuration is not supported.")

        self.data_collator_train = DataCollatorTrain(
            diff_bos_token_id=self.diff_tokenizer.bos_token_id,
            diff_eos_token_id=self.diff_tokenizer.eos_token_id,
            diff_pad_token_id=self.diff_tokenizer.pad_token_id,
            msg_bos_token_id=self.msg_tokenizer.bos_token_id,
            msg_eos_token_id=self.msg_tokenizer.eos_token_id,
            msg_pad_token_id=self.msg_tokenizer.pad_token_id,
            msg_sep_token_id=self.msg_tokenizer.sep_token_id,
            encoder_input_type=input_cfg.encoder_input_type,
            encoder_context_max_len=model_cfg.encoder_context_max_len,
            decoder_context_max_len=model_cfg.decoder_context_max_len,
            with_history=input_cfg.train_with_history,
            process_retrieved=process_retrieved,
            shift_labels=shift_labels,
            testing=dataset_cfg.testing,
        )
        self.data_collator_test = DataCollatorTest(
            diff_bos_token_id=self.diff_tokenizer.bos_token_id,
            diff_eos_token_id=self.diff_tokenizer.eos_token_id,
            diff_pad_token_id=self.diff_tokenizer.pad_token_id,
            msg_bos_token_id=self.msg_tokenizer.bos_token_id,
            msg_eos_token_id=self.msg_tokenizer.eos_token_id,
            msg_pad_token_id=self.msg_tokenizer.pad_token_id,
            msg_sep_token_id=self.msg_tokenizer.sep_token_id,
            diff_tokenizer=self.diff_tokenizer,
            msg_tokenizer=self.msg_tokenizer,
            encoder_input_type=input_cfg.encoder_input_type,
            encoder_context_max_len=model_cfg.encoder_context_max_len,
            decoder_context_max_len=model_cfg.decoder_context_max_len,
            with_history=input_cfg.generate_with_history,
            context_ratio=input_cfg.context_ratio,
            process_retrieved=process_retrieved,
            testing=dataset_cfg.testing,
        )

        self.train_dataloader_conf = dataset_cfg.train_dataloader_conf
        self.val_dataloader_conf = dataset_cfg.val_dataloader_conf
        self.test_dataloader_conf = dataset_cfg.test_dataloader_conf

        # datasets are initialized later
        self.train = None
        self.val = None
        self.test = None

    def _add_special_tokens(self, tokenizer: PreTrainedTokenizerFast, configuration: str) -> PreTrainedTokenizerFast:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[NL]"]})  # type: ignore[attr-defined]
        if not tokenizer.sep_token:  # type: ignore[attr-defined]
            tokenizer.add_special_tokens({"sep_token": "[SEP]"})  # type: ignore[attr-defined]
        if not tokenizer.pad_token:  # type: ignore[attr-defined]
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # type: ignore[attr-defined]

        if configuration == "codereviewer":
            tokenizer.add_special_tokens({"additional_special_tokens": ["<add>", "<del>", "<keep>"]})  # type: ignore[attr-defined]
        elif configuration == "race":
            tokenizer.add_special_tokens(  # type: ignore[attr-defined]
                {
                    "additional_special_tokens": [
                        "<KEEP>",
                        "<KEEP_END>",
                        "<INSERT>",
                        "<INSERT_END>",
                        "<DELETE>",
                        "<DELETE_END>",
                        "<REPLACE_OLD>",
                        "<REPLACE_NEW>",
                        "<REPLACE_END>",
                    ]
                }
            )
        return tokenizer

    def _load_tokenizers(
        self, msg_tokenizer_name_or_path: str, diff_tokenizer_name_or_path: Optional[str], configuration: str
    ):
        try:
            msg_tokenizer = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path)
        except ValueError:
            msg_tokenizer = AutoTokenizer.from_pretrained(hydra.utils.to_absolute_path(msg_tokenizer_name_or_path))

        msg_tokenizer = self._add_special_tokens(msg_tokenizer, configuration)

        if not diff_tokenizer_name_or_path:
            logging.warning("Diff tokenizer is not set, using message tokenizer as a default")
            diff_tokenizer = deepcopy(msg_tokenizer)
        elif diff_tokenizer_name_or_path == msg_tokenizer_name_or_path:
            diff_tokenizer = deepcopy(msg_tokenizer)
        else:
            try:
                diff_tokenizer = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path)
            except ValueError:
                diff_tokenizer = AutoTokenizer.from_pretrained(
                    hydra.utils.to_absolute_path(diff_tokenizer_name_or_path)
                )
            diff_tokenizer = self._add_special_tokens(diff_tokenizer, configuration)

        return diff_tokenizer, msg_tokenizer

    def prepare_data(self) -> None:  # type: ignore[override]
        if self._use_cache:
            logging.info("Using preprocessed input")
            for part in ["train", "val", "test"]:
                assert f"{part}_processed.jsonl" in os.listdir(self._data_path)
                assert f"{part}_history.json" in os.listdir(self._data_path)
                if self._process_retrieved:
                    assert f"retrieved_{part}_processed.jsonl" in os.listdir(self._data_path)
        else:
            for part in ["train", "val", "test"]:
                self._preprocessor.process(
                    input_dir=self._dataset_root,
                    data_dir=self._data_path,
                    part=part,
                    message_kwargs={},
                    diff_kwargs={"max_len": self._encoder_context_max_len, "line_sep": self._line_sep},
                )
                if self._process_retrieved:
                    self._preprocessor.process_retrieved(
                        data_dir=self._data_path,
                        retrieved_dir=f"{self._dataset_root}/retrieval",
                        part=part,
                    )

            self._use_cache = True

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = CMCDatasetWithHistory.load_data(
                history_path=os.path.join(self._data_path, "train_history.json"),
                data_path=os.path.join(self._data_path, "train_shuffled.jsonl"),
                retrieved_data_path=os.path.join(self._data_path, "retrieved_train_shuffled.jsonl")
                if self._process_retrieved
                else None,
                rank=self._local_rank,
                world_size=self._world_size,
            )
            self.val = CMCDatasetWithHistory.load_data(
                history_path=os.path.join(self._data_path, "val_history.json"),
                data_path=os.path.join(self._data_path, "val_processed.jsonl"),
                retrieved_data_path=os.path.join(self._data_path, "retrieved_val_processed.jsonl")
                if self._process_retrieved
                else None,
                rank=self._local_rank,
                world_size=self._world_size,
            )

        if stage == "test" or stage is None:
            self.test = CMCDatasetWithHistory.load_data(
                history_path=os.path.join(self._data_path, "test_history.json"),
                data_path=os.path.join(self._data_path, "test_processed.jsonl"),
                retrieved_data_path=os.path.join(self._data_path, "retrieved_test_processed.jsonl")
                if self._process_retrieved
                else None,
                rank=self._local_rank,
                world_size=self._world_size,
            )
        if stage == "sweep":
            # when tuning hyperparameters, run test logic but on validation data
            self.test = CMCDatasetWithHistory.load_data(
                history_path=os.path.join(self._data_path, "val_history.json"),
                data_path=os.path.join(self._data_path, "val_processed.jsonl"),
                retrieved_data_path=os.path.join(self._data_path, "retrieved_val_processed.jsonl")
                if self._process_retrieved
                else None,
                rank=self._local_rank,
                world_size=self._world_size,
            )

    def train_dataloader(self):
        return self.train.get_dataloader(
            batch_size=self.train_dataloader_conf.batch_size,
            num_workers=self.train_dataloader_conf.num_workers,
            collate_fn=self.data_collator_train,
        )

    def val_dataloader(self):
        return self.val.get_dataloader(
            batch_size=self.val_dataloader_conf.batch_size,
            num_workers=self.val_dataloader_conf.num_workers,
            collate_fn=self.data_collator_train,
        )

    def test_dataloader(self):
        return self.test.get_dataloader(
            batch_size=self.test_dataloader_conf.batch_size,
            num_workers=self.test_dataloader_conf.num_workers,
            collate_fn=self.data_collator_test,
        )
