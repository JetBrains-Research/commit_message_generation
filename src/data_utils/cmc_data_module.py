import logging
import os
from copy import deepcopy
from typing import Optional, Tuple

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
from src.utils import get_decoder_start_token_id

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

        self._data_path = self._create_path(
            msg_tokenizer_name_or_path=model_cfg.msg_tokenizer_name_or_path,
            diff_tokenizer_name_or_path=model_cfg.diff_tokenizer_name_or_path,
            encoder_context_max_len=model_cfg.encoder_context_max_len,
            preprocessor_configuration=model_cfg.preprocessor_configuration,
        )

        self._local_rank = local_rank
        self._world_size = world_size

        self._encoder_context_max_len = model_cfg.encoder_context_max_len
        self._decoder_context_max_len = model_cfg.decoder_context_max_len
        self._add_history_to_inputs = dataset_cfg.add_history_to_inputs
        self._train_with_history = input_cfg.train_with_history
        self._generate_with_history = input_cfg.generate_with_history

        self._line_sep = dataset_cfg.line_sep
        self._use_cache = dataset_cfg.use_cache
        self._process_retrieved = process_retrieved
        self._use_eval_downsample = dataset_cfg.use_eval_downsample
        self._use_train_downsample = False
        if dataset_cfg.stage == "sweep":
            logging.info("Setup for sweep: will use dataset subset for all parts.")
            self._use_eval_downsample = True
            self._use_train_downsample = True

        self.diff_tokenizer, self.msg_tokenizer = self._load_tokenizers(
            msg_tokenizer_name_or_path=model_cfg.msg_tokenizer_name_or_path,
            diff_tokenizer_name_or_path=model_cfg.diff_tokenizer_name_or_path,
            configuration=model_cfg.preprocessor_configuration,
        )

        self._preprocessor: BasePreprocessor = self._init_preprocessor(
            preprocessor_configuration=model_cfg.preprocessor_configuration,
            preprocessor_chunksize=dataset_cfg.preprocessor_chunksize,
        )

        self.data_collator_fit = self._init_collator_fit(
            input_cfg=input_cfg,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
            process_retrieved=process_retrieved,
            shift_labels=shift_labels,
        )
        self.data_collator_test = self._init_collator_generate(
            input_cfg=input_cfg, model_cfg=model_cfg, dataset_cfg=dataset_cfg, process_retrieved=process_retrieved
        )

        self.train_dataloader_conf = dataset_cfg.train_dataloader_conf
        self.val_dataloader_conf = dataset_cfg.val_dataloader_conf
        self.test_dataloader_conf = dataset_cfg.test_dataloader_conf

        # datasets are initialized later
        self.train = None
        self.val = None
        self.test = None

    def _init_preprocessor(self, preprocessor_configuration: str, preprocessor_chunksize: int) -> BasePreprocessor:
        """Initializes a correct preprocessor type based on passed configuration.

        Args:
            preprocessor_configuration: A type of preprocessor to initialize.
            preprocessor_chunksize: A number of example in a single chunk during preprocessing.

        Returns:
            Initialized preprocessor.
        """
        if preprocessor_configuration == "default":
            return DefaultPreprocessor(
                diff_tokenizer=self.diff_tokenizer,
                msg_tokenizer=self.msg_tokenizer,
                chunksize=preprocessor_chunksize,
            )
        elif preprocessor_configuration == "race":
            return RACEPreprocessor(
                diff_tokenizer=self.diff_tokenizer,
                msg_tokenizer=self.msg_tokenizer,
                chunksize=preprocessor_chunksize,
            )
        elif preprocessor_configuration == "codereviewer":
            return CodeReviewerPreprocessor(
                diff_tokenizer=self.diff_tokenizer,
                msg_tokenizer=self.msg_tokenizer,
                chunksize=preprocessor_chunksize,
            )
        else:
            raise ValueError(f"Current preprocessor configuration ({preprocessor_configuration}) is not supported.")

    def _init_collator_fit(
        self,
        input_cfg: InputConfig,
        model_cfg: BaseModelConfig,
        dataset_cfg: DatasetConfig,
        process_retrieved: bool,
        shift_labels: bool,
    ) -> DataCollatorTrain:
        return DataCollatorTrain(
            diff_bos_token_id=self.diff_tokenizer.bos_token_id,  # type: ignore[attr-defined]
            diff_eos_token_id=self.diff_tokenizer.eos_token_id,  # type: ignore[attr-defined]
            diff_pad_token_id=self.diff_tokenizer.pad_token_id,  # type: ignore[attr-defined]
            msg_bos_token_id=self.msg_tokenizer.bos_token_id,  # type: ignore[attr-defined]
            msg_eos_token_id=self.msg_tokenizer.eos_token_id,  # type: ignore[attr-defined]
            msg_pad_token_id=self.msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
            msg_sep_token_id=self.msg_tokenizer.sep_token_id,  # type: ignore[attr-defined]
            encoder_input_type=input_cfg.encoder_input_type,
            encoder_context_max_len=model_cfg.encoder_context_max_len,
            decoder_context_max_len=model_cfg.decoder_context_max_len,
            with_history=input_cfg.train_with_history,
            process_retrieved=process_retrieved,
            shift_labels=shift_labels,
            testing=dataset_cfg.testing,
            decoder_start_token_id=get_decoder_start_token_id(model_cfg),
        )

    def _init_collator_generate(
        self, input_cfg: InputConfig, model_cfg: BaseModelConfig, dataset_cfg: DatasetConfig, process_retrieved: bool
    ) -> DataCollatorTest:
        return DataCollatorTest(
            diff_bos_token_id=self.diff_tokenizer.bos_token_id,  # type: ignore[attr-defined]
            diff_eos_token_id=self.diff_tokenizer.eos_token_id,  # type: ignore[attr-defined]
            diff_pad_token_id=self.diff_tokenizer.pad_token_id,  # type: ignore[attr-defined]
            msg_bos_token_id=self.msg_tokenizer.bos_token_id,  # type: ignore[attr-defined]
            msg_eos_token_id=self.msg_tokenizer.eos_token_id,  # type: ignore[attr-defined]
            msg_pad_token_id=self.msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
            msg_sep_token_id=self.msg_tokenizer.sep_token_id,  # type: ignore[attr-defined]
            diff_tokenizer=self.diff_tokenizer,
            msg_tokenizer=self.msg_tokenizer,
            encoder_input_type=input_cfg.encoder_input_type,
            encoder_context_max_len=model_cfg.encoder_context_max_len,
            decoder_context_max_len=model_cfg.decoder_context_max_len,
            with_history=input_cfg.generate_with_history,
            context_ratio=input_cfg.context_ratio,
            process_retrieved=process_retrieved,
            testing=dataset_cfg.testing,
            decoder_start_token_id=get_decoder_start_token_id(model_cfg),
        )

    def _create_path(
        self,
        msg_tokenizer_name_or_path: str,
        diff_tokenizer_name_or_path: Optional[str],
        encoder_context_max_len: int,
        preprocessor_configuration: str,
    ) -> str:
        """Builds a path to preprocessed data based on given configuration."""
        if "/" in msg_tokenizer_name_or_path:
            msgs_tok_dir = msg_tokenizer_name_or_path.split("/")[-1]
        else:
            msgs_tok_dir = msg_tokenizer_name_or_path

        if not diff_tokenizer_name_or_path:
            diffs_tok_dir = msgs_tok_dir
        elif "/" in diff_tokenizer_name_or_path:
            diffs_tok_dir = diff_tokenizer_name_or_path.split("/")[-1]
        else:
            diffs_tok_dir = diff_tokenizer_name_or_path

        data_path = os.path.join(
            self._dataset_root,
            "-".join(
                [
                    diffs_tok_dir,
                    str(encoder_context_max_len),
                    msgs_tok_dir,
                    preprocessor_configuration,
                ]
            ),
        )

        os.makedirs(data_path, exist_ok=True)
        return data_path

    def _add_special_tokens(
        self, tokenizer: PreTrainedTokenizerFast, preprocessor_configuration: str
    ) -> PreTrainedTokenizerFast:
        """Adds special tokens to tokenizer based on preprocessor configuration.

        * sep_token is necessary for correct history construction.
        * pad_token is necessary for correct batch construction.
        * Several models employ additional special tokens in diffs representation.
        """
        if not tokenizer.sep_token:  # type: ignore[attr-defined]
            tokenizer.add_special_tokens({"sep_token": "[SEP]"})  # type: ignore[attr-defined]
        if not tokenizer.pad_token:  # type: ignore[attr-defined]
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # type: ignore[attr-defined]

        if preprocessor_configuration == "codereviewer":
            tokenizer.add_special_tokens({"additional_special_tokens": ["<add>", "<del>", "<keep>"]})  # type: ignore[attr-defined]
        elif preprocessor_configuration == "race":
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
    ) -> Tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
        """Initializes tokenizers and adds special tokens when necessary."""
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
        for part in ["train", "val", "test"]:
            input_dir = self._dataset_root
            data_dir = self._data_path

            if (part != "train" and self._use_eval_downsample) or (part == "train" and self._use_train_downsample):
                logging.info(f"Will use dataset subset for {part}.")
                input_dir = os.path.join(input_dir, "downsample")
                data_dir = os.path.join(data_dir, "downsample")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(data_dir, exist_ok=True)

            self._preprocessor.process(
                input_dir=input_dir,
                data_dir=data_dir,
                part=part,
                message_kwargs={},
                diff_kwargs={"max_len": self._encoder_context_max_len, "line_sep": self._line_sep},
                use_cache=self._use_cache,
                add_history_to_inputs=self._add_history_to_inputs,
                decoder_context_max_length=self._decoder_context_max_len,
            )
            if self._process_retrieved:
                self._preprocessor.process_retrieved(
                    data_dir=self._data_path,
                    retrieved_dir=os.path.join(input_dir, "retrieval"),
                    part=part,
                    use_cache=self._use_cache,
                )

        self._use_cache = True

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage == "sweep" or stage is None:

            self.train = CMCDatasetWithHistory.load_data(
                history_path=os.path.join(
                    self._data_path, ("downsample" if self._use_train_downsample else ""), "train_history.json"
                ),
                data_path=os.path.join(
                    self._data_path, ("downsample" if self._use_train_downsample else ""), "train_shuffled.jsonl"
                ),
                retrieved_data_path=os.path.join(
                    self._data_path,
                    ("downsample" if self._use_train_downsample else ""),
                    "retrieved_train_shuffled.jsonl",
                )
                if self._process_retrieved
                else None,
                rank=self._local_rank,
                world_size=self._world_size,
                history_mode="io" if self._add_history_to_inputs else "ram",
            )
            self.val = CMCDatasetWithHistory.load_data(
                history_path=os.path.join(
                    self._data_path, ("downsample" if self._use_eval_downsample else ""), "val_history.json"
                ),
                data_path=os.path.join(
                    self._data_path,
                    ("downsample" if self._use_eval_downsample else ""),
                    f"val_processed{'_history' if self._add_history_to_inputs else ''}.jsonl",
                ),
                retrieved_data_path=os.path.join(
                    self._data_path,
                    ("downsample" if self._use_eval_downsample else ""),
                    "retrieved_val_processed.jsonl",
                )
                if self._process_retrieved
                else None,
                rank=self._local_rank,
                world_size=self._world_size,
                history_mode="io" if self._add_history_to_inputs else "ram",
            )

        if stage == "test" or stage is None:
            self.test = CMCDatasetWithHistory.load_data(
                history_path=os.path.join(
                    self._data_path, ("downsample" if self._use_eval_downsample else ""), "test_history.json"
                ),
                data_path=os.path.join(
                    self._data_path,
                    ("downsample" if self._use_eval_downsample else ""),
                    f"test_processed{'_history' if self._add_history_to_inputs else ''}.jsonl",
                ),
                retrieved_data_path=os.path.join(
                    self._data_path,
                    ("downsample" if self._use_eval_downsample else ""),
                    "retrieved_test_processed.jsonl",
                )
                if self._process_retrieved
                else None,
                rank=self._local_rank,
                world_size=self._world_size,
                history_mode="io" if self._add_history_to_inputs else "ram",
            )

    def train_dataloader(self):
        return self.train.get_dataloader(
            batch_size=self.train_dataloader_conf.batch_size,
            num_workers=self.train_dataloader_conf.num_workers,
            collate_fn=self.data_collator_fit,
        )

    def val_dataloader(self):
        return self.val.get_dataloader(
            batch_size=self.val_dataloader_conf.batch_size,
            num_workers=self.val_dataloader_conf.num_workers,
            collate_fn=self.data_collator_fit,
        )

    def test_dataloader(self):
        return self.test.get_dataloader(
            batch_size=self.test_dataloader_conf.batch_size,
            num_workers=self.test_dataloader_conf.num_workers,
            collate_fn=self.data_collator_test,
        )
