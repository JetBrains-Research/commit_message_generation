import pytorch_lightning as pl

from transformers import RobertaTokenizer, GPT2Tokenizer

import hydra
from omegaconf import DictConfig

from src.dataset_utils.cmg_dataset_w_history import CMGDatasetWithHistory
from src.dataset_utils.data_collators import DataCollatorWithHistory, DataCollatorWithoutHistory


class CMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        history_max_len: int,
        with_history: bool,
        encoder_name_or_path: str,
        decoder_name_or_path: str,
        local_rank: int,
        world_size: int,
        test_dataloader_conf: DictConfig,
        **kwargs
    ):
        super().__init__()

        self.dataset_root = hydra.utils.to_absolute_path(dataset_root)
        self.history_max_len = history_max_len

        self.local_rank = local_rank
        self.world_size = world_size

        self.test_dataloader_conf = test_dataloader_conf

        self._src_tokenizer = RobertaTokenizer.from_pretrained(encoder_name_or_path)
        self._trg_tokenizer = GPT2Tokenizer.from_pretrained(decoder_name_or_path)

        # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
        # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
        self._trg_tokenizer.pad_token = self._trg_tokenizer.unk_token

        if with_history:
            self.data_collator = DataCollatorWithHistory(
                src_tokenizer=self._src_tokenizer, trg_tokenizer=self._trg_tokenizer, max_len=self.history_max_len
            )
        else:
            self.data_collator = DataCollatorWithoutHistory(
                src_tokenizer=self._src_tokenizer, trg_tokenizer=self._trg_tokenizer, max_len=self.history_max_len
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
