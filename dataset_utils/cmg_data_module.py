import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, GPT2Tokenizer

import hydra
from omegaconf import DictConfig

from dataset_utils.cmg_dataset_w_history import CMGDatasetWithHistory
from dataset_utils.data_collator_w_history import DataCollatorWithHistory
from dataset_utils.sampler_by_author import SamplerByAuthor


class CMGDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_root: str,
                 diff_max_len: int,
                 msg_max_len: int,
                 history_max_len: int,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 train_dataloader_conf: DictConfig,
                 test_dataloader_conf: DictConfig):
        super().__init__()

        self.dataset_root = hydra.utils.to_absolute_path(dataset_root)

        self.diff_max_len = diff_max_len
        self.msg_max_len = msg_max_len

        self.train_dataloader_conf = train_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf

        self._src_tokenizer = RobertaTokenizer.from_pretrained(encoder_name_or_path)
        self._trg_tokenizer = GPT2Tokenizer.from_pretrained(decoder_name_or_path)
        # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
        # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
        self._trg_tokenizer.pad_token = self._trg_tokenizer.unk_token

        self.data_collator = DataCollatorWithHistory(tokenizer=self._trg_tokenizer, max_len=history_max_len)

    def setup(self, stage=None):
        self.test = CMGDatasetWithHistory.load_data(self._src_tokenizer, self._trg_tokenizer,
                                                    path=f"{self.dataset_root}/test.csv")
        self.test_sampler = SamplerByAuthor(self.test)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_conf,
                          collate_fn=self.data_collator, sampler=self.test_sampler)
