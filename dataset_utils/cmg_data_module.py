import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, GPT2Tokenizer

import hydra
from omegaconf import DictConfig

from dataset_utils.cmg_dataset_w_history import CMGDatasetWithHistory
from dataset_utils.data_collator_w_history import DataCollatorWithHistory
from dataset_utils.data_preprocessor import DataPreprocessor


class CMGDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_root: str,
                 history_max_len: int,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 local_rank: int,
                 world_size: int,
                 train_dataloader_conf: DictConfig,
                 val_dataloader_conf: DictConfig,
                 test_dataloader_conf: DictConfig):
        super().__init__()

        self.dataset_root = hydra.utils.to_absolute_path(dataset_root)
        self.history_max_len = history_max_len

        self.local_rank = local_rank
        self.world_size = world_size

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf

        self._src_tokenizer = RobertaTokenizer.from_pretrained(encoder_name_or_path)
        self._trg_tokenizer = GPT2Tokenizer.from_pretrained(decoder_name_or_path)

        # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
        # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
        self._trg_tokenizer.pad_token = self._trg_tokenizer.unk_token

        self.data_collator = DataCollatorWithHistory(src_tokenizer=self._src_tokenizer,
                                                     trg_tokenizer=self._trg_tokenizer,
                                                     max_len=self.history_max_len,
                                                     testing=False)

        # datasets are initialized later
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        # called only on 1 GPU
        if 'train.json' not in os.listdir(self.dataset_root):
            DataPreprocessor.create_files(self.dataset_root)

    def setup(self, stage=None):
        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train = CMGDatasetWithHistory.load_data(self.dataset_root + '/train',
                                                         rank=self.local_rank,
                                                         world_size=self.world_size)

            self.val = CMGDatasetWithHistory.load_data(self.dataset_root + '/val',
                                                       rank=self.local_rank,
                                                       world_size=self.world_size)
        if stage == 'test' or stage is None:
            self.test = CMGDatasetWithHistory.load_data(self.dataset_root + '/test',
                                                        rank=self.local_rank,
                                                        world_size=self.world_size)

    def train_dataloader(self):
        return self.train.get_dataloader(**self.train_dataloader_conf, collate_fn=self.data_collator)

    def val_dataloader(self):
        return self.val.get_dataloader(**self.val_dataloader_conf, collate_fn=self.data_collator)

    def test_dataloader(self):
        return self.test.get_dataloader(**self.test_dataloader_conf, collate_fn=self.data_collator)
