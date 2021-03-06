import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, GPT2Tokenizer

import hydra
from omegaconf import DictConfig

from dataset_utils.cmg_dataset import CMGDataset
from dataset_utils.diff_preprocessor import DiffPreprocessor


class CMGDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_root: str,
                 diff_max_len: int,
                 msg_max_len: int,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 train_dataloader_conf: DictConfig,
                 val_dataloader_conf: DictConfig,
                 test_dataloader_conf: DictConfig):
        super().__init__()

        self.dataset_root = hydra.utils.to_absolute_path(dataset_root)

        self.train_data_dir = os.path.join(self.dataset_root, 'train')
        self.val_data_dir = os.path.join(self.dataset_root, 'val')
        self.test_data_dir = os.path.join(self.dataset_root, 'test')

        self.diff_max_len = diff_max_len
        self.msg_max_len = msg_max_len

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf

        # make sure GPT2 appends EOS in begin and end
        # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            return outputs

        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        self._src_tokenizer = RobertaTokenizer.from_pretrained(encoder_name_or_path)
        self._trg_tokenizer = GPT2Tokenizer.from_pretrained(decoder_name_or_path)
        # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
        # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
        self._trg_tokenizer.pad_token = self._trg_tokenizer.unk_token

    def prepare_data(self):
        # called only on 1 GPU
        if 'new_diff.txt' not in os.listdir(self.train_data_dir):
            DiffPreprocessor.create_files(self.dataset_root)

    def setup(self, stage=None):
        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train = CMGDataset.load_data(self._src_tokenizer, self._trg_tokenizer, path=self.train_data_dir,
                                              diff_max_len=self.diff_max_len,
                                              msg_max_len=self.msg_max_len)

            self.val_jiang = CMGDataset.load_data(self._src_tokenizer, self._trg_tokenizer, path=self.val_data_dir,
                                                  diff_max_len=self.diff_max_len,
                                                  msg_max_len=self.msg_max_len)
            self.val_github = CMGDataset.load_data(self._src_tokenizer, self._trg_tokenizer,
                                                   path=f"{hydra.utils.to_absolute_path('raw_data')}/github_data/val",
                                                   diff_max_len=512,
                                                   msg_max_len=512)
        if stage == 'test' or stage is None:
            self.test = CMGDataset.load_data(self._src_tokenizer, self._trg_tokenizer, path=self.test_data_dir,
                                             diff_max_len=self.diff_max_len,
                                             msg_max_len=self.msg_max_len)

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf)

    def val_dataloader(self):
        return [DataLoader(self.val_jiang, **self.val_dataloader_conf),
                DataLoader(self.val_github, **self.val_dataloader_conf)]

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_conf)
