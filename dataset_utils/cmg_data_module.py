import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from transformers import RobertaTokenizer, GPT2Tokenizer

import hydra
from omegaconf import DictConfig

from dataset_utils.cmg_dataset import CMGDataset
from dataset_utils.cmg_dataset_w_history import CMGDatasetWithHistory

from dataset_utils.data_collator_w_history import DataCollatorWithHistory
from dataset_utils.data_collator import DataCollator


class CMGDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_root: str,
                 diff_max_len: int,
                 msg_max_len: int,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 with_history: bool,
                 train_dataloader_conf: DictConfig,
                 test_dataloader_conf: DictConfig):
        super().__init__()

        self.dataset_root = hydra.utils.to_absolute_path(dataset_root)

        self.train_data_dir = os.path.join(self.dataset_root, 'train')
        self.test_data_dir = os.path.join(self.dataset_root, 'test')

        self.diff_max_len = diff_max_len
        self.msg_max_len = msg_max_len

        self.train_dataloader_conf = train_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf

        self.with_history = with_history

        self._src_tokenizer = RobertaTokenizer.from_pretrained(encoder_name_or_path)
        self._trg_tokenizer = GPT2Tokenizer.from_pretrained(decoder_name_or_path)
        # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
        # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
        self._trg_tokenizer.pad_token = self._trg_tokenizer.unk_token

        if self.with_history:
            self.data_collator = DataCollatorWithHistory(tokenizer=self._trg_tokenizer, max_len=1024)
        else:
            # make sure GPT2 appends EOS in begin and end
            # (from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16)
            def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
                outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
                return outputs

            GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
            self.data_collator = DataCollator(tokenizer=self._trg_tokenizer)

    def setup(self, stage=None):
        # called on every GPU
        if self.with_history:
            self.test = CMGDatasetWithHistory.load_data(self._trg_tokenizer, path=self.dataset_root,
                                                        diff_max_len=self.diff_max_len,
                                                        msg_max_len=self.msg_max_len)
        else:
            if stage == 'fit' or stage is None:
                self.train = CMGDataset.load_data(self._src_tokenizer, self._trg_tokenizer, path=self.train_data_dir,
                                                  diff_max_len=self.diff_max_len,
                                                  msg_max_len=self.msg_max_len)

            if stage == 'test' or stage is None:
                self.test = CMGDataset.load_data(self._src_tokenizer, self._trg_tokenizer, path=self.test_data_dir,
                                                 diff_max_len=self.diff_max_len,
                                                 msg_max_len=self.msg_max_len)

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_conf, collate_fn=self.data_collator)
