import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from transformers import RobertaTokenizer

from dataset_utils.CMGDataset import CMGDataset
from dataset_utils.DiffPreprocessor import DiffPreprocessor


class CMGDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_size = config['TRAIN_BATCH_SIZE']
        self.val_batch_size = config['VAL_BATCH_SIZE']
        self.test_batch_size = config['TEST_BATCH_SIZE']

        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.config._CONFIG['PAD_TOKEN_ID'] = self.tokenizer.pad_token_id
        self.config._CONFIG['BOS_TOKEN_ID'] = self.tokenizer.bos_token_id
        self.config._CONFIG['EOS_TOKEN_ID'] = self.tokenizer.eos_token_id
        self.config._CONFIG['VOCAB_SIZE'] = self.tokenizer.vocab_size

    def prepare_data(self):
        # called only on 1 GPU
        if 'prev.txt' not in os.listdir(os.path.join(self.config['DATASET_ROOT'], 'train')):
            DiffPreprocessor.create_files(self.config['DATASET_ROOT'])

    def setup(self, stage=None):
        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train = CMGDataset.load_data(self.tokenizer, path=os.path.join(self.config['DATASET_ROOT'], 'train'),
                                              diff_max_len=self.config['DIFF_MAX_LEN'],
                                              msg_max_len=self.config['MSG_MAX_LEN'])
            self.val = CMGDataset.load_data(self.tokenizer, path=os.path.join(self.config['DATASET_ROOT'], 'val'),
                                            diff_max_len=self.config['DIFF_MAX_LEN'],
                                            msg_max_len=self.config['MSG_MAX_LEN'])
        if stage == 'test' or stage is None:
            self.test = CMGDataset.load_data(self.tokenizer, path=os.path.join(self.config['DATASET_ROOT'], 'test'),
                                             diff_max_len=self.config['DIFF_MAX_LEN'],
                                             msg_max_len=self.config['MSG_MAX_LEN'])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, num_workers=4)
