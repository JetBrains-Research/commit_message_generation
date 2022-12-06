import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .cmg_dataset import RetrievalDataset


def collate_fn(x):
    return x


class CMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        train_dataloader_conf: DictConfig,
        val_dataloader_conf: DictConfig,
        test_dataloader_conf: DictConfig,
    ):
        super().__init__()

        self._dataset_root = hydra.utils.to_absolute_path(dataset_root)

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf

        self.train: RetrievalDataset
        self.val: RetrievalDataset
        self.test: RetrievalDataset

    @property
    def train_path(self):
        return os.path.join(self._dataset_root, "train.jsonl")

    @property
    def val_path(self):
        return os.path.join(self._dataset_root, "val.jsonl")

    @property
    def test_path(self):
        return os.path.join(self._dataset_root, "test.jsonl")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = RetrievalDataset(data_filename=self.train_path)
            self.val = RetrievalDataset(data_filename=self.val_path)
        if stage == "test" or stage is None:
            self.test = RetrievalDataset(data_filename=self.test_path)

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, **self.val_dataloader_conf, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_conf, collate_fn=collate_fn)
