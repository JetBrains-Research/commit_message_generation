import os
from typing import Dict

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data_utils.cmg_dataset import RetrievalDataset
from src.embedders import BaseEmbedder
from src.utils import RetrievalExample


class CMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        train_fname: str,
        val_fname: str,
        test_fname: str,
        train_dataloader_conf: DictConfig,
        val_dataloader_conf: DictConfig,
        test_dataloader_conf: DictConfig,
    ):
        super().__init__()

        self._dataset_root = hydra.utils.to_absolute_path(dataset_root)
        self._train_fname = train_fname
        self._val_fname = val_fname
        self._test_fname = test_fname

        self.train_dataloader_conf = train_dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataloader_conf = test_dataloader_conf

        self.train: RetrievalDataset
        self.val: RetrievalDataset
        self.test: RetrievalDataset

    @property
    def train_path(self):
        return os.path.join(self._dataset_root, self._train_fname)

    @property
    def val_path(self):
        return os.path.join(self._dataset_root, self._val_fname)

    @property
    def test_path(self):
        return os.path.join(self._dataset_root, self._test_fname)

    def setup(self, embedder: BaseEmbedder, stage=None):
        if stage == "fit" or stage is None:
            self.train = RetrievalDataset(data_filename=self.train_path, embedder=embedder)
            self.val = RetrievalDataset(data_filename=self.val_path, embedder=embedder)
        if stage == "test" or stage is None:
            self.test = RetrievalDataset(data_filename=self.test_path, embedder=embedder)

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf, collate_fn=lambda x: x)

    def val_dataloader(self):
        return DataLoader(self.val, **self.val_dataloader_conf, collate_fn=lambda x: x)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_conf, collate_fn=lambda x: x)
