import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math


class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data=None, batch_size=32, num_workers=0, transform=None):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.train_data
            self.val_dataset = self.val_data

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            if self.test_data is not None:
                self.test_dataset = self.test_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_data is not None:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
