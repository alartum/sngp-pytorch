from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays
        self.lens = np.array([arr.shape[0] for arr in self.arrays])
        if not np.all(self.lens == self.lens[0]):
            raise ValueError(
                "All arrays must have the same number of samples, "
                f"but got {self.lens.tolist()}."
            )

    def __len__(self):
        return self.lens[0]

    def __getitem__(self, index):
        samples = []
        for arr in self.arrays:
            sample = arr[index]
            if len(arr.shape) > 1:
                sample = torch.from_numpy(sample)
            samples.append(sample)

        return samples


class EmbeddingsDataModule(LightningDataModule):
    def __init__(
        self,
        train: Tuple[Union[str, Path], Union[str, Path]],
        val: Optional[Tuple[Union[str, Path], Union[str, Path]]] = None,
        test: Optional[Tuple[Union[str, Path], Union[str, Path]]] = None,
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        mmap_mode: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.mmap_mode = mmap_mode

        self.train = train
        self.val = val
        self.test = test

        self.n_classes = None
        self.embeddings = {}
        self.targets = {}

    def load_mmap_(self, mode: str = "train"):
        emb_path, tgt_path = getattr(self, mode)
        self.embeddings[mode] = np.load(emb_path, mmap_mode=self.mmap_mode)
        self.targets[mode] = np.load(tgt_path, mmap_mode=self.mmap_mode)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.load_mmap_("train")

            if self.val is not None:
                self.load_mmap_("val")

        if stage == "test" or stage is None:
            if self.test is not None:
                self.load_mmap_("test")

    @property
    def num_classes(self) -> int:
        if self.n_classes is None:
            self.n_classes = len(np.unique(self.targets["train"]))
        return self.n_classes

    def get_dataloader_(
        self, mode: str = "train", batch_size: Optional[int] = None
    ):
        if batch_size is None:
            batch_size = self.batch_size

        dataset = NumpyDataset(self.embeddings[mode], self.targets[mode])
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=self.shuffle and mode == "train",
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def train_dataloader(self, batch_size: Optional[int] = None):
        return self.get_dataloader_("train", batch_size)

    def val_dataloader(self, batch_size: Optional[int] = None):
        return self.get_dataloader_("val", batch_size)

    def test_dataloader(self, batch_size: Optional[int] = None):
        return self.get_dataloader_("test", batch_size)
