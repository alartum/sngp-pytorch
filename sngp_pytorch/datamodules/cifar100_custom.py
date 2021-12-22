from pathlib import Path
from typing import Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from image_uncertainty.cifar.cifar_datasets import get_training_dataloader


class CIFAR100CustomDataModule(LightningDataModule):
    n_classes = 100

    def __init__(
        self,
        root: Union[Path, str],
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        val_size: float = 0.8,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.val_size = val_size

        self.train_mean = (
            0.5070751592371323,
            0.48654887331495095,
            0.4409178433670343,
        )
        self.train_std = (
            0.2673342858792401,
            0.2564384629170883,
            0.27615047132568404,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.loaders = {}
        self.loaders["train"], self.loaders["val"] = get_training_dataloader(
            self.root,
            self.train_mean,
            self.train_std,
            self.batch_size,
            self.num_workers,
            self.shuffle,
            None,
            self.seed,
            self.val_size,
        )

    @property
    def num_classes(self) -> int:
        return self.n_classes

    def get_dataloader_(
        self, mode: str = "train", batch_size: Optional[int] = None
    ):
        if batch_size is None:
            batch_size = self.batch_size

        dataset = self.loaders[mode].dataset
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
