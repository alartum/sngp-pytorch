from typing import Dict, Optional  # List, Callable

# import torch
from pytorch_lightning import LightningDataModule

# from warnings import warn


# from torch.utils.data import DataLoader


class EmbeddingsDataModule(LightningDataModule):
    def __init__(self, embs=Dict[str, str], tgts=Dict[str, str]):
        super().__init__()

    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage=stage)
