# Source
# https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/datamodules/cifar10_datamodule.py

from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import CIFAR100
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    CIFAR100 = None


class CIFAR100DataModule(VisionDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/
        wp-content/uploads/2019/01/
        Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png

        :width: 400
        :alt: CIFAR-100

    Specs:
        - 100 classes (1 per class)
        - Each image is (3 x 32 x 32)

    Standard CIFAR100, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])

    Example::

        from pl_bolts.datamodules import CIFAR100DataModule

        dm = CIFAR100DataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "cifar100"
    dataset_cls = CIFAR100
    dims = (3, 32, 32)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use
                       for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors
                        into CUDA pinned memory before returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

        self.train_transforms = transform_lib.Compose(
            [
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                cifar10_normalization(),
            ]
        )
        self.val_transforms = transform_lib.Compose(
            [transform_lib.ToTensor(), cifar10_normalization()]
        )
        self.test_transforms = transform_lib.Compose(
            [transform_lib.ToTensor(), cifar10_normalization()]
        )

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            100
        """
        return 100

    def default_transforms(self) -> Callable:
        transforms_list = [
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
        ]
        if self.normalize:
            transforms_list.append(cifar10_normalization())

        return transform_lib.Compose(transforms_list)
