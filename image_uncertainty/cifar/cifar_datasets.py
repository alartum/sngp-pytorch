import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.svhn import SVHN

from image_uncertainty.datasets.smooth_random import SmoothRandom

from . import settings

label_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "computer_keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

label_mapping = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetables": [
        "apple",
        "mushroom",
        "orange",
        "pear",
        "sweet_pepper",
    ],
    "household electrical device": [
        "clock",
        "computer_keyboard",
        "lamp",
        "telephone",
        "television",
    ],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": [
        "bridge",
        "castle",
        "house",
        "road",
        "skyscraper",
    ],
    "large natural outdoor scenes": [
        "cloud",
        "forest",
        "mountain",
        "plain",
        "sea",
    ],
    "large omnivores and herbivores": [
        "camel",
        "cattle",
        "chimpanzee",
        "elephant",
        "kangaroo",
    ],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": [
        "maple_tree",
        "oak_tree",
        "palm_tree",
        "pine_tree",
        "willow_tree",
    ],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


def ood_classes(ood_categories):
    oods = []
    for category in ood_categories:
        for klass in label_mapping[category]:
            oods.append(label_names.index(klass))
    return oods


class CIFAR100_WITH_OOD(torchvision.datasets.CIFAR100):
    def __init__(self, ood_classes, *args, ood=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ood = ood
        self.ood_classes = ood_classes

        if ood:
            ids = np.isin(self.targets, self.ood_classes)
        else:
            ids = ~np.isin(self.targets, self.ood_classes)

        # self.data = self.data[ids][:1000]
        # self.targets = np.array(self.targets)[ids][:1000]
        self.data = self.data[ids]
        self.targets = np.array(self.targets)[ids]


def get_training_dataloader(
    root,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    batch_size=16,
    num_workers=8,
    shuffle=True,
    ood_name="vehicles",
    seed=42,
    val_size=0.1,
):

    # ood_categories = settings.OOD_CATEGORIES[ood_name]

    transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    cifar100 = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train
    )

    torch.manual_seed(seed)
    val_size = int(len(cifar100) * val_size)
    train_size = len(cifar100) - val_size
    train, val = torch.utils.data.random_split(
        cifar100, [train_size, val_size]
    )
    training_loader = DataLoader(
        train,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
    )

    return training_loader, val_loader


def get_external_dataset(ood_name, transform_test):
    if ood_name == "svhn":
        test_dataset = SVHN(
            "./experiments/data",
            split="test",
            download=True,
            transform=transform_test,
        )
    elif ood_name == "lsun":
        test_dataset = LocalImageDataset(
            "./experiments/data/LSUN_resize/test", transform_test
        )
    elif ood_name == "isun":
        test_dataset = LocalImageDataset(
            "./experiments/data/iSUN/test", transform_test
        )
    elif ood_name == "cifar10":
        test_dataset = CIFAR10(
            "./experiments/data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif ood_name == "smooth":
        test_dataset = SmoothRandom(transform_test)
    else:
        raise ValueError(ood_name)

    return test_dataset


def get_test_dataloader(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    batch_size=16,
    num_workers=2,
    shuffle=True,
    ood=False,
    ood_name="vehicles",
    subsample=None,
    seed=42,
):
    # ood_categories = settings.OOD_CATEGORIES[ood_name]

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    torch.manual_seed(seed)

    if ood and ood_name in ["svhn", "cifar10", "lsun", "isun", "smooth"]:
        test_dataset = get_external_dataset(ood_name, transform_test)
    else:
        test_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        # test_dataset = CIFAR100_WITH_OOD(
        #     ood=ood, ood_classes=ood_classes(ood_categories), root='./data',
        #     train=False, download=True, transform=transform_test
        # )
        #
        # if ood:
        #     cifar_train_ood = CIFAR100_WITH_OOD(
        #         ood=True, ood_classes=ood_classes(ood_categories), root='./data',
        #         train=True, download=True, transform=transform_test
        #     )
        #     test_dataset = ConcatDataset((test_dataset, cifar_train_ood))

    if subsample:
        sampler = SequentialSampler(np.arange(subsample))
    else:
        sampler = None

    test_loader = DataLoader(
        test_dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
    )

    return test_loader


# Using it for iSUN and LSUn based on very specific file placementss
class LocalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = Path(root_dir)
        self.image_names = self._parse_names()
        self.transform = transform

    def _parse_names(self):
        files = [
            name
            for name in os.listdir(self.img_dir)
            if name.endswith("jpeg") or name.endswith("jpg")
        ]
        print(len(files), "images loaded")
        files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        return files

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.image_names[idx])
        lbl = 0  # dummy label, cause it's for OOD only
        return self.transform(img), lbl
