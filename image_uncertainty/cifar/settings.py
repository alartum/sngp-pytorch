from datetime import datetime

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (
    0.5070751592371323,
    0.48654887331495095,
    0.4409178433670343,
)
CIFAR100_TRAIN_STD = (
    0.2673342858792401,
    0.2564384629170883,
    0.27615047132568404,
)

# directory to save weights file
CHECKPOINT_PATH = "checkpoint"

# total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]
# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
SHOW_FIGURES = (
    False  # if set to False, the figures would be saved to figures folder
)

# DATE_FORMAT = '%d_%B_%Y_%Hh_%Mm_%Ss'
DATE_FORMAT = "%Y_%m_%d_%Hh_%Mm_%Ss"
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# tensorboard log dir
LOG_DIR = "runs"

OOD_CATEGORIES = {
    "vehicles": ["vehicles 1", "vehicles 2"],
    "sea_animals": ["aquatic mammals", "fish"],
    "large_objects": [
        "large man-made outdoor things",
        "large natural outdoor scenes",
    ],
    "lsun": [],
    "svhn": [],
    "cifar10": [],
    "isun": [],
    "smooth": [],
}

#
# TODO: delete debug setting
EPOCH = 1
SAVE_EPOCH = 2
MILESTONES = [1, 2, 3]
SHOW_FIGURES = True
