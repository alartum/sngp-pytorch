import random

import numpy as np
import torch
from skimage.filters import gaussian
from torch.utils.data import Dataset

SEED = 42


class SmoothRandom(Dataset):
    def __init__(
        self,
        transforms,
        image_size=(32, 32, 3),
        samples=10_000,
        radius=(1, 2.5),
    ):
        """
        image_size - the size of the images
        samples - number of generated images
        radius - (min, max) of radius for gaussian smoothing
        """
        self.images = self._generate(image_size, samples, radius)
        self.transforms = transforms

    def _generate(self, image_size, samples, smooth_r):
        random.seed(SEED)
        np.random.seed(SEED)

        noise_images = np.random.random((samples, *image_size))
        radiuses = (smooth_r[1] - smooth_r[0]) * np.random.random(
            samples
        ) + smooth_r[0]
        smoothed = np.array(
            [
                gaussian(img, r, multichannel=3)
                for img, r in zip(noise_images, radiuses)
            ]
        )
        return smoothed.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fake_label = 0
        return self.transforms(self.images[idx]), fake_label
