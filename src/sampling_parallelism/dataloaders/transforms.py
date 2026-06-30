"""Image transforms used by the vision experiments.

Besides the standard CIFAR augmentation pipeline, this provides
:class:`RNGCrop` and :class:`RNGHorizontalFlip`: deterministic variants of the
torchvision augmentations that draw from their own seeded generator. They are
useful when every process must apply the *same* augmentation to a shared batch
(as in sampling parallelism), rather than independent random ones.
"""

import torch
import torchvision.transforms.functional as F
from torchvision import transforms

#: Per-channel mean/std used to normalise CIFAR-style 3-channel images.
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


class RNGCrop:
    """Random crop driven by a private, once-seeded generator.

    Unlike :class:`torchvision.transforms.RandomCrop`, the crop coordinates come
    from an internal generator seeded a single time, so the sequence of crops is
    reproducible and identical across processes sharing the same seed.
    """

    def __init__(self, size: int, padding: int = 0, seed: int = 1234) -> None:
        self.size = size
        self.padding = padding
        self.g = torch.Generator()
        self.g.manual_seed(seed)  # seed ONCE

    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        w, h = img.size
        th, tw = self.size, self.size

        # Sample coordinates from our own generator.
        i = torch.randint(0, h - th + 1, (1,), generator=self.g).item()
        j = torch.randint(0, w - tw + 1, (1,), generator=self.g).item()

        return F.crop(img, i, j, th, tw)


class RNGHorizontalFlip:
    """Random horizontal flip driven by a private, once-seeded generator."""

    def __init__(self, p: float = 0.5, seed: int = 5678) -> None:
        self.p = p
        self.g = torch.Generator()
        self.g.manual_seed(seed)

    def __call__(self, img):
        if torch.rand(1, generator=self.g).item() < self.p:
            return F.hflip(img)
        return img


def build_augmentation_transform() -> transforms.Compose:
    """Training transform for CIFAR-style images: crop, flip, normalise."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Evaluation transform for CIFAR-style images: tensor + normalise only."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
