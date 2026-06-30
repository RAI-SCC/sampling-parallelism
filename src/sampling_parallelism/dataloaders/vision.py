from typing import Any, Callable, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
import math
from .samplers import HybridDistributedSampler

from sampling_parallelism.utils import find_in_module


def get_dataloaders_torchvision(
    dataset_name: str,
    batch_size: int,
    data_root: str = "data",
    validation_fraction: float = 0.1,
    train_transforms: Optional[Callable] = None,
    tests_transforms: Optional[Callable] = None,
    seed: int = 123,
    distributed: bool = False,
    fixed_samples: Optional[int] = None,
    hybrid_dist: bool = False,
    #parallelization_type: str = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get dataloaders for training, validation, and testing on the FashionMNIST dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    batch_size : int
        The mini-batch size.
    data_root : str
        The path to the dataset.
    validation_fraction : float
        The fraction of the original training data used for validation.
    train_transforms : Callable
        The transform applied to the training data.
    tests_transforms : Callable
        The transform applied to the validation/testing data (inference).
    seed : int
        The seed for the validation-train split.
    distributed : bool
        Whether to use distributed data loading.
    fixed_samples : Optional[int]
        If not None, fix the number of samples used for training. Else use all samples.
    parallelization_type: str
        Which type of parallelization is to be used

    Returns
    -------
    DataLoader
        The training dataloader.
    DataLoader
        The validation dataloader.
    DataLoader
        The testing dataloader.
    """
    dataset: Type[VisionDataset] = find_in_module(dataset_name, datasets)

    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if tests_transforms is None:
        tests_transforms = transforms.ToTensor()

    train_valid_dataset = None
    tests_dataset = None

    if (not distributed) and (not hybrid_dist):
        train_valid_dataset = dataset(
            root=data_root, train=True, transform=train_transforms, download=True
        )
        tests_dataset = dataset(root=data_root, train=False, transform=tests_transforms)

    else:
        # Only root shall download dataset if data is not already there.
        if dist.get_rank() == 0:
            train_valid_dataset = dataset(
                root=data_root, train=True, transform=train_transforms, download=True
            )
            tests_dataset = dataset(
                root=data_root, train=False, transform=tests_transforms
            )
        dist.barrier()
        # Other ranks must not download dataset at the same time in parallel.
        if dist.get_rank() != 0:
            train_valid_dataset = dataset(
                root=data_root, train=True, transform=train_transforms
            )
            tests_dataset = dataset(
                root=data_root, train=False, transform=tests_transforms
            )

    # Split into training and validation dataset according to specified validation fraction.
    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(
        train_valid_dataset,
        [1.0 - validation_fraction, validation_fraction],
        generator=generator,
    )

    # Reduce training samples, if requested
    if fixed_samples is not None:
        train_dataset, _ = random_split(
            train_dataset,
            [fixed_samples, len(train_dataset) - fixed_samples],
            generator=generator,
        )

    if (not distributed) and (not hybrid_dist):
        train_sampler = None
        valid_sampler = None
    elif distributed and not hybrid_dist:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
    else:
        train_sampler = HybridDistributedSampler(
            dataset = train_dataset,
            num_nodes=math.ceil(dist.get_world_size()/4),
            node_rank=math.floor(dist.get_rank()/4),
            shuffle=True,
            drop_last=True,
        )
        valid_sampler = HybridDistributedSampler(
            valid_dataset,
            num_nodes=math.ceil(dist.get_world_size()/4),
            node_rank=math.floor(dist.get_rank()/4),
            shuffle=True,
            drop_last=True,
        )

    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, sampler=valid_sampler
    )
    if (not hybrid_dist) and (not distributed):
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            sampler=train_sampler,
            generator=g,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            sampler=train_sampler,
        )

    tests_loader = DataLoader(
        dataset=tests_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, tests_loader
