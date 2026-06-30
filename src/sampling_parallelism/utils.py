"""Small, dependency-light helpers shared across scripts."""

import os
import random
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar, Union
import numpy as np

import torch

if TYPE_CHECKING:  # avoid importing argparse at runtime just for the annotation
    import argparse

T = TypeVar("T")


def find_in_module(obj: Union[str, T], module: ModuleType) -> T:
    """Return ``obj`` directly, or look it up by name in ``module``.

    Lets configuration refer to classes/functions by string name (e.g.
    ``"Adam"`` resolved against :mod:`torch.optim`) while still accepting the
    object itself.
    """
    if isinstance(obj, str):
        return getattr(module, obj)
    else:
        return obj


def set_seeds(seed_value: int = 42) -> None:
    """Seed every relevant RNG for reproducibility.

    Seeds Python's :mod:`random`, the PyTorch CPU and CUDA generators, and the
    hash seed, and forces deterministic cuDNN kernels.
    """
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # PyTorch CPU RNG
    torch.cuda.manual_seed(seed_value)  # PyTorch current-GPU RNG
    torch.cuda.manual_seed_all(seed_value)  # PyTorch all-GPU RNG
    torch.backends.cudnn.deterministic = True  # use deterministic algorithms
    torch.backends.cudnn.benchmark = False  # disable autotuner for determinism
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash seed
    np.random.seed(seed_value)


def get_save_name(
    parsed: "argparse.Namespace",
    gpus: int,
    output_dir: str,
    suffix: str = ".pt",
) -> str:
    """Build the output path encoding the run's configuration.

    The filename records the model, parallelization strategy, GPU count, global
    sample/batch budget, prior, dataset, and run number, so that results from
    different configurations do not collide. It is placed inside ``output_dir``.
    """
    if parsed.non_bayesian:
        filename = f"{parsed.model_name}NB{parsed.dataset}_{parsed.run_nr}{suffix}"
    else:
        filename = (
            f"{parsed.model_name}_{parsed.parallelization}{gpus}"
            f"_{parsed.global_sample_num}_{parsed.global_batch_size}"
            f"{parsed.prior_name}{parsed.dataset}_{parsed.run_nr}{suffix}"
        )
    return os.path.join(output_dir, filename)
