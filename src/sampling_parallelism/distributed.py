"""Thin helpers around ``torch.distributed`` and the SLURM launcher.

These wrap the boilerplate needed to launch and tear down a process group and to
reduce metrics across processes, so the training scripts stay readable.
"""

from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.distributed as dist


def get_slurm_rank_world() -> tuple[int, int]:
    """Read the global rank and world size from the SLURM environment.

    Returns
    -------
    rank : int
        The global rank of this process (``SLURM_PROCID``).
    world_size : int
        The total number of processes (``SLURM_NTASKS``).
    """
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    return rank, world_size


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialise the default process group.

    Parameters
    ----------
    rank : int
        The global rank of this process.
    world_size : int
        The total number of processes.
    backend : str
        The collective-communication backend (``"nccl"`` for CUDA GPUs).
    """
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    """Destroy the default process group."""
    dist.destroy_process_group()


def average_across_processes(
    values: Sequence[float], device: torch.device
) -> torch.Tensor:
    """Average a sequence of scalars across all processes.

    Each process contributes ``values``; the returned tensor holds the
    element-wise mean over the whole world. Used to aggregate per-epoch metrics
    (timings, losses, accuracies) for reporting.

    Parameters
    ----------
    values : Sequence[float]
        The per-process values to average (e.g. one entry per epoch).
    device : torch.device
        The device to run the all-reduce on (must match the backend, i.e. CUDA
        for ``nccl``).

    Returns
    -------
    torch.Tensor
        The world-averaged values.
    """
    reduced = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced
