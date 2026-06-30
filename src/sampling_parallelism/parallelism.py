"""Workload splitting for sampling-parallel Bayesian neural network training.

This module is the heart of the method. A Bayesian forward pass draws several
weight *samples* and averages their predictions. Two quantities can therefore be
distributed across processes:

* the **mini-batch** (classical data parallelism), and
* the **number of weight samples** (sampling parallelism).

Given a *global* budget (the total number of weight samples and the total
mini-batch size the user wants), :func:`compute_local_workload` returns the
*per-process* workload for the chosen strategy. The strategies are:

``local``
    Single process. Each process handles the full batch and all samples.
``SP`` (Sampling Parallelism)
    The weight samples are split evenly across processes; every process sees the
    full mini-batch. Predictions are averaged across processes (see
    :mod:`sampling_parallelism.losses`). This is the method proposed in the paper.
``DDP`` (Distributed Data Parallelism)
    The mini-batch is split across processes; every process draws all samples.
    This is the classical baseline.
``HYBRID``
    Sampling parallelism *within* a node and data parallelism *across* nodes.
    Samples are split across the ``gpus_per_node`` GPUs of a node, while the
    mini-batch is split across nodes.

The split logic is kept as a pure function (no torch, no environment access) so
it is trivial to read, unit-test, and reuse for other architectures or datasets.
:func:`resolve_parallel_config` wraps it with the SLURM/process-group plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Default number of GPUs per compute node, used by the ``HYBRID`` strategy.
GPUS_PER_NODE = 4

#: The parallelization strategies understood by this module.
PARALLELIZATION_MODES = ("local", "SP", "DDP", "HYBRID")


@dataclass
class ParallelConfig:
    """Fully resolved parallelization settings for a single process.

    Attributes
    ----------
    mode : str
        One of :data:`PARALLELIZATION_MODES`.
    rank : int
        Global rank of this process (0 for ``local``).
    world_size : int
        Total number of processes (1 for ``local``).
    local_samples : int
        Number of weight samples this process draws per forward pass.
    local_batch_size : int
        Mini-batch size this process loads.
    distributed : bool
        Whether the dataloader must shard the data with a ``DistributedSampler``
        (true only for ``DDP``).
    hybrid : bool
        Whether the dataloader must shard the data per node (true only for
        ``HYBRID``).
    """

    mode: str
    rank: int
    world_size: int
    local_samples: int
    local_batch_size: int
    distributed: bool
    hybrid: bool

    @property
    def is_main_process(self) -> bool:
        """Whether this process should perform rank-0-only work (logging, saving)."""
        return self.rank == 0


def compute_local_workload(
    mode: str,
    global_sample_num: int,
    global_batch_size: int,
    world_size: int,
    gpus_per_node: int = GPUS_PER_NODE,
) -> tuple[int, int, bool, bool]:
    """Split a global sample/batch budget into a per-process workload.

    Parameters
    ----------
    mode : str
        Parallelization strategy, one of :data:`PARALLELIZATION_MODES`.
    global_sample_num : int
        Total number of weight samples to draw across all processes.
    global_batch_size : int
        Total mini-batch size across all processes.
    world_size : int
        Total number of processes.
    gpus_per_node : int
        GPUs per node, only used by the ``HYBRID`` strategy.

    Returns
    -------
    local_samples : int
        Weight samples drawn per process.
    local_batch_size : int
        Mini-batch size loaded per process.
    distributed : bool
        Whether to shard data across all processes (``DistributedSampler``).
    hybrid : bool
        Whether to shard data per node (``HybridDistributedSampler``).

    Raises
    ------
    ValueError
        If ``mode`` is not a recognised parallelization strategy.
    """
    if mode == "local":
        return global_sample_num, global_batch_size, False, False
    if mode == "SP":
        # Split the weight samples; keep the full mini-batch on every process.
        return global_sample_num // world_size, global_batch_size, False, False
    if mode == "DDP":
        # Split the mini-batch; every process draws all weight samples.
        return global_sample_num, global_batch_size // world_size, True, False
    if mode == "HYBRID":
        # Split samples within a node, split the batch across nodes.
        num_nodes = world_size // gpus_per_node
        return (
            global_sample_num // gpus_per_node,
            global_batch_size // num_nodes,
            False,
            True,
        )
    raise ValueError(
        f"Unknown parallelization mode {mode!r}; expected one of {PARALLELIZATION_MODES}."
    )


def resolve_parallel_config(
    mode: str,
    global_sample_num: int,
    global_batch_size: int,
    gpus_per_node: int = GPUS_PER_NODE,
) -> ParallelConfig:
    """Initialise the process group (if needed) and resolve the local workload.

    For every non-``local`` mode this reads the global rank and world size from
    the SLURM environment and initialises the default process group. The actual
    sample/batch split is delegated to :func:`compute_local_workload`.

    Parameters
    ----------
    mode : str
        Parallelization strategy, one of :data:`PARALLELIZATION_MODES`.
    global_sample_num : int
        Total number of weight samples across all processes.
    global_batch_size : int
        Total mini-batch size across all processes.
    gpus_per_node : int
        GPUs per node, only used by the ``HYBRID`` strategy.

    Returns
    -------
    ParallelConfig
        The resolved per-process configuration.
    """
    if mode == "local":
        rank, world_size = 0, 1
    else:
        # Imported lazily so the pure split logic above stays torch-free.
        from .distributed import get_slurm_rank_world, setup_distributed

        rank, world_size = get_slurm_rank_world()
        setup_distributed(rank, world_size)

    local_samples, local_batch_size, distributed, hybrid = compute_local_workload(
        mode, global_sample_num, global_batch_size, world_size, gpus_per_node
    )

    return ParallelConfig(
        mode=mode,
        rank=rank,
        world_size=world_size,
        local_samples=local_samples,
        local_batch_size=local_batch_size,
        distributed=distributed,
        hybrid=hybrid,
    )
