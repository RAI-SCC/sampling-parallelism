"""Sampling parallelism for fast and efficient Bayesian neural network training.

Public API overview
--------------------
* :mod:`sampling_parallelism.parallelism` - split a global sample/batch budget
  into a per-process workload (``local`` / ``SP`` / ``DDP`` / ``HYBRID``).
* :mod:`sampling_parallelism.distributed` - process-group and SLURM plumbing.
* :mod:`sampling_parallelism.training` - the model-agnostic training loop.
* :mod:`sampling_parallelism.losses` - loss factory and the sampling-parallel
  objective.
* :mod:`sampling_parallelism.models`, :mod:`sampling_parallelism.dataloaders`,
  :mod:`sampling_parallelism.evaluation` - models, data and metrics.

See ``scripts/train.py`` for an end-to-end example.
"""

from .parallelism import (
    GPUS_PER_NODE,
    PARALLELIZATION_MODES,
    ParallelConfig,
    compute_local_workload,
    resolve_parallel_config,
)

__all__ = [
    "ParallelConfig",
    "compute_local_workload",
    "resolve_parallel_config",
    "GPUS_PER_NODE",
    "PARALLELIZATION_MODES",
]
