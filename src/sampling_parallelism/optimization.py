"""Optimizer and learning-rate scheduler construction from a config."""

import argparse
from typing import Optional, Type

import torch
from torch.optim.lr_scheduler import LRScheduler

from .utils import find_in_module


def build_optimizer(
    parsed: argparse.Namespace, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """Build the optimizer named by ``parsed.optimizer_name``."""
    optimizer_class: Type[torch.optim.Optimizer] = find_in_module(
        parsed.optimizer_name, torch.optim
    )
    return optimizer_class(
        model.parameters(),
        lr=parsed.learning_rate,
        weight_decay=1e-4,
    )


def build_scheduler(
    parsed: argparse.Namespace, optimizer: torch.optim.Optimizer
) -> Optional[LRScheduler]:
    """Build the learning-rate scheduler named by ``parsed.scheduler_name``.

    Returns ``None`` when ``parsed.scheduler_name`` is ``None``.
    """
    if parsed.scheduler_name is None:
        return None

    scheduler_class: Type[LRScheduler] = find_in_module(
        parsed.scheduler_name, torch.optim.lr_scheduler
    )
    scheduler_specs = dict(
        StepLR=dict(step_size=parsed.step_size, gamma=parsed.gamma),
        MultiStepLR=dict(milestones=[80, 120], gamma=parsed.gamma),
        CosineAnnealingLR=dict(
            T_max=parsed.num_epochs - (1 if parsed.warmup is None else 1 + parsed.warmup)
        ),
    )
    return scheduler_class(optimizer=optimizer, **scheduler_specs[parsed.scheduler_name])
