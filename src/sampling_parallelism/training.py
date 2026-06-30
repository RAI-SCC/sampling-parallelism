"""Training and evaluation loop for sampling-parallel VI BNNs.

:func:`train_model` is intentionally agnostic to the model, dataset and loss: it
receives a ready-built ``model``, ``loss_fn``, dataloaders and optimizer, plus
the per-process sample count and the parallelization mode. This makes it reusable
for new architectures and datasets — only the ``loss_fn`` and the dataloaders
change.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.optim.lr_scheduler import ConstantLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader
from torch_blue.vi import VIModule

from .distributed import average_across_processes
from .evaluation.metrics import compute_accuracy


@dataclass
class TrainingHistory:
    """Everything :func:`train_model` records over a training run.

    Attributes
    ----------
    loss_history : Dict[str, List]
        The loss tracker's ``data_fitting`` / ``prior_matching`` curves (suitable
        for :func:`plot_loss`).
    train_losses : Any
        Final training loss per epoch (world-averaged tensor when distributed,
        otherwise a list of floats).
    valid_accuracy : Any
        Validation accuracy per epoch (world-averaged when distributed).
    valid_acc_history : List[Dict[str, float]]
        The full validation metric dict for each epoch.
    brier_history : List[float]
        Validation Brier score per epoch (classification only).
    epoch_times : Any
        Wall-clock seconds per epoch (world-averaged when distributed).
    """

    loss_history: Dict[str, List] = field(default_factory=dict)
    train_losses: Any = None
    valid_accuracy: Any = None
    valid_acc_history: List[Dict[str, float]] = field(default_factory=list)
    brier_history: List[float] = field(default_factory=list)
    epoch_times: Any = None


class LinearWarmupScheduler:
    """Linearly ramp the learning rate from 0 to its target over ``warmup_steps``."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self) -> None:
        """Advance one step and update the learning rate during warmup."""
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.step_num / self.warmup_steps * self.optimizer.param_groups[0]["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr


def train_model(
    model: VIModule,
    loss_fn,
    num_epochs: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str = "classification",
    logging_interval: int = 50,
    scheduler: Optional[LRScheduler] = None,
    num_samples: int = 10,
    non_bayesian: bool = False,
    warmup: Optional[int] = None,
    rank: int = 0,
    parallelization: str = "local",
):
    """Train ``model`` and validate after every epoch.

    Parameters
    ----------
    model : VIModule
        The model to train (already wrapped in DDP if applicable).
    loss_fn : Callable
        The training objective, e.g. built by
        :func:`sampling_parallelism.losses.build_loss_fn`. For Bayesian losses it
        is expected to expose a ``.log`` dict of history lists.
    num_epochs : int
        Number of epochs to train.
    train_loader, valid_loader : DataLoader
        Training and validation dataloaders.
    optimizer : torch.optim.Optimizer
        The optimizer.
    device : torch.device
        Device to train on.
    task : str
        ``"classification"`` or ``"regression"`` (controls validation metrics).
    logging_interval : int
        Print the training loss every this many batches (rank 0 only).
    scheduler : Optional[LRScheduler]
        Optional learning-rate scheduler, stepped once per epoch.
    num_samples : int
        Weight samples drawn per forward pass on *this* process.
    non_bayesian : bool
        Whether the model is trained as a plain point estimate.
    warmup : Optional[int]
        If given, the number of epochs over which to linearly warm up the lr.
    rank : int
        Global rank; only rank 0 logs.
    parallelization : str
        Parallelization mode. ``"DDP"``/``"HYBRID"`` reshuffle via the sampler's
        ``set_epoch``; any non-``"local"`` mode averages reported metrics across
        processes.

    Returns
    -------
    TrainingHistory
        The loss curves, per-epoch train loss, validation accuracy, full
        validation metrics, Brier history and epoch timings.
    """
    valid_acc_history: List[Dict[str, float]] = []
    loss_log: List[float] = []

    # Optional linear warmup, optionally chained before the main scheduler.
    warmup_scheduler = None
    if warmup is not None:
        warmup_steps = warmup * len(train_loader)
        warmup_scheduler = LinearWarmupScheduler(optimizer, warmup_steps)
    if scheduler is not None and warmup is not None:
        scheduler = SequentialLR(
            optimizer, [ConstantLR(optimizer, 1.0), scheduler], [warmup]
        )

    start = time.time()
    last_elapsed = start

    epoch_times: List[float] = []
    train_losses: List[float] = []
    valid_accuracy: List[float] = []
    brier_history: List[float] = []

    for epoch in range(num_epochs):
        checkpoint = time.time()
        dataloading_time = forward_time = backward_time = 0.0
        optimizer_time = validation_time = 0.0

        # Reshuffle: distributed samplers need set_epoch; the local loader reseeds
        # its generator so the shuffle is reproducible per epoch.
        if parallelization in ("DDP", "HYBRID"):
            train_loader.sampler.set_epoch(epoch)
        else:
            train_loader.generator.manual_seed(epoch)

        model.train()
        model.return_log_probs = not non_bayesian

        epoch_time = 0.0
        loss = None
        for batch_idx, sample in enumerate(train_loader):
            # Move every tensor in the sample (features..., target) to the device.
            for i, tensor in enumerate(sample):
                sample[i] = tensor.to(device)
            dataloading_time += time.time() - checkpoint
            checkpoint = time.time()

            # Forward pass: the last element of `sample` is the target.
            if non_bayesian:
                logits = model(*sample[:-1], samples=1)
                logits = logits.squeeze(0)
            else:
                logits = model(*sample[:-1], samples=num_samples)

            loss = loss_fn(logits, sample[-1])

            print(logits.shape)
            print(logits.mean().item())
            print(logits.std().item())
            print(loss_fn.log["data_fit"][-1])
            print(loss_fn.log["kl"][-1])

            assert not torch.isinf(loss)
            assert not torch.isnan(loss)
            if non_bayesian:
                loss_log.append(loss.item())

            forward_time += time.time() - checkpoint
            checkpoint = time.time()

            optimizer.zero_grad()
            loss.backward()

            backward_time += time.time() - checkpoint
            checkpoint = time.time()



            optimizer.step()

            if warmup_scheduler is not None:
                warmup_scheduler.step()

            optimizer_time += time.time() - checkpoint
            checkpoint = time.time()
            epoch_time = time.time() - last_elapsed

            if rank == 0 and not batch_idx % logging_interval:
                print(
                    f"Epoch: {epoch + 1:03d}/{num_epochs:03d} "
                    f"| Batch {(batch_idx + 1):04d}/{len(train_loader):04d} "
                    f"| Loss: {loss.item():.4f}"
                )

        # Validation.
        model.eval()
        model.return_log_probs = False
        epoch_times.append(epoch_time)
        train_losses.append(loss.item())

        with torch.no_grad():
            valid_acc = compute_accuracy(
                model, valid_loader, device, samples=num_samples, task=task
            )
            if rank == 0:
                print(f"Epoch: {epoch + 1:03d}/{num_epochs:03d} | Validation:")
                for key in valid_acc:
                    print(f"\t{key}:\t{valid_acc[key]:.3f}")

            valid_accuracy.append(valid_acc["accuracy"])
            valid_acc_history.append(valid_acc)
            if "avg_brier_score" in valid_acc:
                brier_history.append(valid_acc["avg_brier_score"])

        if scheduler is not None:
            scheduler.step()

        validation_time += time.time() - checkpoint
        checkpoint = time.time()

        elapsed = time.time() - start
        last_elapsed = time.time()
        if rank == 0:
            print(f"Elapsed time: {elapsed:.2f}s")
            print(f"Elapsed time in last epoch: {epoch_time:.2f}s")
            print(f"Dataloading time: {dataloading_time:.2f}s")
            print(f"Forward time: {forward_time:.2f}s")
            print(f"Backward time: {backward_time:.2f}s")
            print(f"Optimizer time: {optimizer_time:.2f}s")
            print(f"Validation time: {validation_time:.2f}s")
            expected_remaining = elapsed / (epoch + 1) * (num_epochs - epoch - 1)
            print(f"Estimated remaining time: {expected_remaining:.2f}s")

    elapsed = time.time() - start
    if rank == 0:
        print(f"Total training time: {elapsed:.2f}s")

    # The Bayesian losses track their own data-fit / prior-matching curves; the
    # non-Bayesian path records the raw loss instead.
    if not non_bayesian:
        loss_history: Dict[str, List[Tensor]] = loss_fn.log
    else:
        loss_history = dict(data_fitting=loss_log, prior_matching=[torch.tensor(0.0)])

    # Average the per-epoch metrics across processes so every rank reports the
    # same world-level numbers.
    print(
        f"Rank {rank}: len(epoch_times)={len(epoch_times)}, "
        f"len(train_losses)={len(train_losses)}, "
        f"len(valid_accuracy)={len(valid_accuracy)}"
    )
    print(parallelization)
    if parallelization != "local":
        epoch_times = average_across_processes(epoch_times, torch.device("cuda"))
        train_losses = average_across_processes(train_losses, torch.device("cuda"))
        valid_accuracy = average_across_processes(valid_accuracy, torch.device("cuda"))

    return TrainingHistory(
        loss_history=loss_history,
        train_losses=train_losses,
        valid_accuracy=valid_accuracy,
        valid_acc_history=valid_acc_history,
        brier_history=brier_history,
        epoch_times=epoch_times,
    )


def eval_model(
    model: VIModule,
    test_loader: DataLoader,
    task: str,
    device: torch.device,
    samples: int,
) -> Dict[str, float]:
    """Evaluate ``model`` on the test set and print the metrics."""
    test_acc = compute_accuracy(model, test_loader, device, samples=samples, task=task)
    print("Test accuracy:")
    for key in test_acc:
        print(f"{key}: {test_acc[key]:.2f}")
    return test_acc
