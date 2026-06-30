"""Predictive-performance metrics for (Bayesian) classifiers."""

from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch_blue.vi import VIModule

from .scoring_rules import brier_score


def compute_accuracy(
    model: VIModule,
    data_loader: DataLoader,
    device: torch.device,
    samples: int = 10,
    task: str = "classification",
) -> Dict[str, float]:
    """Evaluate a model's predictive metrics on labelled data.

    Predictions are obtained by drawing ``samples`` weight samples per input and
    averaging the resulting class probabilities (Bayesian model averaging).

    Parameters
    ----------
    model : VIModule
        The model to evaluate.
    data_loader : DataLoader
        The data to evaluate on.
    device : torch.device
        The device to run inference on.
    samples : int
        Number of weight samples drawn per forward pass.
    task : str
        ``"classification"`` (the Brier score is only meaningful here) or
        ``"regression"``.

    Returns
    -------
    Dict[str, float]
        ``accuracy`` and ``err`` (percentages), ``nll`` (mean negative
        log-likelihood) and, for classification, ``avg_brier_score``.
    """
    with torch.no_grad():  # disable gradients to reduce memory
        correct_pred, num_examples = 0, 0
        brier_total = 0.0
        log_likelihood = 0.0
        model.eval()
        model.return_log_probs = False

        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            # Bayesian model average: mean of the per-sample class probabilities.
            logits = model(features, samples=samples)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            mean_probs = probs.mean(dim=0)  # [N, num_classes]

            log_likelihood -= (
                torch.gather(mean_probs, 1, targets.unsqueeze(-1)).log().sum().item()
            )
            prediction = mean_probs.argmax(dim=1)
            correct_pred += prediction.eq(targets).sum().item()
            num_examples += targets.size(0)

            if task == "classification":
                brier_total += brier_score(mean_probs, targets).item()

        metrics = dict(
            accuracy=correct_pred / num_examples * 100,
            err=(1 - correct_pred / num_examples) * 100,
            nll=log_likelihood / num_examples,
        )
        if task == "classification":
            metrics["avg_brier_score"] = brier_total / len(data_loader)
        return metrics
