"""Plotting helpers for inspecting training runs."""

from typing import Dict, List

import torch
from matplotlib import pyplot as plt
from torch import Tensor


def plot_loss(loss_history: Dict[str, List[Tensor]]) -> None:
    """Plot the data-fitting (NLL) and prior-matching (KL) loss curves."""
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(torch.tensor(loss_history["data_fitting"]), label="nll")
    axes[0].set_ylabel("nll")
    axes[1].plot(torch.tensor(loss_history["prior_matching"]), label="kl_term")
    axes[1].set_ylabel("kl_term")
    axes[1].set_yscale("log")
    fig.show()
