"""Loss functions and a factory for the standard training objectives."""

import torch
from torch_blue import vi
from torch_blue.vi.distributions import Categorical, MeanFieldNormal

from .up_loss import GaussianAggregation, UP_AllGaussLoss

__all__ = [
    "build_loss_fn",
    "UP_AllGaussLoss",
    "GaussianAggregation",
]


def build_loss_fn(
    task: str,
    dataset_size: int,
    heat: float = 1.0,
    non_bayesian: bool = False,
    track: bool = True,
):
    """Build the standard training loss for a task.

    For Bayesian models this is the (tracked) ELBO via
    :class:`~torch_blue.vi.KullbackLeiblerLoss`, with the predictive distribution
    chosen from the task. For non-Bayesian models it falls back to the usual
    point-estimate losses. The specialised sampling-parallel objective lives in
    :class:`UP_AllGaussLoss`.

    Parameters
    ----------
    task : str
        ``"classification"`` or ``"regression"``.
    dataset_size : int
        Number of training examples, used to scale the data-fit term.
    heat : float
        Weight of the prior-matching (KL) term.
    non_bayesian : bool
        If ``True``, return a plain point-estimate loss instead of the ELBO.
    track : bool
        Whether the KL loss should record its data-fit / prior-matching history.

    Returns
    -------
    Callable
        A loss callable taking ``(model_output, target)``.
    """
    if non_bayesian:
        if task == "regression":
            return torch.nn.MSELoss()
        if task == "classification":
            return torch.nn.CrossEntropyLoss()
        raise ValueError(f'Unknown task "{task}"')

    if task == "classification":
        predictive_distribution = Categorical()
    elif task == "regression":
        predictive_distribution = MeanFieldNormal()
    else:
        raise ValueError(f'Unknown task "{task}"')

    return vi.KullbackLeiblerLoss(
        predictive_distribution,
        dataset_size=dataset_size,
        heat=heat,
        track=track,
    )
