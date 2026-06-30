"""Construct models and infer dataset-dependent shapes from a config.

These helpers turn a parsed configuration (an ``argparse.Namespace``) into a
ready-to-train :class:`~torch_blue.vi.VIModule`. They are deliberately data- and
model-agnostic: new datasets are added to :func:`get_dataset_spec` and new
architectures are picked up automatically as long as they live in
:mod:`sampling_parallelism.models` and accept the standard VI arguments.
"""

import argparse
from typing import Any, Dict, Optional, Type

import torch
from torch import Tensor, nn
from torch_blue import vi
from torch_blue.vi import VIModule
from torch_blue.vi.distributions import Distribution, MeanFieldNormal, NonBayesian

from sampling_parallelism import models
from sampling_parallelism.utils import find_in_module


def get_dataset_spec(parsed: argparse.Namespace) -> Dict[str, Any]:
    """Return the input/output shapes a model needs for ``parsed.dataset``.

    For image datasets this derives ``in_features``/``out_features`` from the
    image size and class count. For time-series datasets it derives them from the
    historic/forecast windows (optionally folding in cyclic meta-features when
    ``parsed.recycle`` is set).
    """
    dataset_specs = dict(
        MNIST=dict(image_size=28, in_channels=1, num_classes=10),
        FashionMNIST=dict(image_size=28, in_channels=1, num_classes=10),
        CIFAR10=dict(image_size=32, in_channels=3, num_classes=10),
        ENTSOE=dict(
            country="de",
            historic_window=parsed.historic_window,
            forecast_window=parsed.forecast_window,
            meta_features=8,
            cycle_length=24,
        ),
        ETT=dict(
            historic_window=parsed.historic_window,
            forecast_window=parsed.forecast_window,
            meta_features=8,
            cycle_length=24,
        ),
        Traffic=dict(
            historic_window=parsed.historic_window,
            forecast_window=parsed.forecast_window,
            meta_features=8,
            cycle_length=24,
        ),
    )

    dataset_spec = dataset_specs[parsed.dataset]

    if "image_size" in dataset_spec:
        dataset_spec["in_features"] = (
            dataset_spec["image_size"] ** 2 * dataset_spec["in_channels"]
        )
        dataset_spec["out_features"] = dataset_spec["num_classes"]
    elif "historic_window" in dataset_spec:
        dataset_spec["in_features"] = dataset_spec["historic_window"]
        if parsed.recycle:
            dataset_spec["recycle"] = True
            dataset_spec["in_features"] += dataset_spec["meta_features"] * (
                dataset_spec["historic_window"] // dataset_spec["cycle_length"]
            )
        else:
            dataset_spec["meta_features"] = 0
        dataset_spec["out_features"] = dataset_spec["forecast_window"]
        dataset_spec["unflatten"] = (-1, dataset_spec["cycle_length"])
    return dataset_spec


def build_model(
    parsed: argparse.Namespace,
    dataset_spec: Dict[str, Any],
    device: torch.device,
    state_dict: Optional[Dict[str, Tensor]] = None,
) -> VIModule:
    """Instantiate the configured VI model on ``device``.

    The model class, activation and prior are resolved by name (see
    :func:`find_in_module`), so swapping architectures or priors only requires
    changing the corresponding string in the config.
    """
    prior_class: Type[Distribution] = find_in_module(parsed.prior_name, vi.distributions)

    if parsed.non_bayesian:
        prior = NonBayesian()
        vardist = NonBayesian()
        parsed.prior_name = "NonBayesian"
    else:
        prior = prior_class()
        vardist = MeanFieldNormal(std=0.01)

    model_class: Type[VIModule] = find_in_module(parsed.model_name, models)
    activation_class: Type[nn.Module] = find_in_module(parsed.activation, nn)
    model_args = dict(
        activation=activation_class(), pre_flatten=parsed.pre_flatten, **dataset_spec
    )
    model = model_class(
        **model_args,
        prior=prior,
        variational_distribution=vardist,
        kaiming_initialization=parsed.kaiming_init,
        prior_initialization=parsed.prior_init,
        rescale_prior=parsed.rescale_prior,
    )

    if parsed.recycle:
        # ReCycleWrapper wraps time-series models; import where it is defined.
        from sampling_parallelism.models import ReCycleWrapper

        model = ReCycleWrapper(model)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    model.to(device)
    return model
