"""Command-line / config-file configuration shared by the training scripts.

Configuration can be supplied three ways, in increasing precedence:

1. the built-in defaults below (algorithmic knobs only -- never paths),
2. a YAML config file passed with ``--config path/to/config.yaml``,
3. individual command-line flags, which override the config file.

Environment-specific values (``data_root``, ``output_dir``) have **no** default
and must be provided by the user's config or on the command line, so nothing
machine-specific is baked into the repository. See ``configs/example.yaml``.
"""

import argparse
from argparse import BooleanOptionalAction
from typing import Any, Dict, Optional, Tuple

import yaml

from .dataloaders.transforms import build_augmentation_transform, build_eval_transform
from .models.builder import get_dataset_spec

#: Datasets handled as regression (time-series forecasting) rather than classification.
REGRESSION_DATASETS = ("ENTSOE", "ETT", "Traffic")

#: Datasets that use the CIFAR-style augmentation/normalisation pipeline.
AUGMENTED_DATASETS = ("CIFAR10", "FashionMNIST")

#: Arguments that must be provided by the user (no machine-specific defaults ship
#: in the repo). ``parse_args`` errors out if any is still unset.
REQUIRED_ARGS = (
    "data_root",
    "output_dir",
    "global_sample_num",
    "global_batch_size",
)


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a flat dict (``{}`` if the file is empty)."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser shared by the training scripts.

    Only algorithmic defaults are set here. Paths and the global budget are left
    unset and must come from ``--config`` or explicit flags.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file providing defaults for any of the flags.",
    )

    # Environment-specific paths: no defaults, must be supplied by the user.
    parser.add_argument(
        "--data_root", type=str, default=None, help="Directory holding the datasets."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to write results to."
    )

    # Parallelization: the global budget to distribute across processes.
    parser.add_argument("--parallelization", type=str, default="local")
    parser.add_argument("--global_batch_size", type=int, default=None)
    parser.add_argument("--global_sample_num", type=int, default=None)
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=4,
        help="GPUs per node, used by the HYBRID strategy.",
    )

    # Model.
    parser.add_argument("--model_name", type=str, default="TinySimpleViT")
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--pre_flatten", type=int, default=None)
    parser.add_argument("--plot", action=BooleanOptionalAction, default=False)
    parser.add_argument("--non_bayesian", type=bool, default=False)
    parser.add_argument("--kaiming_init", type=bool, default=True)
    parser.add_argument("--prior_init", type=bool, default=False)
    parser.add_argument("--rescale_prior", type=bool, default=False)

    # Training schedule.
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--rand_seed", type=int, default=84)
    parser.add_argument("--fixed_samples", type=int, default=None)

    # Time-series windows.
    parser.add_argument("--historic_window", type=int, default=21 * 24)
    parser.add_argument("--forecast_window", type=int, default=7 * 24)
    parser.add_argument("--recycle", type=bool, default=False)

    # Dataset.
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--image_channels", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)

    # Optimization.
    parser.add_argument("--optimizer_name", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--heat", type=float, default=0.1)
    parser.add_argument("--scheduler_name", type=str, default=None)
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=80)

    parser.add_argument("--prior_name", type=str, default="MeanFieldNormal")

    parser.add_argument("--logging_interval", type=int, default=None)
    parser.add_argument("--run_nr", type=int, default=6)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse arguments, merging an optional ``--config`` YAML file under the CLI.

    Precedence is CLI > config file > built-in defaults. Required arguments
    (see :data:`REQUIRED_ARGS`) must be present after merging.
    """
    parser = build_arg_parser()

    # First pass: find --config without tripping on the other flags.
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config is not None:
        config = load_config_file(preliminary.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown = set(config) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown key(s) in config {preliminary.config}: {sorted(unknown)}"
            )
        # Config values become defaults; explicit CLI flags still override them.
        parser.set_defaults(**config)

    parsed = parser.parse_args(argv)

    missing = [name for name in REQUIRED_ARGS if getattr(parsed, name) is None]
    if missing:
        raise SystemExit(
            "Missing required configuration: "
            + ", ".join(missing)
            + ". Provide them via --config <file.yaml> or the matching --<name> flags "
            "(see configs/example.yaml)."
        )
    return parsed


def prepare_experiment(
    parsed: argparse.Namespace,
) -> Tuple[str, Optional[Any], Optional[Any], Dict[str, Any]]:
    """Derive task, transforms and dataset spec from the parsed config.

    Also fills in dataset-dependent defaults on ``parsed`` (``pre_flatten``,
    ``recycle``, ``step_size``).

    Returns
    -------
    task : str
        ``"classification"`` or ``"regression"``.
    train_transforms, tests_transforms : Optional[Callable]
        Image transforms (``None`` for non-image datasets).
    dataset_spec : Dict[str, Any]
        Input/output shape spec consumed by :func:`build_model`.
    """
    if parsed.step_size is None:
        parsed.step_size = parsed.num_epochs // 3

    if parsed.dataset in AUGMENTED_DATASETS:
        train_transforms = build_augmentation_transform()
        tests_transforms = build_eval_transform()
    else:
        train_transforms = None
        tests_transforms = None

    if parsed.model_name == "VITimeSeriesTransformer":
        parsed.recycle = True

    if parsed.dataset in REGRESSION_DATASETS:
        task = "regression"
        if parsed.pre_flatten is None:
            parsed.pre_flatten = 2
    else:
        task = "classification"
        if parsed.pre_flatten is None:
            parsed.pre_flatten = 3
        parsed.recycle = False

    dataset_spec = get_dataset_spec(parsed)
    return task, train_transforms, tests_transforms, dataset_spec
