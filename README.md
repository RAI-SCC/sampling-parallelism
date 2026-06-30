# Sampling Parallelism for Fast and Efficient Bayesian Learning

Official repository for the paper:

> *Sampling Parallelism for Fast and Efficient Bayesian Learning*

A Bayesian forward pass draws several **weight samples** and averages their
predictions. This lets us distribute training in two ways by spliting the **number of weight samples** across processes (*sampling parallelism*).


## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for the test/lint tooling
```

## Configuration

Runs are configured from a YAML file you provide. Copy [`configs/example.yaml`](configs/example.yaml), set `data_root`
and `output_dir`, and pass it with `--config`. Any field can be overridden on the
command line (CLI > config file > built-in defaults).

## Quick start

Single-process (debug) run:

```bash
python scripts/train.py --config configs/example.yaml --parallelization local
```

Distributed runs are launched under SLURM

```bash
srun python scripts/train.py --config configs/example.yaml --parallelization SP     # split samples
srun python scripts/train.py --config configs/example.yaml --parallelization DDP    # split batch
srun python scripts/train.py --config configs/example.yaml --parallelization HYBRID # samples/node, batch/node
```

## Project layout

The training script is a thin entrypoint, everything reusable lives in the
installable `sampling_parallelism` package:

```
src/sampling_parallelism/
  parallelism.py     # core: split a global sample/batch budget per process
  distributed.py     # process-group setup/cleanup, SLURM rank, metric averaging
  training.py        # model-agnostic train_model / eval_model + TrainingHistory
  optimization.py    # build_optimizer / build_scheduler
  config.py          # YAML/CLI parsing (parse_args) + dataset/task wiring
  losses/            # build_loss_fn factory + up_loss.UP_AllGaussLoss
  models/            # VI model zoo + build_model / get_dataset_spec
  dataloaders/       # vision loaders, samplers, transforms
  evaluation/        # metrics (accuracy) + scoring_rules (Brier, CRPS)
  plotting.py        # loss-curve plotting
configs/             # example.yaml -- copy and edit for your environment
scripts/             # train.py
```

### The core idea in one function

`parallelism.compute_local_workload` maps the global budget to the per-process
workload for each strategy (it is pure Python, no torch — see
`tests/test_parallelism.py`):

| mode     | local samples              | local batch size            |
|----------|----------------------------|-----------------------------|
| `local`  | `global_sample_num`        | `global_batch_size`         |
| `SP`     | `global_sample_num / W`    | `global_batch_size`         |
| `DDP`    | `global_sample_num`        | `global_batch_size / W`     |
| `HYBRID` | `global_sample_num / G`    | `global_batch_size / nodes` |

where `W` is the world size and `G` is the number of GPUs per node.

## Applying this to your own model or dataset

* **New architecture:** add a `VIModule` under `sampling_parallelism/models/`
  and pass `--model_name YourModel`. `train_model` needs no changes.
* **New dataset:** add its shapes to `get_dataset_spec` and, if needed, its
  task/transforms tables in `config.prepare_experiment`.
* **Reuse directly:** `resolve_parallel_config(...)` + `train_model(...)` are the
  two calls you need; see `scripts/train.py` for the full flow.

