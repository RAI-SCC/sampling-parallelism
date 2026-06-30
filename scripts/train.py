"""Train a VI Bayesian neural network with configurable sampling parallelism.

This is a thin orchestration script. The reusable machinery lives in the
:mod:`sampling_parallelism` package:

* workload splitting -> :mod:`sampling_parallelism.parallelism`
* distributed plumbing -> :mod:`sampling_parallelism.distributed`
* the training loop    -> :mod:`sampling_parallelism.training`
* model / data / loss  -> ``models``, ``dataloaders``, ``losses``

All paths and the global budget come from the user's config; nothing is
hardcoded. Run, for example::

    python scripts/train.py --config configs/example.yaml

or override individual values on the command line::

    python scripts/train.py --config configs/example.yaml --parallelization SP

Use ``--parallelization SP|DDP|HYBRID`` under a SLURM launcher for the
distributed strategies.
"""

import os
from math import ceil, log10

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from sampling_parallelism.config import parse_args, prepare_experiment
from sampling_parallelism.dataloaders.vision import get_dataloaders_torchvision
from sampling_parallelism.distributed import cleanup_distributed
from sampling_parallelism.losses import build_loss_fn
from sampling_parallelism.models.builder import build_model
from sampling_parallelism.optimization import build_optimizer, build_scheduler
from sampling_parallelism.parallelism import resolve_parallel_config
from sampling_parallelism.plotting import plot_loss
from sampling_parallelism.training import eval_model, train_model
from sampling_parallelism.utils import get_save_name, set_seeds


def main() -> None:
    if torch.cuda.is_available():
        rank = int(os.getenv("SLURM_PROCID"))  # Get individual process ID.
        world_size = int(os.getenv("SLURM_NTASKS"))  # Total number of processes.
        slurm_localid = int(os.getenv("SLURM_LOCALID", rank))


        device_id = slurm_localid if torch.cuda.device_count() != 1 else 0
        device = torch.device("cuda", device_id)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print("Device:", device)

    parsed = parse_args()
    parsed.rand_seed += parsed.run_nr
    set_seeds(parsed.rand_seed)

    # Resolve the dataset/task wiring and the per-process workload split.
    task, train_transforms, tests_transforms, dataset_spec = prepare_experiment(parsed)
    pconf = resolve_parallel_config(
        parsed.parallelization,
        parsed.global_sample_num,
        parsed.global_batch_size,
        gpus_per_node=parsed.gpus_per_node,
    )

    # Data.
    data_dir = os.path.join(parsed.data_root, parsed.dataset.lower())
    train_loader, valid_loader, test_loader = get_dataloaders_torchvision(
        dataset_name=parsed.dataset,
        batch_size=pconf.local_batch_size,
        data_root=data_dir,
        train_transforms=train_transforms,
        tests_transforms=tests_transforms,
        seed=parsed.rand_seed,
        fixed_samples=parsed.fixed_samples,
        distributed=pconf.distributed,
        hybrid_dist=pconf.hybrid,
        **dataset_spec,
    )

    logging_interval = parsed.logging_interval or 10 ** (ceil(log10(len(train_loader))) - 1)
    if pconf.is_main_process:
        print(f"Finished loading {parsed.dataset} with {len(train_loader.dataset)} samples.")
        for sample in train_loader:
            print("Data batch dimensions:", sample[0].shape)
            print("Label dimensions:", sample[-1].shape)
            break

    # Model.

    model = build_model(parsed, dataset_spec, device=device)
    print(next(model.parameters()).device)
    model.return_log_probs = not parsed.non_bayesian
    if pconf.mode != "local":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    

    # Loss, optimizer, scheduler.
    loss_fn = build_loss_fn(
        task=task,
        dataset_size=len(train_loader.dataset),
        heat=parsed.heat,
        non_bayesian=parsed.non_bayesian,
    )
    optimizer = build_optimizer(parsed, model)
    scheduler = build_scheduler(parsed, optimizer)

    # Train. Re-seed per rank so each process draws independent weight samples.
    set_seeds(parsed.rand_seed + pconf.rank)
    history = train_model(
        model=model,
        loss_fn=loss_fn,
        num_epochs=parsed.num_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        task=task,
        scheduler=scheduler,
        logging_interval=logging_interval,
        num_samples=pconf.local_samples,
        non_bayesian=parsed.non_bayesian,
        warmup=parsed.warmup,
        rank=pconf.rank,
        parallelization=pconf.mode,
    )
    if pconf.is_main_process:
        print(history.brier_history)

    # Test (re-seed so testing alone is reproducible).
    set_seeds(parsed.rand_seed + pconf.rank)
    test_accuracy = eval_model(
        model=model,
        test_loader=test_loader,
        task=task,
        device=device,
        samples=pconf.local_samples,
    )

    if parsed.plot:
        plot_loss(history.loss_history)

    # Save results (rank 0 only).
    save_name = get_save_name(parsed, pconf.world_size, parsed.output_dir)
    if pconf.is_main_process:
        os.makedirs(parsed.output_dir, exist_ok=True)
        print(f"Saving to {save_name}")
        torch.save(
            dict(
                parsed=parsed,
                state_dict=model.state_dict(),
                times=history.epoch_times,
                loss_history=history.loss_history,
                valid_acc_history=history.valid_acc_history,
                test_accuracy=test_accuracy,
            ),
            save_name,
        )

    if pconf.mode != "local":
        cleanup_distributed()


if __name__ == "__main__":
    main()
