from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_blue.vi as vi
from torch_blue.vi.distributions import Categorical


class UP_AllGaussLoss(nn.Module):
    def __init__(self, model, dataset_size, process_group, heat=1.0, track=False):
        super().__init__()

        self.pg = process_group

        #for module in model.modules():
        #    if (
        #        not hasattr(module, "random_variables")
        #    ) or module.random_variables is None:
        #        continue

        #    for var, prior in zip(module.random_variables, module.prior.values()):
        #        print(var)
        #        print(prior)
        #print(model.variational_distribution)


        self.kl_loss = vi.AnalyticalKullbackLeiblerLoss(
            model,
            Categorical(),
            divergence_type=vi.NormalNormalDivergence(),
            dataset_size=dataset_size,
            heat=heat,
            track=track,
        )

        self.dataset_size = dataset_size

        self.aggregate_statistics = GaussianAggregation.apply

    def data_fitting(self, mean, target):
        data_fit = (
            -self.dataset_size
            * self.kl_loss.predictive_distribution.log_prob_from_parameters(
                target, mean
            )
            .mean(dim=0)
            .sum()
        )

        return data_fit

    def prior_matching(self) -> torch.Tensor:
        total_kl = None
        for module in self.kl_loss.model.modules():
            if (
                not hasattr(module, "random_variables")
            ) or module.random_variables is None:
                continue

            for var, prior in zip(module.random_variables, module.prior.values()):
                if prior is None:
                    continue
                prior_params = []
                for param in prior.distribution_parameters:
                    prior_params.append(getattr(prior, param))
                variational_params = module.get_variational_parameters(var)

                variable_kl = self.kl_loss.kl_module(prior_params, variational_params)
                if total_kl is None:
                    total_kl = variable_kl
                else:
                    total_kl = total_kl + variable_kl

        return total_kl

    def forward(self, model_output, target):
        prior_matching = self.kl_loss.heat * self.prior_matching()

        pred_class_prob = self.kl_loss.predictive_distribution.predictive_parameters_from_samples(model_output)
        
        mean = self.aggregate_statistics(pred_class_prob, self.pg)

        data_fit = self.data_fitting(mean, target)

        if self.kl_loss._track and self.kl_loss.log is not None:
            self.kl_loss.log["data_fitting"].append(data_fit.item())
            self.kl_loss.log["prior_matching"].append(prior_matching.item())

        return data_fit + prior_matching

    @property
    def log(self):
        return self.kl_loss.log


class GaussianAggregation(torch.autograd.Function):
    @staticmethod
    def forward(samples, process_group):

        local_sum_scaled = (samples / process_group.size()).contiguous()
        mu_handle = dist.all_reduce(
            local_sum_scaled, op=dist.ReduceOp.SUM, group=process_group, async_op=True
        )

        mu_handle.wait()
        mean = local_sum_scaled

        return mean

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        samples, process_group = inputs
        mean = output

        ctx.process_group = process_group
        ctx.save_for_backward(samples, mean)

    @staticmethod
    def backward(ctx: Any, grad_mean) -> Any:
        process_group = ctx.process_group
        samples, mean = ctx.saved_tensors

        N_glob = process_group.size()


        grad_samples = (grad_mean / N_glob)

        return grad_samples, None