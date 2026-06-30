import torch
import torch.nn.functional as F


def brier_score(probs, labels):
    """Mean Brier score for a batch of categorical predictions.

    Parameters
    ----------
    probs : Tensor
        Predicted class probabilities, shape ``[N, num_classes]``.
    labels : Tensor
        Integer class labels, shape ``[N]``.
    """
    # Infer the number of classes from the predictions so the metric works for
    # any classification task, not just 10-class datasets.
    num_classes = probs.size(-1)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))
    return score

def crps_mc(samples, targets):
    """
    samples: [S, N]  (MC samples from predictive distribution)
    targets: [N]
    """
    S = samples.size(0)

    # E|X - y|
    term1 = torch.mean(torch.abs(samples - targets.unsqueeze(0)), dim=0)

    # E|X - X'|
    # pairwise absolute differences
    diff = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1))
    term2 = 0.5 * torch.mean(diff, dim=(0,1))

    crps = term1 - term2
    return crps.mean()   # dataset average