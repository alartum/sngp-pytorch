import torch


def dempster_shafer_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    K = logits.shape[-1]
    return K / (K + logits.exp().sum(dim=-1))
