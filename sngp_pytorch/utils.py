from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch import functional as F
from torch.nn import Module
from torch.nn.utils.parametrizations import _SpectralNorm, parametrize


def mean_field_logits(
    logits: torch.Tensor,
    variances: torch.Tensor,
    mean_field_factor: float = 1.0,
) -> torch.Tensor:
    logits_scale = (1.0 + variances * mean_field_factor) ** 0.5
    if len(logits.shape) > 1:
        logits_scale = logits_scale[:, None]

    return logits / logits_scale


def dempster_shafer_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    K = logits.shape[-1]
    return K / (K + logits.exp().sum(dim=-1))


class _SpectralNormMultiplier(_SpectralNorm):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        norm_multiplier: float = 1.0,
    ) -> None:
        super().__init__(weight, n_power_iterations, dim, eps)
        self.norm_multiplier = norm_multiplier

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear,
            # but it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))

            if (self.norm_multiplier / sigma) < 1:
                return (self.norm_multiplier / sigma) * weight
            else:
                return weight


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
    norm_multiplier: float = 1.0,
) -> Module:

    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(
                module, name
            )
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(
        module,
        name,
        _SpectralNormMultiplier(
            weight, n_power_iterations, dim, eps, norm_multiplier
        ),
    )
    return module


def apply_spectral_norm(
    module,
    name: str = "model",
    norm_multiplier: float = 1.0,
    verbose: bool = False,
):
    to_replace = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    for child_name, child in module.named_children():
        apply_spectral_norm(child, child_name, norm_multiplier, verbose)
        child_ref = getattr(module, child_name)
        if isinstance(child_ref, to_replace):
            if verbose:
                print(f"  {name}.{child_name} ({type(child_ref)})")
            setattr(
                module,
                child_name,
                spectral_norm(child_ref, norm_multiplier=norm_multiplier),
            )


def get_last_fc(module):
    """Used to extract the last fully connected layer from the model.
    It can then be replaced with an arbitrary module via 'getattr'
    """
    state = {}

    def check_fc(module, state):
        for child_name, child in module.named_children():
            check_fc(child, state)
            child_ref = getattr(module, child_name)
            if isinstance(child_ref, nn.Linear):
                state["parent"] = module
                state["fc_name"] = child_name

    check_fc(module, state)

    return state["fc_name"], state["parent"]
