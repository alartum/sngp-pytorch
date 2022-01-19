import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Cos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor):
        return torch.cos(X)


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        backbone: nn.Module = nn.Identity(),
        n_inducing: int = 1024,
        momentum: float = 0.9,
        ridge_penalty: float = 1e-6,
        activation: nn.Module = Cos(),
        rff_type: str = "orf",
        verbose: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_inducing = n_inducing
        self.momentum = momentum
        self.ridge_penalty = ridge_penalty
        self.verbose = verbose
        self.backbone = backbone

        # Random Fourier features (RFF) layer
        projection = nn.Linear(in_features, n_inducing)
        projection.weight.requires_grad_(False)
        projection.bias.requires_grad_(False)

        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L96
        if rff_type == "rff":
            nn.init.normal_(projection.weight, gain=0.05)
        elif rff_type == "orf":
            nn.init.orthogonal_(projection.weight, gain=0.05)
        else:
            raise ValueError(f"Unsupported `rff_type`: {rff_type}")
        nn.init.uniform_(projection.bias, 0, 2 * math.pi)

        self.rff = nn.Sequential(
            OrderedDict(
                [
                    ("backbone", backbone),
                    ("projection", projection),
                    ("activation", activation),
                ]
            )
        )

        # Weights for RFF
        self.weight = nn.Linear(n_inducing, out_features, bias=False)
        # Should be normally distributed a priori
        nn.init.xavier_normal_(self.weight.weight, gain=math.sqrt(5))

        self.pipeline = nn.Sequential(self.rff, self.weight)

        # RFF precision and covariance matrices
        self.is_fitted = False
        self.covariance = Parameter(
            1 / self.ridge_penalty * torch.eye(self.n_inducing),
            requires_grad=False,
        )
        # Ridge penalty is used to stabilize the inverse computation
        self.precision_initial = self.ridge_penalty * torch.eye(
            self.n_inducing, requires_grad=False
        )
        self.precision = Parameter(
            self.precision_initial,
            requires_grad=False,
        )

    def forward(
        self,
        X: torch.Tensor,
        with_variance: bool = False,
        update_precision: bool = False,
    ):
        features = self.rff(X)
        if update_precision:
            self.update_precision_(features)

        logits = self.weight(features)
        if not with_variance:
            return logits
        else:
            if not self.is_fitted:
                raise ValueError(
                    "`compute_covariance` should be called before setting "
                    "`with_variance` to True"
                )
            with torch.no_grad():
                variances = torch.bmm(
                    features[:, None, :],
                    (features @ self.covariance)[:, :, None],
                ).reshape(-1)

            return logits, variances

    def reset_precision(self):
        self.precision[...] = self.precision_initial.detach()

    def update_precision_(self, features: torch.Tensor):
        with torch.no_grad():
            if self.momentum < 0:
                # Use this to compute the precision matrix for the whole
                # dataset at once
                self.precision[...] = self.precision + features.T @ features
            else:
                self.precision[...] = (
                    self.momentum * self.precision
                    + (1 - self.momentum) * features.T @ features
                )

    def update_precision(self, X: torch.Tensor):
        with torch.no_grad():
            features = self.rff(X)
            self.update_precision_(features)

    def update_covariance(self):
        if not self.is_fitted:
            self.covariance[...] = (
                self.ridge_penalty * self.precision.cholesky_inverse()
            )
            self.is_fitted = True

    def reset_covariance(self):
        self.is_fitted = False
        self.covariance.zero_()
