import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm.auto import tqdm

from .utils import mean_field_logits


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
        batch_size: Optional[int] = 1000,
        ridge_penalty: float = 1e-6,
        activation: nn.Module = Cos(),
        verbose: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_inducing = n_inducing
        self.momentum = momentum
        self.batch_size = batch_size
        self.ridge_penalty = ridge_penalty
        self.verbose = verbose
        self.backbone = backbone

        # Random Fourier features (RFF) layer
        projection = nn.Linear(in_features, n_inducing)
        projection.weight.requires_grad_(False)
        projection.bias.requires_grad_(False)

        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L96
        nn.init.kaiming_normal_(projection.weight, a=math.sqrt(5))
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
        nn.init.kaiming_normal_(self.weight.weight, a=math.sqrt(5))

        self.pipeline = nn.Sequential(self.rff, self.weight)

        # RFF covariance matrix
        self.covariance = Parameter(
            torch.zeros(self.n_inducing, self.n_inducing), requires_grad=False
        )
        self.is_fitted = False

    def forward(self, X: torch.Tensor, with_variance: bool = False):
        if with_variance:
            if not self.is_fitted:
                raise ValueError(
                    "`compute_covariance` should be called before setting"
                    "`return_covariance` to True"
                )
            features = self.rff(X)
            logits = self.weight(features)
            with torch.no_grad():
                variances = torch.bmm(
                    features[:, None, :],
                    (features @ self.covariance)[:, :, None],
                ).reshape(-1)

            return mean_field_logits(logits, variances), variances
        else:
            logits = self.pipeline(X)
            return logits

    def compute_covariance(self, X: torch.Tensor):
        with torch.no_grad():
            # To stabilize the inverse computation
            precision = self.ridge_penalty * torch.eye(self.n_inducing)
            if self.batch_size is None:
                features = self.rff(X)
                precision = precision + features.T @ features
            else:
                n = math.ceil(X.shape[0] / self.batch_size)
                it = tqdm(
                    torch.tensor_split(X, n), total=n, disable=not self.verbose
                )
                for X_part in it:
                    features = self.rff(X_part)
                    precision = (
                        self.momentum * precision
                        + (1 - self.momentum) * features.T @ features
                    )

            self.covariance[...] = precision.cholesky_inverse()
            self.is_fitted = True

    def reset_covariance(self):
        self.is_fitted = False
        self.covariance.zero_()
