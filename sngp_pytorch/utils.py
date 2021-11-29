from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .backbones.resnet import BasicBlock, ResNet, SN_wrapper


def get_resnet(
    out_features: int = 128,
    input_planes: int = 3,
    n_ch: int = 16,
    last_shape: Tuple[int, int] = (4, 4),
    use_sn: bool = True,
) -> ResNet:
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_ch=[n_ch] * 4,
        use_sn=use_sn,
        input_planes=input_planes,
    )
    model.linear = SN_wrapper(
        nn.Linear(n_ch * np.prod(last_shape), out_features), use_sn=use_sn
    )

    return model


def mean_field_logits(
    logits: torch.Tensor,
    variances: torch.Tensor,
    mean_field_factor: float = 1.0,
) -> torch.Tensor:
    logits_scale = (1.0 + variances * mean_field_factor) ** 0.5
    if len(logits.shape) > 1:
        logits_scale = logits_scale[:, None]

    return logits / logits_scale
