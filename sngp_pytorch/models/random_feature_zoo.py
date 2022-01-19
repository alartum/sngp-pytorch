from typing import List

import torch.nn as nn
import torchvision  # noqa F401

import sngp_pytorch  # noqa F401
from image_uncertainty.cifar.cifar_evaluate import load_model
from image_uncertainty.models import get_model

from ..backbones import resnet
from ..utils import apply_spectral_norm, get_last_fc
from .lit_random_feature import LitRandomFeatureGaussianProcess


class LitBatchNorm1dRFGP(LitRandomFeatureGaussianProcess):
    def __init__(self, in_features: int, n_classes: int, **kwargs):
        backbone = nn.BatchNorm1d(in_features)
        self.save_hyperparameters()
        super().__init__(
            backbone_dim=in_features,
            n_classes=n_classes,
            backbone=backbone,
            save_hyperparameters=False,
            **kwargs
        )


class LitBackboneRFGP(LitRandomFeatureGaussianProcess):
    def __init__(
        self,
        backbone_init: str = "sngp_pytorch.backbones.resnet50()",
        spectral_normalization: bool = True,
        norm_multiplier=6.0,
        n_classes: int = 10,
        **kwargs
    ):
        backbone = eval(backbone_init)
        if spectral_normalization:
            apply_spectral_norm(backbone, norm_multiplier=norm_multiplier)
        fc_name, parent = get_last_fc(backbone)
        backbone_dim = getattr(parent, fc_name).in_features
        setattr(parent, fc_name, nn.Identity())

        self.save_hyperparameters()
        super().__init__(
            backbone_dim=backbone_dim,
            n_classes=n_classes,
            backbone=backbone,
            save_hyperparameters=False,
            **kwargs
        )


class LitResnetRFGP(LitRandomFeatureGaussianProcess):
    def __init__(
        self,
        model_name: str = "ResNet50",
        spectral_normalization: bool = True,
        n_classes: int = 10,
        input_planes: int = 3,
        n_channels: List[int] = [64, 128, 256, 512],
        **kwargs
    ):
        backbone_builder = getattr(resnet, model_name)
        backbone = backbone_builder(
            use_sn=spectral_normalization,
            num_ch=n_channels,
            num_classes=n_classes,
            input_planes=input_planes,
        )
        backbone.linear = nn.Identity()

        self.save_hyperparameters()
        super().__init__(
            backbone_dim=n_channels[-1] * 4,
            n_classes=n_classes,
            backbone=backbone,
            save_hyperparameters=False,
            **kwargs
        )


class LitPretrainedRFGP(LitRandomFeatureGaussianProcess):
    def __init__(self, model_name, model_path, reset=False, **kwargs):
        if reset:
            backbone = get_model(model_name, False)
        else:
            backbone = load_model(model_name, model_path, False)
            backbone.train()

        backbone_dim, n_classes = (
            backbone.linear.in_features,
            backbone.linear.out_features,
        )
        backbone.linear = nn.Identity()

        self.save_hyperparameters()
        super().__init__(
            backbone_dim=backbone_dim,
            n_classes=n_classes,
            backbone=backbone,
            save_hyperparameters=False,
            **kwargs
        )

    def freeze_backbone(self, freeze=True):
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze
