from typing import List

import torch.nn as nn

from image_uncertainty.cifar.cifar_evaluate import load_model

from ..backbones import resnet
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
    def __init__(self, model_name, model_path, **kwargs):
        backbone = load_model(model_name, model_path, False)
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
