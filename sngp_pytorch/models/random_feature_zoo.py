from .lit_random_feature import LitRandomFeatureGaussianProcess
import torch.nn as nn


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
