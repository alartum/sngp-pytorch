import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import mean_field_logits
from .random_feature import Cos, RandomFeatureGaussianProcess


class LitRandomFeatureGaussianProcess(pl.LightningModule):
    def __init__(
        self,
        backbone_dim: int,
        n_classes: int,
        backbone: nn.Module = nn.Identity(),
        n_inducing: int = 1024,
        momentum: float = 0.9,
        ridge_penalty: float = 1e-6,
        activation: str = "cos",
        verbose: bool = False,
        l2_reg=1e-4,
        learning_rate=1e-3,
        optimizer="adam",
        log_covariance=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        supported = ["cos"]
        if activation not in supported:
            raise ValueError(
                f"Expected `activation` one of {supported}, "
                f"but got {activation} instead"
            )
        elif activation == "cos":
            activation = Cos()

        self.model = RandomFeatureGaussianProcess(
            in_features=backbone_dim,
            out_features=n_classes,
            backbone=backbone,
            n_inducing=n_inducing,
            momentum=momentum,
            ridge_penalty=ridge_penalty,
            activation=activation,
            verbose=verbose,
        )

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_covariance = log_covariance

        self.criterion = CrossEntropyLoss(reduction="mean")
        self.l2 = MSELoss(reduction="mean")

    def forward(
        self,
        X: torch.Tensor,
        with_variance: bool = False,
        mean_field: bool = False,
    ):
        if with_variance or mean_field:
            logits, variances = self.model.forward(X, with_variance=True)
        else:
            logits = self.model.forward(X, with_variance=False)
            return logits

        # Compute the mean field logits expectation
        if mean_field:
            logits = mean_field_logits(logits, variances)

        if with_variance:
            return logits, variances
        else:
            return logits

    def training_step(self, batch, _):
        # Keep covariance unfitted
        self.model.reset_covariance()

        X, y = batch

        logits = self.model(X, with_variance=False, update_precision=True)

        # Combine classification loss with prior regularization
        class_loss = self.criterion(logits, y)
        betas = self.model.weight.weight
        l2_loss = self.l2_reg * self.l2(betas, betas.new_zeros(betas.shape))
        loss = class_loss + l2_loss

        with torch.no_grad():
            y_pr = logits.argmax(dim=-1)
            acc = (y_pr == y).to(torch.float32).mean()

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        stats = {
            "train/total": loss,
            "train/class": class_loss,
            "train/prior": l2_loss,
            "train/acc": acc,
        }
        self.logger.log_metrics(stats)

        return loss

    def validation_step(self, batch, _):
        # Fit the covariance and show its matrix once
        show_image = not self.model.is_fitted
        self.model.update_covariance()
        if self.log_covariance and show_image:
            scale = self.model.covariance.max() - self.model.covariance.min()
            min = self.model.covariance.min()
            self.logger.log_image(
                "covariance", [(self.model.covariance - min) / scale]
            )

        X, y = batch
        logits, variances = self.model(
            X, with_variance=True, update_precision=False
        )

        # Combine classification loss with prior regularization
        class_loss = self.criterion(logits, y)
        betas = self.model.weight.weight
        l2_loss = self.l2_reg * self.l2(betas, betas.new_zeros(betas.shape))
        loss = class_loss + l2_loss

        # Compute the expectation of logits
        logits = mean_field_logits(logits, variances)
        class_expect_loss = self.criterion(logits, y)

        y_pr = logits.argmax(dim=-1)
        acc = (y_pr == y).to(torch.float32).mean()

        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val/acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        stats = {
            "val/total": loss,
            "val/class": class_loss,
            "val/prior": l2_loss,
            "val/class_expect": class_expect_loss,
            "val/acc": acc,
        }
        self.logger.log_metrics(stats)

        return loss

    def test_step(self, batch, _):
        X, y = batch
        logits, variances = self.model(
            X, with_variance=True, update_precision=False
        )

        # Combine classification loss with prior regularization
        class_loss = self.criterion(logits, y)
        betas = self.model.weight.weight
        l2_loss = self.l2_reg * self.l2(betas, betas.new_zeros(betas.shape))
        loss = class_loss + l2_loss

        # Compute the expectation of logits
        logits = mean_field_logits(logits, variances)
        class_expect_loss = self.criterion(logits, y)

        y_pr = logits.argmax(dim=-1)
        acc = (y_pr == y).to(torch.float32).mean()

        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "test/acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        stats = {
            "test/total": loss,
            "test/class": class_loss,
            "test/prior": l2_loss,
            "test/class_expect": class_expect_loss,
            "test/acc": acc,
        }
        self.logger.log_metrics(stats)

        return loss

    def configure_optimizers(self):
        supported = ["adam", "sgd"]
        if self.optimizer not in supported:
            raise ValueError(
                f"Expected `self.optimizer` one of {supported}, "
                f"but got {self.optimizer} instead"
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            patience=5,
            factor=0.1,
            threshold=1e-4,
            threshold_mode="rel",
            verbose=True,
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "monitor": "train/loss_epoch",
        }

        return [optimizer], [scheduler]
