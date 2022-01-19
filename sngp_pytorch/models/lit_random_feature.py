import pytorch_lightning as pl
import torch
import torch.optim
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .random_feature import Cos, RandomFeatureGaussianProcess


def mean_field_logits(
    logits: torch.Tensor,
    variances: torch.Tensor,
    mean_field_factor: float = 1.0,
) -> torch.Tensor:
    logits_scale = (1.0 + variances * mean_field_factor) ** 0.5
    if len(logits.shape) > 1:
        logits_scale = logits_scale[:, None]

    return logits / logits_scale


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
        optimizer_cfg=dict(
            name="SGD", lr=1e-4, momentum=0.9, weight_decay=1e-4
        ),
        lr_scheduler_cfg=dict(
            name="StepLR",
            step_size=30,
            gamma=0.1,
            verbose=True,
        ),
        log_covariance=False,
        save_hyperparameters=True,
    ):
        super().__init__()
        if save_hyperparameters:
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

        self.optimizer_name = optimizer_cfg.pop("name")
        self.optimizer_kwargs = optimizer_cfg
        self.lr_scheduler_name = lr_scheduler_cfg.pop("name")

        self.lr_scheduler_config = lr_scheduler_cfg.pop("config", {})
        self.lr_scheduler_kwargs = lr_scheduler_cfg
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
            y_pr = logits.argsort(descending=True, dim=-1)
            acc1 = (
                (y_pr[:, :1] == y[:, None])
                .to(torch.float32)
                .sum(dim=-1)
                .mean()
            )
            acc5 = (
                (y_pr[:, :5] == y[:, None])
                .to(torch.float32)
                .sum(dim=-1)
                .mean()
            )

        self.log("train/loss", loss, prog_bar=True)

        self.log(
            "train/acc@1",
            acc1,
            prog_bar=True,
        )

        self.log("train/acc@5", acc5, prog_bar=True)

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

        y_pr = logits.argsort(descending=True, dim=-1)
        acc1 = (y_pr[:, :1] == y[:, None]).to(torch.float32).sum(dim=-1).mean()
        acc5 = (y_pr[:, :5] == y[:, None]).to(torch.float32).sum(dim=-1).mean()

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
        )

        self.log(
            "val/acc@1",
            acc1,
            prog_bar=True,
        )

        self.log(
            "val/acc@5",
            acc5,
            prog_bar=True,
        )

        self.log("val/mean_field_loss", class_expect_loss)

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

        y_pr = logits.argsort(descending=True, dim=-1)
        acc1 = (y_pr[:, :1] == y[:, None]).to(torch.float32).sum(dim=-1).mean()
        acc5 = (y_pr[:, :5] == y[:, None]).to(torch.float32).sum(dim=-1).mean()

        self.log(
            "test/loss",
            loss,
            prog_bar=True,
        )

        self.log(
            "test/acc@1",
            acc1,
            prog_bar=True,
        )

        self.log(
            "test/acc@5",
            acc5,
            prog_bar=True,
        )

        self.log("test/mean_field_loss", class_expect_loss)

        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )

        if self.lr_scheduler_name == "SequentialLR":
            schedulers_cfg = self.lr_scheduler_kwargs.pop("schedulers")
            schedulers = []
            for cfg in schedulers_cfg:
                name = cfg.pop("name")
                schedulers.append(
                    getattr(torch.optim.lr_scheduler, name)(optimizer, **cfg)
                )

            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers, **self.lr_scheduler_kwargs
            )
            # Optimizer is not saved by default
            lr_scheduler.optimizer = optimizer

        else:
            lr_scheduler = getattr(
                torch.optim.lr_scheduler, self.lr_scheduler_name
            )(optimizer, **self.lr_scheduler_kwargs)

        scheduler = {"scheduler": lr_scheduler, **self.lr_scheduler_config}

        return [optimizer], [scheduler]
