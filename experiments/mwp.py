import hydra
import pytorch_lightning as pl
import sklearn.datasets
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pl_bolts.datamodules import SklearnDataModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from sngp_pytorch import LitRandomFeatureGaussianProcess


@hydra.main(config_path="config", config_name="mwp.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    make_data = dict(cfg.make_data)
    data_fn_name = make_data.pop("call")
    n_dims = make_data.pop("n_dims")
    n_classes = make_data.pop("n_classes")
    data_fn = getattr(sklearn.datasets, data_fn_name)

    X, y = data_fn(**make_data)
    datamodule = SklearnDataModule(
        X,
        y,
        random_state=cfg.seed,
        batch_size=cfg.batch_size,
        test_split=0.0,
        num_workers=cfg.num_workers,
    )

    filename = f"{data_fn_name}" "-{epoch:02d}-{val_loss:.2f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="val/loss_epoch",
        filename=filename,
        save_top_k=3,
        mode="min",
    )

    model = LitRandomFeatureGaussianProcess(
        backbone_dim=n_dims,
        n_classes=n_classes,
        backbone=nn.BatchNorm1d(n_dims),
        **cfg.model,
    )

    wandb_logger = WandbLogger(project="sngp-mwp")
    wandb_logger.watch(model)

    trainer = pl.Trainer(
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/loss_epoch", patience=5),
            checkpoint_callback,
        ],
        logger=wandb_logger,
        **cfg.trainer,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
