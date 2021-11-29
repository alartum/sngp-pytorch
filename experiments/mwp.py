import hydra
import pytorch_lightning as pl
import sklearn.datasets
from omegaconf import DictConfig, OmegaConf
from pl_bolts.datamodules import SklearnDataModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from sngp_pytorch.datamodules import EmbeddingsDataModule  # noqa F401
from sngp_pytorch.models import LitBatchNorm1dRFGP


@hydra.main(config_path="config", config_name="mwp.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    data = dict(cfg.data)
    data_fn_name = data.pop("call")
    n_dims = data.pop("n_dims")
    n_classes = data.pop("n_classes")
    num_workers = data.pop("num_workers")
    batch_size = data.pop("batch_size")

    try:
        data_fn = getattr(sklearn.datasets, data_fn_name)

        X, y = data_fn(**data)

        datamodule = SklearnDataModule(
            X,
            y,
            random_state=cfg.seed,
            batch_size=batch_size,
            test_split=0.0,
            num_workers=num_workers,
        )
    except AttributeError:
        data_fn = globals()[data_fn_name]

        datamodule = data_fn(
            **data, batch_size=batch_size, num_workers=num_workers
        )

    filename = f"{data_fn_name}" "-{epoch:02d}-{val_loss:.2f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="val/loss_epoch",
        filename=filename,
        save_top_k=3,
        mode="min",
    )

    model = LitBatchNorm1dRFGP(
        in_features=n_dims,
        n_classes=n_classes,
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
    # Optional batch size tuning
    # trainer.tune(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
