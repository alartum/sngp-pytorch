from copy import deepcopy

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pl_bolts.datamodules import ImagenetDataModule  # noqa F401
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchinfo import summary

from sngp_pytorch.datamodules import CIFAR100DataModule  # noqa F401
from sngp_pytorch.models import LitBackboneRFGP, LitResnetRFGP  # noqa F401


@hydra.main(config_path="config", config_name="cifar100.yaml")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    cfg_backup = deepcopy(cfg)

    pl.seed_everything(cfg["seed"], workers=True)

    data = cfg["data"]
    data.setdefault("data_dir", to_absolute_path("data"))
    data_fn_name = data.pop("name")
    n_classes = data.pop("n_classes")
    num_workers = data.pop("num_workers")
    batch_size = data.pop("batch_size")

    data_fn = globals()[data_fn_name]
    datamodule_cfg = dict(
        **data,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=cfg["seed"],
    )
    try:
        datamodule = data_fn(**datamodule_cfg)
    except TypeError as e:
        # If the data module doesn't expect 'seed' parameter
        if "seed" in repr(e):
            datamodule_cfg.pop("seed")
            datamodule = data_fn(**datamodule_cfg)

    filename = (
        f"{data_fn_name.replace('DataModule', '')}-seed{cfg['seed']}"
        "-epoch{epoch:03d}-val_loss{val/loss:.5f}-val_acc{val/acc@1:.4f}"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="val/loss",
        filename=filename,
        auto_insert_metric_name=False,
        save_top_k=3,
        mode="min",
        every_n_epochs=1,
    )

    model_cfg = cfg["model"]
    model_name = model_cfg.pop("name")

    model_fn = globals()[model_name]
    model = model_fn(
        n_classes=n_classes,
        **model_cfg,
        optimizer_cfg=cfg["optimizer"],
        lr_scheduler_cfg=cfg["lr_scheduler"],
    )

    summary(model, (1, 3, 224, 224))

    wandb_logger = WandbLogger(
        project=f"sngp-{data_fn_name.replace('DataModule', '')}",
        config=cfg_backup,
        log_model=True,
    )
    wandb_logger.watch(model)

    print(OmegaConf.to_yaml(cfg_backup))

    trainer_cfg = cfg["trainer"]
    if trainer_cfg.get("strategy", None) == "ddp":
        # All parameters are used, no need to spend time finding them
        trainer_cfg["strategy"] = DDPPlugin(find_unused_parameters=False)

    # 'val/loss' is logged per epoch by default
    trainer = pl.Trainer(
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val/loss", patience=100),
            checkpoint_callback,
        ],
        logger=wandb_logger,
        **trainer_cfg,
    )

    # Optional batch size tuning
    # trainer.tune(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
