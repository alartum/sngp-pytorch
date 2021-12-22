from os.path import join

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sngp_pytorch.datamodules import CIFAR100CustomDataModule  # noqa F401
from sngp_pytorch.models import LitPretrainedRFGP


@hydra.main(config_path="config", config_name="images_finetune.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    data = dict(cfg.data)
    data_fn_name = data.pop("call")
    _ = data.pop("n_classes")
    num_workers = data.pop("num_workers")
    batch_size = data.pop("batch_size")

    data_fn = globals()[data_fn_name]

    datamodule = data_fn(
        to_absolute_path("data"),
        **data,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )

    filename = f"{data_fn_name}" "-{epoch:02d}-{val_loss:.6f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="val/loss_epoch",
        filename=filename,
        save_top_k=3,
        mode="min",
    )

    with open_dict(cfg):
        model_path = cfg.model.pop("model_path")
        freeze = cfg.model.pop("freeze")
    # Correct seed if using custom CIFAR100
    if data_fn_name == "CIFAR100CustomDataModule":
        model_path = model_path.replace("42", str(cfg.seed))
    model_path = join(to_absolute_path("models"), model_path)
    model = LitPretrainedRFGP(
        model_path=model_path,
        **cfg.model,
    )
    model.freeze_backbone(freeze)

    wandb_logger = WandbLogger(
        project=f"sngp-finetune-{data_fn_name.replace('DataModule', '')}"
    )
    wandb_logger.watch(model)

    trainer = pl.Trainer(
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
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
