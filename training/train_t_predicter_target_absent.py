import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from asteroid.engine.optimizers import make_optimizer
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    LambdaLR,
    SequentialLR,
)

from flow_matching.path import CondOTProbPath
from data.datasets import get_dataloaders
from models.t_predicter import TPredicter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def parse_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TargetAbsentLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TPredicter(**config["model"])
        self.config = config
        self.save_hyperparameters(config)
        self.target_absent_prob = config["train"].get("target_absent_prob", 0.3)

    def forward(self, x, enrollment):
        return self.model(x, enrollment, aug=False)

    def training_step(self, batch, batch_idx):
        source = batch["source_rescaled"]
        background = batch["background_rescaled"]
        enrollment = batch["enroll"]
        batch_size = source.size(0)
        path = CondOTProbPath()

        alpha = torch.rand((batch_size,), device=source.device)

        absent_mask = torch.rand((batch_size,), device=source.device) < self.target_absent_prob

        source_modified = source.clone()
        alpha_target = alpha.clone()

        if absent_mask.any():
            source_modified[absent_mask] = 0.0
            alpha_target[absent_mask] = 0.0

        path_sample = path.sample(t=alpha, x_0=background, x_1=source_modified)
        mixture = path_sample.x_t

        alpha_hat = self.model(
            mixture,
            enrollment,
            aug=self.config["train"].get("spectral_aug", False),
        )
        loss = torch.nn.functional.mse_loss(alpha_hat, alpha_target)

        n_absent = absent_mask.sum().item()
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_absent_frac", n_absent / batch_size,
                 on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        source = batch["source_rescaled"]
        background = batch["background_rescaled"]
        enrollment = batch["enroll"]
        batch_size = source.size(0)
        path = CondOTProbPath()

        alpha = torch.rand((batch_size,), device=source.device)

        absent_mask = torch.rand((batch_size,), device=source.device) < self.target_absent_prob

        source_modified = source.clone()
        alpha_target = alpha.clone()
        if absent_mask.any():
            source_modified[absent_mask] = 0.0
            alpha_target[absent_mask] = 0.0

        path_sample = path.sample(t=alpha, x_0=background, x_1=source_modified)
        mixture = path_sample.x_t

        alpha_hat = self.model(mixture, enrollment, aug=False)
        loss = torch.nn.functional.mse_loss(alpha_hat, alpha_target)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = make_optimizer(
            self.model.parameters(), **self.config["optim"]
        )
        sched_cfg = self.config["scheduler"]

        if sched_cfg["type"] == "ReduceLROnPlateau":
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    factor=sched_cfg["lr_reduce_factor"],
                    patience=sched_cfg["lr_reduce_patience"],
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif sched_cfg["type"] == "CosineAnnealingLR":
            warmup = LambdaLR(
                optimizer,
                lr_lambda=lambda ep: ep / sched_cfg["warmup_epochs"]
                if ep < sched_cfg["warmup_epochs"]
                else 1.0,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg["t_max"],
                eta_min=sched_cfg["eta_min"],
            )
            scheduler = {
                "scheduler": SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[sched_cfg["warmup_epochs"]],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_dataloaders(
            self.config, is_ddp=False, world_size=1, rank=0
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main():
    args = parse_args()
    config = parse_config(args.config)
    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")

    os.makedirs(config["train"]["log_dir"], exist_ok=True)
    os.makedirs(config["checkpoint"]["dir"], exist_ok=True)

    if config["checkpoint"]["load_weights_only"] and config["checkpoint"]["resume"]:
        model = TargetAbsentLightningModule.load_from_checkpoint(
            config["checkpoint"]["resume"],
            config=config,
            strict=True,
        )
        print(f"Loaded weights from {config['checkpoint']['resume']}")
    else:
        model = TargetAbsentLightningModule(config)

    data_module = DataModule(config)

    callbacks = []
    if config["early_stopping"]["enabled"]:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["monitor"],
                patience=config["early_stopping"]["patience"],
                verbose=config["early_stopping"]["verbose"],
                mode=config["early_stopping"]["mode"],
                min_delta=config["early_stopping"]["delta"],
            )
        )

    callbacks.append(
        ModelCheckpoint(
            dirpath=config["checkpoint"]["dir"],
            filename=config["checkpoint"]["ckpt_name"],
            save_top_k=config["checkpoint"]["save_best"],
            save_last=config["checkpoint"]["save_last"],
            verbose=config["checkpoint"]["verbose"],
            monitor=config["checkpoint"]["monitor"],
            mode=config["checkpoint"]["mode"],
        )
    )

    if config["train"]["log_lr"]:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    tb_logger = TensorBoardLogger(
        save_dir=config["train"]["log_dir"],
        name="lightning_logs",
        version="target_absent",
    )

    trainer = pl.Trainer(
        max_epochs=config["train"]["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        accumulate_grad_batches=config["train"]["accumulation_steps"],
        callbacks=callbacks,
        default_root_dir=config["train"]["log_dir"],
        logger=tb_logger,
        log_every_n_steps=config["train"]["log_interval"],
        precision=config["train"]["precision"],
        gradient_clip_val=config["train"]["gradient_clip_val"],
        detect_anomaly=config["train"]["detect_anomaly"],
        limit_train_batches=config["train"]["limit_train_batches"],
        limit_val_batches=config["train"]["limit_val_batches"],
    )

    trainer.fit(model, datamodule=data_module)
    print(f"\nTraining complete. Best checkpoint at: {config['checkpoint']['dir']}")


if __name__ == "__main__":
    main()
