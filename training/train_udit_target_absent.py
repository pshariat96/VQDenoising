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
from flow_matching.solver.ode_solver import ODESolver

from utils import neg_sisdr_loss_wrapper
from data.datasets import get_dataloaders
from models.udit.udit import UDiT
from utils.transforms import istft_torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def parse_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TargetAbsentUDiTModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = UDiT(**config["model"])
        self.config = config
        self.save_hyperparameters(config)
        self.neg_si_sdr = neg_sisdr_loss_wrapper
        self.target_absent_prob = config["train"].get("target_absent_prob", 0.3)

    def forward(self, x, t, enrollment):
        return self.model(x, t, enrollment)

    def training_step(self, batch, batch_idx):
        source = batch["source_rescaled_spec"]
        background = batch["background_rescaled_spec"]
        enrollment = batch["enroll_spec"]
        batch_size = source.size(0)
        path = CondOTProbPath()

        alpha = torch.rand((batch_size,), device=source.device)

        absent_mask = (
            torch.rand((batch_size,), device=source.device) < self.target_absent_prob
        )

        source_modified = source.clone()
        if absent_mask.any():
            source_modified[absent_mask] = 0.0

        path_sample = path.sample(t=alpha, x_0=background, x_1=source_modified)
        mixture = path_sample.x_t
        vector_field = path_sample.dx_t
        vector_field_hat = self.model(mixture, alpha, enrollment)

        loss = torch.pow(vector_field - vector_field_hat, 2).mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        source = batch["source_rescaled_spec"]
        background = batch["background_rescaled_spec"]
        enrollment = batch["enroll_spec"]
        batch_size = source.size(0)
        path = CondOTProbPath()

        if self.config["dataset"]["snr_range"] is None:
            alpha = torch.rand((1,), device=source.device)
        else:
            alpha = batch["alpha"].squeeze(1)

        path_sample = path.sample(t=alpha, x_0=background, x_1=source)
        mixture = path_sample.x_t

        solver = ODESolver(velocity_model=self.model)
        alpha_grid = torch.tensor(
            [alpha.mean().item(), 1.0], device=source.device
        )
        source_hat_spec = solver.sample(
            time_grid=alpha_grid,
            x_init=mixture,
            method=self.config["solver"]["method"],
            step_size=self.config["solver"]["step_size"],
            enrollment=enrollment,
        )
        source_hat = istft_torch(
            source_hat_spec,
            n_fft=self.config["dataset"]["n_fft"],
            hop_length=self.config["dataset"]["hop_length"],
            win_length=self.config["dataset"]["win_length"],
            length=batch["source"].shape[-1],
        )
        loss = self.neg_si_sdr(source_hat, batch["source"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = make_optimizer(
            self.model.parameters(), **self.config["optim"]
        )
        sched_cfg = self.config["scheduler"]

        if sched_cfg["type"] == "ReduceLROnPlateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        factor=sched_cfg["lr_reduce_factor"],
                        patience=sched_cfg["lr_reduce_patience"],
                    ),
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            }

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
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": SequentialLR(
                        optimizer,
                        schedulers=[warmup, cosine],
                        milestones=[sched_cfg["warmup_epochs"]],
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_dataloaders(
            self.config,
            is_ddp=self.config.get("ddp", {}).get("use_ddp", False),
            world_size=self.world_size,
            rank=self.rank,
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
        model = TargetAbsentUDiTModule.load_from_checkpoint(
            config["checkpoint"]["resume"],
            config=config,
            strict=True,
        )
        print(f"Loaded weights from {config['checkpoint']['resume']}")
    else:
        model = TargetAbsentUDiTModule(config)

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

    ddp_config = config.get("ddp", {})
    use_ddp = ddp_config.get("use_ddp", False)
    num_gpus = ddp_config.get("num_gpus", torch.cuda.device_count())
    num_nodes = ddp_config.get("num_nodes", 1)

    if use_ddp and num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
        trainer = pl.Trainer(
            max_epochs=config["train"]["num_epochs"],
            accelerator="gpu",
            devices=num_gpus,
            num_nodes=num_nodes,
            strategy=strategy,
            accumulate_grad_batches=config["train"]["accumulation_steps"],
            callbacks=callbacks,
            default_root_dir=config["train"]["log_dir"],
            logger=tb_logger,
            log_every_n_steps=config["train"]["log_interval"],
            precision=config["train"]["precision"],
            detect_anomaly=config["train"]["detect_anomaly"],
            gradient_clip_val=config["train"]["gradient_clip_val"],
            limit_train_batches=config["train"]["limit_train_batches"],
            limit_val_batches=config["train"]["limit_val_batches"],
        )
    else:
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

    if config["checkpoint"]["load_weights_only"]:
        trainer.fit(model, datamodule=data_module)
    else:
        ckpt_path = config["checkpoint"]["resume"] or None
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    print(f"\nTraining complete. Best checkpoint: {config['checkpoint']['dir']}")


if __name__ == "__main__":
    main()
