import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import Dataset, DataLoader

from core.speaker_gate import SpeakerVerificationGate


def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, "r") as f:
        f.readline()
        for line in f:
            mix_id, utt_id, *aux = line.strip().split(",")
            aux_it = iter(aux)
            aux = [
                (auxpath, int(float(length)))
                for auxpath, length in zip(aux_it, aux_it)
            ]
            data[mix_id][utt_id] = aux
    return data


class SpeakerGateDataset(Dataset):

    def __init__(self, csv_dir, librimix_meta_dir, task="sep_noisy",
                 sample_rate=16000, n_src=2, segment=3, segment_aux=3):
        self.sample_rate = sample_rate
        self.seg_len = int(segment * sample_rate) if segment else None
        self.seg_len_aux = int(segment_aux * sample_rate) if segment_aux else None

        from asteroid.data import LibriMix as AsteroidLibriMix
        base = AsteroidLibriMix(csv_dir, task, sample_rate, n_src, segment)
        self.df = base.df

        enrollment_path = Path(csv_dir) / "mixture2enrollment.csv"
        self.data_aux = read_enrollment_csv(enrollment_path)

        if self.seg_len_aux is not None:
            self.data_aux = {
                m: {
                    u: [(p, l) for p, l in self.data_aux[m][u] if l >= self.seg_len_aux]
                    for u in self.data_aux[m]
                }
                for m in self.data_aux
            }

        self.pairs = []
        spk_to_enrollment = defaultdict(list)

        for m in self.data_aux:
            for u in self.data_aux[m]:
                if self.data_aux[m][u]:
                    spk = u.split("-")[0]
                    spk_to_enrollment[spk].extend(self.data_aux[m][u])
                    self.pairs.append((m, u))

        self.all_speakers = sorted(spk_to_enrollment.keys())
        self.spk_to_enrollment = {
            s: list(set(spk_to_enrollment[s])) for s in spk_to_enrollment
        }

        self._pair_to_wrong_speakers = {}
        for m, u in self.pairs:
            mix_spks = set()
            for part in m.split("_"):
                mix_spks.add(part.split("-")[0])
            wrong = [s for s in self.all_speakers if s not in mix_spks]
            self._pair_to_wrong_speakers[(m, u)] = wrong

    def __len__(self):
        return len(self.pairs) * 2

    def _load_segment(self, path, seg_len):
        info = sf.info(path)
        total = info.frames
        if seg_len and total > seg_len:
            start = random.randint(0, total - seg_len)
            wav, _ = sf.read(path, dtype="float32", start=start, stop=start + seg_len)
        else:
            wav, _ = sf.read(path, dtype="float32")
        wav = torch.from_numpy(wav)
        if seg_len and wav.shape[-1] < seg_len:
            wav = torch.nn.functional.pad(wav, (0, seg_len - wav.shape[-1]))
        return wav

    def __getitem__(self, idx):
        is_positive = idx % 2 == 0
        pair_idx = idx // 2
        mix_id, utt_id = self.pairs[pair_idx]

        row = self.df[self.df["mixture_ID"] == mix_id].iloc[0]
        mixture_path = row["mixture_path"]
        mix_wav = self._load_segment(mixture_path, self.seg_len)

        if is_positive:
            enr_path, enr_len = random.choice(self.data_aux[mix_id][utt_id])
            enr_wav = self._load_segment(enr_path, self.seg_len_aux)
            label = 1.0
        else:
            wrong_spks = self._pair_to_wrong_speakers[(mix_id, utt_id)]
            if not wrong_spks:
                enr_path, enr_len = random.choice(self.data_aux[mix_id][utt_id])
                enr_wav = self._load_segment(enr_path, self.seg_len_aux)
                label = 1.0
            else:
                wrong_spk = random.choice(wrong_spks)
                wrong_enrollments = [
                    (p, l) for p, l in self.spk_to_enrollment[wrong_spk]
                    if not self.seg_len_aux or l >= self.seg_len_aux
                ]
                if wrong_enrollments:
                    enr_path, enr_len = random.choice(wrong_enrollments)
                    enr_wav = self._load_segment(enr_path, self.seg_len_aux)
                    label = 0.0
                else:
                    enr_path, enr_len = random.choice(
                        self.spk_to_enrollment[wrong_spk]
                    )
                    enr_wav = self._load_segment(enr_path, self.seg_len_aux)
                    label = 0.0

        return mix_wav, enr_wav, torch.tensor(label)


class SpeakerGateLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = SpeakerVerificationGate(**config["model"])
        self.config = config
        self.save_hyperparameters(config)
        self.freeze_backbone = config["train"].get("freeze_backbone", True)
        self.unfreeze_after = config["train"].get("unfreeze_after_epoch", 10)

        if self.freeze_backbone:
            for param in self.model.ecapa_tdnn.parameters():
                param.requires_grad = False

    def on_train_epoch_start(self):
        if self.freeze_backbone and self.current_epoch >= self.unfreeze_after:
            for param in self.model.ecapa_tdnn.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
            self.print(f"Epoch {self.current_epoch}: Unfroze ECAPA backbone")

    def forward(self, mixture, enrollment):
        return self.model(mixture, enrollment, aug=False)

    def training_step(self, batch, batch_idx):
        mix_wav, enr_wav, labels = batch
        logits = self.model(
            mix_wav, enr_wav,
            aug=self.config["train"].get("spectral_aug", False),
        )
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=mix_wav.size(0))
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=mix_wav.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        mix_wav, enr_wav, labels = batch
        logits = self.model(mix_wav, enr_wav, aug=False)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=mix_wav.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=mix_wav.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["optim"]["lr"],
            weight_decay=self.config["optim"]["weight_decay"],
        )
        sched_cfg = self.config["scheduler"]
        warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda ep: max(ep / max(sched_cfg["warmup_epochs"], 1), 0.1)
            if ep < sched_cfg["warmup_epochs"] else 1.0,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=sched_cfg["t_max"], eta_min=sched_cfg["eta_min"],
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[sched_cfg["warmup_epochs"]],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                "interval": "epoch", "frequency": 1}}


class GateDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        ds_cfg = self.config["dataset"]
        self.train_ds = SpeakerGateDataset(
            csv_dir=ds_cfg["train_100_dir"],
            librimix_meta_dir=ds_cfg["librimix_meta_dir"],
            task=ds_cfg["task"],
            sample_rate=ds_cfg["sample_rate"],
            n_src=ds_cfg["n_src"],
            segment=ds_cfg["segment"],
            segment_aux=ds_cfg["segment_aux"],
        )
        self.val_ds = SpeakerGateDataset(
            csv_dir=ds_cfg["val_dir"],
            librimix_meta_dir=ds_cfg["librimix_meta_dir"],
            task=ds_cfg["task"],
            sample_rate=ds_cfg["sample_rate"],
            n_src=ds_cfg["n_src"],
            segment=ds_cfg["segment"],
            segment_aux=ds_cfg["segment_aux"],
        )
        print(f"Train: {len(self.train_ds)} pairs, Val: {len(self.val_ds)} pairs")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["train"]["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config["train"]["batch_size"],
            shuffle=False,
            num_workers=self.config["train"]["num_workers"],
            pin_memory=True,
        )


def load_ecapa_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = {}
    for k, v in ckpt["state_dict"].items():
        if "ecapa_tdnn" in k:
            new_key = k.split("model.")[-1]
            src[new_key] = v
    missing, unexpected = model.load_state_dict(src, strict=False)
    loaded = len(src) - len(unexpected)
    print(f"Loaded {loaded} ECAPA weights from {ckpt_path}")
    if missing:
        missing_ecapa = [k for k in missing if "ecapa" in k]
        if missing_ecapa:
            print(f"  WARNING: {len(missing_ecapa)} ECAPA keys missing")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["seed"])
    torch.set_float32_matmul_precision("medium")

    os.makedirs(config["train"]["log_dir"], exist_ok=True)
    os.makedirs(config["checkpoint"]["dir"], exist_ok=True)

    model = SpeakerGateLightning(config)

    ecapa_ckpt = config["checkpoint"].get("pretrained_ecapa_ckpt")
    if ecapa_ckpt and os.path.exists(ecapa_ckpt):
        model.model = load_ecapa_weights(model.model, ecapa_ckpt)
    else:
        print(f"No pretrained ECAPA checkpoint found at {ecapa_ckpt}, training from scratch")

    data_module = GateDataModule(config)

    callbacks = []
    if config["early_stopping"]["enabled"]:
        callbacks.append(EarlyStopping(
            monitor=config["early_stopping"]["monitor"],
            patience=config["early_stopping"]["patience"],
            verbose=True, mode=config["early_stopping"]["mode"],
        ))
    callbacks.append(ModelCheckpoint(
        dirpath=config["checkpoint"]["dir"],
        filename=config["checkpoint"]["ckpt_name"],
        save_top_k=1, save_last=True, verbose=True,
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
    ))
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    tb_logger = TensorBoardLogger(
        save_dir=config["train"]["log_dir"],
        name="lightning_logs", version="speaker_gate",
    )

    trainer = pl.Trainer(
        max_epochs=config["train"]["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        accumulate_grad_batches=config["train"]["accumulation_steps"],
        callbacks=callbacks,
        default_root_dir=config["train"]["log_dir"],
        logger=tb_logger,
        log_every_n_steps=50,
        precision=config["train"]["precision"],
        gradient_clip_val=config["train"]["gradient_clip_val"],
    )

    trainer.fit(model, datamodule=data_module)
    print(f"\nDone! Best checkpoint: {config['checkpoint']['dir']}")


if __name__ == "__main__":
    main()
