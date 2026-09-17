import csv
import os
import random
import sys
from pathlib import Path


_max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int // 10)

import torch
import soundfile as sf
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader


AUDIO_EXTENSIONS = {'.wav', '.flac'}


def _load_resampled_mono(path, target_sr):
    audio, sr = sf.read(path, dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq"
        )
    return audio


def _segment_pair(noisy, clean, segment_length, rng=None):
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]

    if min_len >= segment_length:
        draw = (rng or random).randint(0, min_len - segment_length)
        noisy = noisy[draw:draw + segment_length]
        clean = clean[draw:draw + segment_length]
    else:
        pad_len = segment_length - min_len
        noisy = np.pad(noisy, (0, pad_len))
        clean = np.pad(clean, (0, pad_len))
    return noisy, clean


class URGENTMetaDataset(Dataset):

    def __init__(self, meta_tsv, sample_rate=16000, segment_length=64000,
                 min_segment_length=16000, data_root=None, max_files=None,
                 deterministic_crop=False):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.min_segment_length = min_segment_length

        self.deterministic_crop = deterministic_crop

        meta_tsv = Path(meta_tsv).resolve()
        if data_root is None:

            data_root = meta_tsv.parent.parent.parent
        self.data_root = Path(data_root)

        self.pairs = []
        with open(meta_tsv, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                noisy = self._resolve(row["noisy_path"])
                clean = self._resolve(row["clean_path"])
                self.pairs.append((noisy, clean, row["id"]))

        if max_files is not None:
            self.pairs = self.pairs[:max_files]

        if len(self.pairs) == 0:
            raise ValueError(f"No pairs found in meta manifest: {meta_tsv}")

    def _resolve(self, p):
        path = Path(p)
        if path.is_absolute():
            return str(path)

        cand = self.data_root / path
        return str(cand)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path, uid = self.pairs[idx]
        noisy = _load_resampled_mono(noisy_path, self.sample_rate)
        clean = _load_resampled_mono(clean_path, self.sample_rate)
        rng = random.Random(idx) if self.deterministic_crop else None
        noisy, clean = _segment_pair(noisy, clean, self.segment_length, rng=rng)
        return (
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float(),
            uid,
        )


class PairedSEDataset(Dataset):

    def __init__(self, noisy_dir, clean_dir, sample_rate=16000,
                 segment_length=64000, min_segment_length=16000,
                 max_files=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.min_segment_length = min_segment_length

        def get_audio_stems(directory):
            stems = {}
            for f in os.listdir(directory):
                stem, ext = os.path.splitext(f)
                if ext.lower() in AUDIO_EXTENSIONS:
                    stems[stem] = f
            return stems

        noisy_stems = get_audio_stems(noisy_dir)
        clean_stems = get_audio_stems(clean_dir)

        common_stems = sorted(set(noisy_stems.keys()) & set(clean_stems.keys()))
        self.pairs = [(noisy_stems[s], clean_stems[s]) for s in common_stems]

        if max_files is not None:
            self.pairs = self.pairs[:max_files]

        if len(self.pairs) == 0:
            raise ValueError(
                f"No matching audio files found in:\n"
                f"  noisy: {noisy_dir}\n"
                f"  clean: {clean_dir}\n"
                f"Supported formats: {AUDIO_EXTENSIONS}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_fname, clean_fname = self.pairs[idx]
        noisy = _load_resampled_mono(
            os.path.join(self.noisy_dir, noisy_fname), self.sample_rate
        )
        clean = _load_resampled_mono(
            os.path.join(self.clean_dir, clean_fname), self.sample_rate
        )
        noisy, clean = _segment_pair(noisy, clean, self.segment_length)
        return (
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float(),
            noisy_fname,
        )


def _urgent_dataset(config):
    d = config['data']
    return URGENTMetaDataset(
        meta_tsv=d['train_meta_tsv'],
        sample_rate=d['sample_rate'],
        segment_length=d['segment_length'],
        min_segment_length=d['min_segment_length'],
        data_root=d.get('data_root') or None,
        max_files=d.get('max_files', None),
    )


def _make_dataset(config, split):
    d = config['data']
    return PairedSEDataset(
        noisy_dir=d[f'{split}_noisy_dir'],
        clean_dir=d[f'{split}_clean_dir'],
        sample_rate=d['sample_rate'],
        segment_length=d['segment_length'],
        min_segment_length=d['min_segment_length'],
        max_files=d.get('max_files', None),
    )


def create_dataloaders(config):
    d = config['data']
    if d.get('train_meta_tsv'):
        if d.get('test_meta_tsv'):
            train_dataset = _urgent_dataset(config)
            test_cfg = {**config, 'data': {**d, 'train_meta_tsv': d['test_meta_tsv']}}
            test_dataset = _urgent_dataset(test_cfg)
        else:
            full = _urgent_dataset(config)
            n = len(full)
            n_test = max(1, int(round(n * float(d.get('val_split', 0.05)))))
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(n, generator=g).tolist()
            test_idx, train_idx = perm[:n_test], perm[n_test:]
            train_dataset = torch.utils.data.Subset(full, train_idx)
            test_dataset = torch.utils.data.Subset(full, test_idx)
    else:
        train_dataset = _make_dataset(config, "train")
        test_dataset = _make_dataset(config, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )

    return train_loader, test_loader
