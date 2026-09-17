import os
import argparse
import json
import numpy as np
import soundfile as sf
from audiomentations import (
    Compose,
    AddGaussianNoise,
    AddGaussianSNR,
    LowPassFilter,
    HighPassFilter,
    ClippingDistortion,
)

SR = 16000

def build_augmentations():
    return {
        "snr_0dB": Compose([AddGaussianSNR(min_snr_db=-2.0, max_snr_db=2.0, p=1.0)]),
        "snr_5dB": Compose([AddGaussianSNR(min_snr_db=3.0, max_snr_db=7.0, p=1.0)]),
        "snr_10dB": Compose([AddGaussianSNR(min_snr_db=8.0, max_snr_db=12.0, p=1.0)]),
        "snr_20dB": Compose([AddGaussianSNR(min_snr_db=18.0, max_snr_db=22.0, p=1.0)]),
    }


def process_directory(input_dir, output_dir, max_files=None):
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])
    if max_files:
        wav_files = wav_files[:max_files]

    augmentations = build_augmentations()

    manifest = {}

    for snr_name, aug in augmentations.items():
        snr_dir = os.path.join(output_dir, snr_name)
        os.makedirs(snr_dir, exist_ok=True)
        manifest[snr_name] = []

        print(f"\n  {snr_name}: augmenting {len(wav_files)} files...")
        for fname in wav_files:
            audio, sr = sf.read(os.path.join(input_dir, fname), dtype='float32')
            if sr != SR:
                import torchaudio
                import torch
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio), sr, SR
                ).numpy()

            noisy = aug(samples=audio, sample_rate=SR)
            out_path = os.path.join(snr_dir, fname)
            sf.write(out_path, noisy, SR)
            manifest[snr_name].append(fname)

    manifest_path = os.path.join(output_dir, 'augmentation_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            "source_dir": input_dir,
            "snr_levels": list(augmentations.keys()),
            "files_per_level": len(wav_files),
        }, f, indent=2)

    print(f"\n  Done. Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of clean wav files")
    parser.add_argument("--output_dir", required=True, help="Output directory for noisy versions")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.max_files)
