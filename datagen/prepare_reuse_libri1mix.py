import argparse
import os
import random
import shutil

import torch
import torchaudio
from tqdm import tqdm


EXPERIMENT_DIR = "reuse_libri1mix_experiment"
CLEAN_S1_DIR = os.environ.get("LIBRIMIX_DIR", "data/Libri2Mix") + "/sep_clean/s1"
SAMPLE_RATE = 16000
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=50,
                   help="Number of samples to prepare for Colab (use 50 for quick, 500 for full)")
    p.add_argument("--noisy_20_dir", default=os.path.join(EXPERIMENT_DIR, "input", "s1_noisy_20"))
    p.add_argument("--noisy_40_dir", default=os.path.join(EXPERIMENT_DIR, "input", "s1_noisy_40"))
    p.add_argument("--output_dir", default=os.path.join(EXPERIMENT_DIR, "input"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)

    noisy_20_files = {f.replace(".wav", "") for f in os.listdir(args.noisy_20_dir) if f.endswith(".wav")}
    noisy_40_files = {f.replace(".wav", "") for f in os.listdir(args.noisy_40_dir) if f.endswith(".wav")}
    clean_s1_files = {f.replace(".wav", "") for f in os.listdir(CLEAN_S1_DIR) if f.endswith(".wav")}

    common_ids = sorted(noisy_20_files & noisy_40_files & clean_s1_files)
    print(f"Found {len(common_ids)} IDs common to noisy_20, noisy_40, and clean_s1")

    selected = random.sample(common_ids, min(args.n_samples, len(common_ids)))
    print(f"Selected {len(selected)} samples for the experiment")

    out_clean = os.path.join(args.output_dir, "clean_s1")
    os.makedirs(out_clean, exist_ok=True)

    for mix_id in tqdm(selected, desc="Copying clean s1 ground truth"):
        src = os.path.join(CLEAN_S1_DIR, f"{mix_id}.wav")
        dst = os.path.join(out_clean, f"{mix_id}.wav")
        shutil.copy2(src, dst)

    manifest_path = os.path.join(args.output_dir, "manifest_libri1mix.txt")
    with open(manifest_path, "w") as f:
        for mix_id in selected:
            f.write(f"{mix_id}\n")

    n20 = len([f for f in os.listdir(args.noisy_20_dir) if f.endswith(".wav")])
    n40 = len([f for f in os.listdir(args.noisy_40_dir) if f.endswith(".wav")])
    n_gt = len(os.listdir(out_clean))

    print(f"\n{'='*60}")
    print(f"  RE-USE Libri1Mix Experiment - Input Prepared")
    print(f"{'='*60}")
    print(f"  s1_noisy_20/   {n20} files (full set)")
    print(f"  s1_noisy_40/   {n40} files (full set)")
    print(f"  clean_s1/      {n_gt} files (selected subset)")
    print(f"  manifest:      {manifest_path}")
    print(f"{'='*60}")
    print(f"\n  Upload {args.output_dir}/ to Google Drive at:")
    print(f"    AD-FlowTSE_checkpoints/reuse_libri1mix_input/")
    print(f"  Then run the reuse_libri1mix_colab.ipynb notebook.")


if __name__ == "__main__":
    main()
