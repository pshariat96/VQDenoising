import os
import random
import shutil
from collections import defaultdict

import torch
import torchaudio
from tqdm import tqdm

LIBRIMIX_DIR = os.environ.get("LIBRIMIX_DIR", "data/Libri2Mix") + "/sep_clean"
DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
OUTPUT_BASE = "reuse_experiment/input"
SAMPLE_RATE = 16000
N_LIBRIMIX = 500
SEED = 42


def parse_rttm(rttm_path):
    segments = defaultdict(list)
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            spk = parts[7]
            start = float(parts[3])
            dur = float(parts[4])
            segments[spk].append((start, start + dur))
    return segments


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio if audio.dim() == 2 else audio.unsqueeze(0), target_sr


def prepare_dihard():
    out_dir = os.path.join(OUTPUT_BASE, "dihard")
    os.makedirs(out_dir, exist_ok=True)

    recordings = ["DH_EVAL_0011", "DH_EVAL_0012"]
    for rec_id in recordings:
        audio_path = os.path.join(DIHARD_DIR, f"flac/{rec_id}.flac")
        if not os.path.exists(audio_path):
            print(f"  Skipping {rec_id}: not found")
            continue

        wav, sr = load_audio(audio_path, SAMPLE_RATE)
        out_path = os.path.join(out_dir, f"{rec_id}.wav")
        torchaudio.save(out_path, wav, SAMPLE_RATE)
        dur = wav.shape[-1] / SAMPLE_RATE
        print(f"  {rec_id}: {dur:.1f}s -> {out_path}")

    print(f"  DIHARD: {len(recordings)} recordings prepared")


def prepare_librimix():
    out_dir = os.path.join(OUTPUT_BASE, "librimix")
    os.makedirs(os.path.join(out_dir, "mixture"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "clean_s1"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "enrollment"), exist_ok=True)

    mix_dir = os.path.join(LIBRIMIX_DIR, "mix_both")
    if not os.path.exists(mix_dir):
        mix_dir = os.path.join(LIBRIMIX_DIR, "mix_clean")
    s1_dir = os.path.join(LIBRIMIX_DIR, "s1")

    mix_ids = {f.replace(".wav", "") for f in os.listdir(mix_dir) if f.endswith(".wav")}
    s1_ids = {f.replace(".wav", "") for f in os.listdir(s1_dir) if f.endswith(".wav")}
    valid_ids = sorted(mix_ids & s1_ids)

    spk_map = defaultdict(list)
    for mid in valid_ids:
        spk_map[mid.split("-")[0]].append(mid)

    eligible = [mid for mid in valid_ids if len(spk_map[mid.split("-")[0]]) >= 2]
    random.seed(SEED)
    selected = random.sample(eligible, min(N_LIBRIMIX, len(eligible)))

    for mix_id in tqdm(selected, desc="  LibriMix"):
        mix_wav, _ = load_audio(os.path.join(mix_dir, f"{mix_id}.wav"), SAMPLE_RATE)
        s1_wav, _ = load_audio(os.path.join(s1_dir, f"{mix_id}.wav"), SAMPLE_RATE)

        spk1 = mix_id.split("-")[0]
        others = [m for m in spk_map[spk1] if m != mix_id]
        enroll_id = random.choice(others)
        enroll_wav, _ = load_audio(os.path.join(s1_dir, f"{enroll_id}.wav"), SAMPLE_RATE)

        if enroll_wav.shape[-1] > SAMPLE_RATE * 3:
            enroll_wav = enroll_wav[:, :SAMPLE_RATE * 3]
        elif enroll_wav.shape[-1] < SAMPLE_RATE * 3:
            enroll_wav = torch.nn.functional.pad(
                enroll_wav, (0, SAMPLE_RATE * 3 - enroll_wav.shape[-1])
            )

        torchaudio.save(os.path.join(out_dir, "mixture", f"{mix_id}.wav"), mix_wav, SAMPLE_RATE)
        torchaudio.save(os.path.join(out_dir, "clean_s1", f"{mix_id}.wav"), s1_wav, SAMPLE_RATE)
        torchaudio.save(os.path.join(out_dir, "enrollment", f"{mix_id}.wav"), enroll_wav, SAMPLE_RATE)

    manifest = os.path.join(out_dir, "manifest.txt")
    with open(manifest, "w") as f:
        for mid in selected:
            spk1 = mid.split("-")[0]
            others = [m for m in spk_map[spk1] if m != mid]
            enroll_id = random.choice(others)
            f.write(f"{mid}\t{enroll_id}\n")

    print(f"  LibriMix: {len(selected)} samples prepared")


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print("Preparing DIHARD recordings...")
    prepare_dihard()

    print("\nPreparing LibriMix samples...")
    prepare_librimix()

    total_files = 0
    for root, dirs, files in os.walk(OUTPUT_BASE):
        total_files += len([f for f in files if f.endswith(".wav")])

    print(f"\n{'='*60}")
    print(f"  All input audio saved to: {OUTPUT_BASE}/")
    print(f"  Total wav files: {total_files}")
    print(f"  Structure:")
    print(f"    dihard/          - 2 full recordings (DH_EVAL_0011, 0012)")
    print(f"    librimix/")
    print(f"      mixture/       - 50 noisy mixtures")
    print(f"      clean_s1/      - 50 clean ground truth (speaker 1)")
    print(f"      enrollment/    - 50 enrollment clips (3s each)")
    print(f"{'='*60}")
    print(f"\nNext: Upload this folder to Colab and run the RE-USE notebook.")


if __name__ == "__main__":
    main()
