import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from core.inference import (
    parse_config,
    load_flowtse_model,
    pad_and_reshape,
    reshape_and_remove_padding,
)
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver


ENROLLMENT_LENGTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]


def si_sdr(estimate, reference):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target
    return (10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )).item()


def try_pesq(estimate_np, reference_np, sr=16000):
    try:
        from pesq import pesq as pesq_fn
        return pesq_fn(sr, reference_np, estimate_np, "wb")
    except Exception:
        return None


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0)


def add_wham_noise(clean_wav, noise_wav, noise_ratio):
    if noise_ratio == 0.0:
        return clean_wav.clone()
    n_samples = clean_wav.shape[-1]
    if noise_wav.shape[-1] >= n_samples:
        start = random.randint(0, noise_wav.shape[-1] - n_samples)
        noise_seg = noise_wav[start:start + n_samples]
    else:
        reps = (n_samples // noise_wav.shape[-1]) + 1
        noise_seg = noise_wav.repeat(reps)[:n_samples]
    rms_clean = clean_wav.pow(2).mean().sqrt().clamp(min=1e-8)
    rms_noise = noise_seg.pow(2).mean().sqrt().clamp(min=1e-8)
    scaled_noise = noise_seg * (rms_clean / rms_noise)
    noisy = clean_wav + noise_ratio * scaled_noise
    peak = noisy.abs().max()
    if peak > 1.0:
        noisy = noisy / peak
    return noisy


def extract_speaker(model, mixture_wav, enrollment_wav, config, device,
                    alpha=0.5, chunk_batch_size=16):
    sr = config["dataset"]["sample_rate"]
    n_fft = config["dataset"]["n_fft"]
    hop_length = config["dataset"]["hop_length"]
    win_length = config["dataset"]["win_length"]

    mixture_spec = stft_torch(mixture_wav, n_fft, hop_length, win_length).unsqueeze(0)
    enrollment_spec = (
        stft_torch(enrollment_wav, n_fft, hop_length, win_length)
        .unsqueeze(0).to(device)
    )

    frames_per_chunk = sr * 3 // hop_length + 1
    mixture_chunks, orig_len = pad_and_reshape(mixture_spec, frames_per_chunk)
    num_chunks = mixture_chunks.shape[0]

    alpha_grid = torch.tensor([alpha, 1.0], device=device)

    all_outputs = []
    with torch.no_grad():
        for i in range(0, num_chunks, chunk_batch_size):
            batch = mixture_chunks[i:i + chunk_batch_size].to(device)
            bs = batch.shape[0]
            solver = ODESolver(velocity_model=model)
            out = solver.sample(
                time_grid=alpha_grid,
                x_init=batch.float(),
                method=config["solver"]["method"],
                step_size=config["solver"]["test_step_size"],
                enrollment=enrollment_spec.repeat(bs, 1, 1),
            )
            all_outputs.append(out.cpu())

    source_hat_spec = torch.cat(all_outputs, dim=0)
    source_hat_spec = reshape_and_remove_padding(source_hat_spec, orig_len)
    source_hat = istft_torch(source_hat_spec, n_fft, hop_length, win_length,
                             length=mixture_wav.shape[-1])
    max_val = source_hat.abs().max()
    if max_val > 1.0:
        source_hat = source_hat / max_val
    return source_hat




def parse_args():
    p = argparse.ArgumentParser(description="Enrollment length vs quality (with optional noise)")
    p.add_argument("--librimix_dir", required=True)
    p.add_argument("--config", default="config/config_FlowTSE_large_noisy.yaml")
    p.add_argument("--output_dir", default="enrollment_length_results")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--noise_ratio", type=float, default=0.0,
                   help="WHAM noise ratio to add to enrollment (0.0=clean, 0.4=40%% noise)")
    p.add_argument("--chunk_batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = parse_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = config["dataset"]["sample_rate"]

    noise_label = "clean" if args.noise_ratio == 0.0 else f"{args.noise_ratio:.0%} noisy"
    print(f"Mode: {noise_label} enrollment")

    sep = os.path.join(args.librimix_dir, "sep_clean")
    mix_dir = os.path.join(sep, "mix_clean")
    s1_dir = os.path.join(sep, "s1")
    s2_dir = os.path.join(sep, "s2")
    noise_dir = os.path.join(sep, "noise")

    mix_ids = {f.replace(".wav", "") for f in os.listdir(mix_dir) if f.endswith(".wav")}
    s1_ids = {f.replace(".wav", "") for f in os.listdir(s1_dir) if f.endswith(".wav")}
    s2_ids = {f.replace(".wav", "") for f in os.listdir(s2_dir) if f.endswith(".wav")}
    valid_ids = sorted(mix_ids & s1_ids & s2_ids)

    noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith(".wav")])

    spk_map = defaultdict(list)
    for mid in valid_ids:
        spk_map[mid.split("-")[0]].append(mid)

    eligible = [mid for mid in valid_ids if len(spk_map[mid.split("-")[0]]) >= 2]
    selected = random.sample(eligible, min(args.n_samples, len(eligible)))
    print(f"Found {len(valid_ids)} valid, {len(eligible)} eligible, selected {len(selected)}")

    enroll_data = {}
    bad_ids = []
    for mix_id in tqdm(selected, desc="Loading enrollment sources"):
        spk1 = mix_id.split("-")[0]
        others = [m for m in spk_map[spk1] if m != mix_id]
        enroll_id = random.choice(others)
        noise_file = random.choice(noise_files)
        try:
            enroll_wav = load_audio(os.path.join(s1_dir, f"{enroll_id}.wav"), sr)
            noise_wav = load_audio(os.path.join(noise_dir, noise_file), sr) if args.noise_ratio > 0 else None
            enroll_data[mix_id] = {
                "enroll_id": enroll_id,
                "enroll_wav": enroll_wav,
                "noise_wav": noise_wav,
            }
        except Exception:
            bad_ids.append(mix_id)
    if bad_ids:
        print(f"Skipped {len(bad_ids)} samples with unreadable files")
        selected = [m for m in selected if m not in bad_ids]

    ckpt_path = config["eval"]["checkpoint"]
    print(f"Loading FlowTSE model from {ckpt_path}")
    model = load_flowtse_model(ckpt_path, config["model"], device)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for enroll_secs in ENROLLMENT_LENGTHS:
        enroll_samples = int(sr * enroll_secs)
        eligible_for_length = [
            mid for mid in selected
            if enroll_data[mid]["enroll_wav"].shape[-1] >= enroll_samples
        ]
        print(f"\n--- {enroll_secs}s ({noise_label}): {len(eligible_for_length)}/{len(selected)} eligible ---")

        if not eligible_for_length:
            continue

        skipped = 0
        for mix_id in tqdm(eligible_for_length, desc=f"{enroll_secs}s"):
            try:
                mixture_wav = load_audio(os.path.join(mix_dir, f"{mix_id}.wav"), sr)
                gt_wav = load_audio(os.path.join(s1_dir, f"{mix_id}.wav"), sr)
            except Exception:
                skipped += 1
                continue

            raw_enroll = enroll_data[mix_id]["enroll_wav"]
            enrollment_wav = raw_enroll[:enroll_samples]
            if enrollment_wav.shape[-1] < sr * 3:
                enrollment_wav = torch.nn.functional.pad(
                    enrollment_wav, (0, sr * 3 - enrollment_wav.shape[-1])
                )

            if args.noise_ratio > 0 and enroll_data[mix_id]["noise_wav"] is not None:
                enrollment_wav = add_wham_noise(
                    enrollment_wav, enroll_data[mix_id]["noise_wav"], args.noise_ratio
                )

            extracted = extract_speaker(
                model, mixture_wav, enrollment_wav, config, device,
                alpha=args.alpha, chunk_batch_size=args.chunk_batch_size,
            )

            min_len = min(gt_wav.shape[-1], extracted.shape[-1])
            gt_t = gt_wav[:min_len]
            ext_t = extracted.squeeze()[:min_len]

            sisdr_val = si_sdr(ext_t, gt_t)
            pesq_val = try_pesq(ext_t.numpy(), gt_t.numpy(), sr)

            all_results.append({
                "mixture_id": mix_id,
                "enrollment_seconds": enroll_secs,
                "noise_ratio": args.noise_ratio,
                "sisdr_dB": round(sisdr_val, 3),
                "pesq": round(pesq_val, 3) if pesq_val is not None else None,
            })
        if skipped:
            print(f"  Skipped {skipped} corrupt/unreadable files")

        length_results = [r for r in all_results if r["enrollment_seconds"] == enroll_secs]
        sisdrs = [r["sisdr_dB"] for r in length_results]
        pesqs = [r["pesq"] for r in length_results if r["pesq"] is not None]
        print(f"  SI-SDR: mean={np.mean(sisdrs):.2f}, median={np.median(sisdrs):.2f}")
        if pesqs:
            print(f"  PESQ:   mean={np.mean(pesqs):.3f}, median={np.median(pesqs):.3f}")

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {len(all_results)} results to {csv_path}")


    import pandas as pd
    df = pd.read_csv(csv_path)
    agg = df.groupby("enrollment_seconds").agg(
        sisdr_mean=("sisdr_dB", "mean"),
        sisdr_median=("sisdr_dB", "median"),
        pesq_mean=("pesq", "mean"),
        pesq_median=("pesq", "median"),
        n=("sisdr_dB", "count"),
    ).reset_index()

    print(f"\n{'='*65}")
    print(f"  ENROLLMENT LENGTH vs QUALITY -- {noise_label.upper()}")
    print(f"{'='*65}")
    print(f"  {'Length':>6s}  {'N':>5s}  {'SI-SDR Mean':>11s}  {'SI-SDR Med':>10s}  {'PESQ Mean':>9s}  {'PESQ Med':>8s}")
    print("  " + "-" * 58)
    for _, row in agg.iterrows():
        pesq_m = f"{row['pesq_mean']:.3f}" if pd.notna(row["pesq_mean"]) else "N/A"
        pesq_med = f"{row['pesq_median']:.3f}" if pd.notna(row["pesq_median"]) else "N/A"
        print(f"  {row['enrollment_seconds']:>5.1f}s  {int(row['n']):>5d}  "
              f"{row['sisdr_mean']:>10.2f}  {row['sisdr_median']:>10.2f}  "
              f"{pesq_m:>9s}  {pesq_med:>8s}")
    print("=" * 65)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Enrollment Length vs Quality -- {noise_label}\n")
        f.write(f"Samples: {len(selected)}, Seed: {args.seed}, Noise ratio: {args.noise_ratio}\n\n")
        f.write(agg.to_string(index=False))
        f.write("\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
