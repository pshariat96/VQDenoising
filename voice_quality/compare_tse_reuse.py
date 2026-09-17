import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


def si_sdr(estimate, reference):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target
    return (10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )).item()


def try_pesq(est_np, ref_np, sr=16000):
    try:
        from pesq import pesq as pesq_fn
        return pesq_fn(sr, ref_np, est_np, "wb")
    except Exception:
        return None


def try_stoi(est_np, ref_np, sr=16000):
    try:
        from pystoi import stoi
        return stoi(ref_np, est_np, sr, extended=False)
    except Exception:
        return None


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="tse_vs_reuse_results")
    return p.parse_args()


def main():
    args = parse_args()
    sr = 16000

    tse_csv = os.path.join(args.results_dir, "tse_metrics.csv")
    reuse_dir = os.path.join(args.results_dir, "audio", "reuse_enhanced")
    gt_dir = os.path.join(args.results_dir, "audio", "clean_s1")
    mix_dir = os.path.join(args.results_dir, "audio", "noisy_mixture")

    if not os.path.exists(reuse_dir):
        print(f"RE-USE enhanced audio not found at {reuse_dir}")
        print(f"Please run RE-USE on the files in {mix_dir} and save output to {reuse_dir}")
        return

    tse_df = pd.read_csv(tse_csv)
    reuse_files = {f.replace(".wav", "") for f in os.listdir(reuse_dir) if f.endswith(".wav")}
    common = sorted(set(tse_df["mixture_id"]) & reuse_files)
    print(f"Found {len(common)} samples with both TSE and RE-USE results")

    results = []
    for mix_id in tqdm(common, desc="Computing RE-USE metrics"):
        try:
            gt_wav = load_audio(os.path.join(gt_dir, f"{mix_id}.wav"), sr)
            reuse_wav = load_audio(os.path.join(reuse_dir, f"{mix_id}.wav"), sr)

            min_len = min(gt_wav.shape[-1], reuse_wav.shape[-1])
            gt_t = gt_wav[:min_len]
            reuse_t = reuse_wav[:min_len]

            sisdr_reuse = si_sdr(reuse_t, gt_t)
            pesq_reuse = try_pesq(reuse_t.numpy(), gt_t.numpy(), sr)
            stoi_reuse = try_stoi(reuse_t.numpy(), gt_t.numpy(), sr)

            results.append({
                "mixture_id": mix_id,
                "sisdr_reuse": round(sisdr_reuse, 3),
                "pesq_reuse": round(pesq_reuse, 3) if pesq_reuse else None,
                "stoi_reuse": round(stoi_reuse, 4) if stoi_reuse else None,
            })
        except Exception as e:
            print(f"  Error on {mix_id}: {e}")

    reuse_df = pd.DataFrame(results)
    merged = tse_df.merge(reuse_df, on="mixture_id", how="inner")

    out_csv = os.path.join(args.results_dir, "comparison_metrics.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved comparison to {out_csv}")

    print(f"\n{'='*70}")
    print(f"  TSE vs RE-USE COMPARISON ({len(merged)} samples)")
    print(f"{'='*70}")
    for metric, cols in [
        ("SI-SDR (dB)", ("sisdr_mixture", "sisdr_tse", "sisdr_reuse")),
        ("PESQ", ("pesq_mixture", "pesq_tse", "pesq_reuse")),
        ("STOI", ("stoi_mixture", "stoi_tse", "stoi_reuse")),
    ]:
        mix_col, tse_col, reuse_col = cols
        if tse_col not in merged.columns or reuse_col not in merged.columns:
            continue
        mix_vals = merged[mix_col].dropna()
        tse_vals = merged[tse_col].dropna()
        reuse_vals = merged[reuse_col].dropna()
        if len(tse_vals) == 0 or len(reuse_vals) == 0:
            continue
        print(f"\n  {metric}:")
        print(f"    Noisy Mixture: mean={mix_vals.mean():.3f}, median={mix_vals.median():.3f}")
        print(f"    TSE:           mean={tse_vals.mean():.3f}, median={tse_vals.median():.3f}")
        print(f"    RE-USE:        mean={reuse_vals.mean():.3f}, median={reuse_vals.median():.3f}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
