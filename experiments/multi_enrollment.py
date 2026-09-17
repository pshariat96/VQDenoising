import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as torchF
import torchaudio
from tqdm import tqdm

from core.inference import (
    parse_config,
    load_flowtse_model,
    load_t_predicter,
    pad_and_reshape,
    reshape_and_remove_padding,
)
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver


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


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0)


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


def compute_embedding(ecapa, waveform, device):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    with torch.no_grad():
        emb = ecapa(waveform.to(device), aug=False)
    return torchF.normalize(emb, dim=-1).cpu()


def parse_args():
    p = argparse.ArgumentParser(description="Multi-enrollment averaging experiment")
    p.add_argument("--librimix_dir", default=os.environ.get("LIBRIMIX_DIR", "data/Libri2Mix"))
    p.add_argument("--config", default="config/config_FlowTSE_large_noisy.yaml")
    p.add_argument("--output_dir", default="multi_enrollment_results")
    p.add_argument("--n_samples", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = parse_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = config["dataset"]["sample_rate"]

    sep = os.path.join(args.librimix_dir, "sep_clean")
    mix_dir = os.path.join(sep, "mix_clean")
    s1_dir = os.path.join(sep, "s1")

    mix_ids = {f.replace(".wav", "") for f in os.listdir(mix_dir) if f.endswith(".wav")}
    s1_ids = {f.replace(".wav", "") for f in os.listdir(s1_dir) if f.endswith(".wav")}
    valid_ids = sorted(mix_ids & s1_ids)

    spk_map = defaultdict(list)
    for mid in valid_ids:
        spk_map[mid.split("-")[0]].append(mid)

    eligible = []
    enroll_sources = {}
    min_enroll_samples = sr * 6

    for mid in valid_ids:
        spk1 = mid.split("-")[0]
        others = [m for m in spk_map[spk1] if m != mid]
        for oid in others:
            try:
                wav = load_audio(os.path.join(s1_dir, f"{oid}.wav"), sr)
                if wav.shape[-1] >= min_enroll_samples:
                    eligible.append(mid)
                    enroll_sources[mid] = {"id": oid, "wav": wav}
                    break
            except Exception:
                continue

    selected = random.sample(eligible, min(args.n_samples, len(eligible)))
    print(f"Found {len(eligible)} eligible (enroll >= 6s), selected {len(selected)}")

    ckpt_path = config["eval"]["checkpoint"]
    print(f"Loading FlowTSE model...")
    model = load_flowtse_model(ckpt_path, config["model"], device)

    print("Loading ECAPA-TDNN for similarity-based selection...")
    t_predicter = load_t_predicter("t_predictor_noisy.ckpt", {"C": 1024}, device)
    ecapa = t_predicter.ecapa_tdnn
    ecapa.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for mix_id in tqdm(selected, desc="Multi-enrollment"):
        try:
            mixture_wav = load_audio(os.path.join(mix_dir, f"{mix_id}.wav"), sr)
            gt_wav = load_audio(os.path.join(s1_dir, f"{mix_id}.wav"), sr)
            raw_enroll = enroll_sources[mix_id]["wav"]

            enroll_a = raw_enroll[:sr * 3]
            enroll_b = raw_enroll[sr * 3:sr * 6]

            ext_a = extract_speaker(model, mixture_wav, enroll_a, config, device)
            ext_b = extract_speaker(model, mixture_wav, enroll_b, config, device)

            min_len = min(gt_wav.shape[-1], ext_a.shape[-1], ext_b.shape[-1])
            gt_t = gt_wav[:min_len]
            ext_a_t = ext_a.squeeze()[:min_len]
            ext_b_t = ext_b.squeeze()[:min_len]

            avg_ext = (ext_a_t + ext_b_t) / 2.0
            peak = avg_ext.abs().max()
            if peak > 1.0:
                avg_ext = avg_ext / peak

            emb_a = compute_embedding(ecapa, enroll_a, device)
            sim_a = torchF.cosine_similarity(
                emb_a, compute_embedding(ecapa, ext_a_t, device), dim=-1
            ).item()
            sim_b = torchF.cosine_similarity(
                emb_a, compute_embedding(ecapa, ext_b_t, device), dim=-1
            ).item()
            best_ext = ext_a_t if sim_a >= sim_b else ext_b_t

            sisdr_single_a = si_sdr(ext_a_t, gt_t)
            sisdr_single_b = si_sdr(ext_b_t, gt_t)
            sisdr_avg = si_sdr(avg_ext, gt_t)
            sisdr_best = si_sdr(best_ext, gt_t)

            pesq_single_a = try_pesq(ext_a_t.numpy(), gt_t.numpy(), sr)
            pesq_single_b = try_pesq(ext_b_t.numpy(), gt_t.numpy(), sr)
            pesq_avg = try_pesq(avg_ext.numpy(), gt_t.numpy(), sr)
            pesq_best = try_pesq(best_ext.numpy(), gt_t.numpy(), sr)

            results.append({
                "mixture_id": mix_id,
                "sisdr_single_a": round(sisdr_single_a, 3),
                "sisdr_single_b": round(sisdr_single_b, 3),
                "sisdr_avg": round(sisdr_avg, 3),
                "sisdr_best_sim": round(sisdr_best, 3),
                "pesq_single_a": round(pesq_single_a, 3) if pesq_single_a else None,
                "pesq_single_b": round(pesq_single_b, 3) if pesq_single_b else None,
                "pesq_avg": round(pesq_avg, 3) if pesq_avg else None,
                "pesq_best_sim": round(pesq_best, 3) if pesq_best else None,
                "sim_a": round(sim_a, 4),
                "sim_b": round(sim_b, 4),
            })
        except Exception as e:
            print(f"  Error on {mix_id}: {e}")
            continue

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} results to {csv_path}")

    sa = [r["sisdr_single_a"] for r in results]
    sb = [r["sisdr_single_b"] for r in results]
    s_avg = [r["sisdr_avg"] for r in results]
    s_best = [r["sisdr_best_sim"] for r in results]
    pa = [r["pesq_single_a"] for r in results if r["pesq_single_a"]]
    pb = [r["pesq_single_b"] for r in results if r["pesq_single_b"]]
    p_avg = [r["pesq_avg"] for r in results if r["pesq_avg"]]
    p_best = [r["pesq_best_sim"] for r in results if r["pesq_best_sim"]]

    print(f"\n{'='*70}")
    print(f"  MULTI-ENROLLMENT RESULTS ({len(results)} samples)")
    print(f"{'='*70}")
    print(f"  SI-SDR:")
    print(f"    Single A (first 3s):   mean={np.mean(sa):.2f}, median={np.median(sa):.2f}")
    print(f"    Single B (second 3s):  mean={np.mean(sb):.2f}, median={np.median(sb):.2f}")
    print(f"    Average of A+B:        mean={np.mean(s_avg):.2f}, median={np.median(s_avg):.2f}")
    print(f"    Best by similarity:    mean={np.mean(s_best):.2f}, median={np.median(s_best):.2f}")
    if pa:
        print(f"  PESQ:")
        print(f"    Single A:   mean={np.mean(pa):.3f}")
        print(f"    Single B:   mean={np.mean(pb):.3f}")
        print(f"    Average:    mean={np.mean(p_avg):.3f}")
        print(f"    Best by sim: mean={np.mean(p_best):.3f}")
    print(f"{'='*70}")


    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Multi-Enrollment Averaging Experiment\n")
        f.write(f"Samples: {len(results)}, Seed: {args.seed}\n\n")
        f.write(f"SI-SDR:\n")
        f.write(f"  Single A (first 3s):  mean={np.mean(sa):.2f}, median={np.median(sa):.2f}\n")
        f.write(f"  Single B (second 3s): mean={np.mean(sb):.2f}, median={np.median(sb):.2f}\n")
        f.write(f"  Average A+B:          mean={np.mean(s_avg):.2f}, median={np.median(s_avg):.2f}\n")
        f.write(f"  Best by similarity:   mean={np.mean(s_best):.2f}, median={np.median(s_best):.2f}\n")
        if pa:
            f.write(f"\nPESQ:\n")
            f.write(f"  Single A:    mean={np.mean(pa):.3f}\n")
            f.write(f"  Single B:    mean={np.mean(pb):.3f}\n")
            f.write(f"  Average:     mean={np.mean(p_avg):.3f}\n")
            f.write(f"  Best by sim: mean={np.mean(p_best):.3f}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
