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


def try_stoi(estimate_np, reference_np, sr=16000):
    try:
        from pystoi import stoi
        return stoi(reference_np, estimate_np, sr, extended=False)
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


def parse_args():
    p = argparse.ArgumentParser(description="TSE vs RE-USE comparison (TSE side)")
    p.add_argument("--librimix_dir", default=os.environ.get("LIBRIMIX_DIR", "data/Libri2Mix"))
    p.add_argument("--config", default="config/config_FlowTSE_large_noisy.yaml")
    p.add_argument("--output_dir", default="tse_vs_reuse_results")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = parse_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = config["dataset"]["sample_rate"]

    sep_noisy = os.path.join(args.librimix_dir, "sep_clean")
    mix_dir = os.path.join(sep_noisy, "mix_both")
    s1_dir = os.path.join(sep_noisy, "s1")
    s2_dir = os.path.join(sep_noisy, "s2")
    noise_dir = os.path.join(sep_noisy, "noise")

    if not os.path.exists(mix_dir):
        print(f"mix_both not found, falling back to mix_clean")
        mix_dir = os.path.join(sep_noisy, "mix_clean")

    mix_ids = {f.replace(".wav", "") for f in os.listdir(mix_dir) if f.endswith(".wav")}
    s1_ids = {f.replace(".wav", "") for f in os.listdir(s1_dir) if f.endswith(".wav")}
    valid_ids = sorted(mix_ids & s1_ids)

    spk_map = defaultdict(list)
    for mid in valid_ids:
        spk_map[mid.split("-")[0]].append(mid)

    eligible = [mid for mid in valid_ids if len(spk_map[mid.split("-")[0]]) >= 2]
    selected = random.sample(eligible, min(args.n_samples, len(eligible)))
    print(f"Selected {len(selected)} samples for comparison")

    enroll_map = {}
    for mix_id in selected:
        spk1 = mix_id.split("-")[0]
        others = [m for m in spk_map[spk1] if m != mix_id]
        enroll_map[mix_id] = random.choice(others)

    ckpt_path = config["eval"]["checkpoint"]
    print(f"Loading FlowTSE model from {ckpt_path}")
    model = load_flowtse_model(ckpt_path, config["model"], device)

    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "noisy_mixture"), exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "clean_s1"), exist_ok=True)
    os.makedirs(os.path.join(audio_dir, "tse_extracted"), exist_ok=True)

    results = []
    for mix_id in tqdm(selected, desc="Running TSE"):
        try:
            mixture_wav = load_audio(os.path.join(mix_dir, f"{mix_id}.wav"), sr)
            gt_wav = load_audio(os.path.join(s1_dir, f"{mix_id}.wav"), sr)
            enroll_id = enroll_map[mix_id]
            enroll_wav = load_audio(os.path.join(s1_dir, f"{enroll_id}.wav"), sr)

            if enroll_wav.shape[-1] > sr * 3:
                enroll_wav = enroll_wav[:sr * 3]
            elif enroll_wav.shape[-1] < sr * 3:
                enroll_wav = torch.nn.functional.pad(
                    enroll_wav, (0, sr * 3 - enroll_wav.shape[-1])
                )

            extracted = extract_speaker(model, mixture_wav, enroll_wav, config, device)

            min_len = min(gt_wav.shape[-1], extracted.shape[-1], mixture_wav.shape[-1])
            gt_t = gt_wav[:min_len]
            ext_t = extracted.squeeze()[:min_len]
            mix_t = mixture_wav[:min_len]

            sisdr_tse = si_sdr(ext_t, gt_t)
            sisdr_mix = si_sdr(mix_t, gt_t)
            pesq_tse = try_pesq(ext_t.numpy(), gt_t.numpy(), sr)
            pesq_mix = try_pesq(mix_t.numpy(), gt_t.numpy(), sr)
            stoi_tse = try_stoi(ext_t.numpy(), gt_t.numpy(), sr)
            stoi_mix = try_stoi(mix_t.numpy(), gt_t.numpy(), sr)

            torchaudio.save(os.path.join(audio_dir, "noisy_mixture", f"{mix_id}.wav"),
                           mixture_wav.unsqueeze(0), sr)
            torchaudio.save(os.path.join(audio_dir, "clean_s1", f"{mix_id}.wav"),
                           gt_wav.unsqueeze(0), sr)
            torchaudio.save(os.path.join(audio_dir, "tse_extracted", f"{mix_id}.wav"),
                           extracted.unsqueeze(0) if extracted.dim() == 1 else extracted, sr)

            results.append({
                "mixture_id": mix_id,
                "sisdr_mixture": round(sisdr_mix, 3),
                "sisdr_tse": round(sisdr_tse, 3),
                "pesq_mixture": round(pesq_mix, 3) if pesq_mix else None,
                "pesq_tse": round(pesq_tse, 3) if pesq_tse else None,
                "stoi_mixture": round(stoi_mix, 4) if stoi_mix else None,
                "stoi_tse": round(stoi_tse, 4) if stoi_tse else None,
            })
        except Exception as e:
            print(f"  Error on {mix_id}: {e}")
            continue

    csv_path = os.path.join(args.output_dir, "tse_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} TSE results to {csv_path}")

    sisdrs_mix = [r["sisdr_mixture"] for r in results]
    sisdrs_tse = [r["sisdr_tse"] for r in results]
    pesqs_tse = [r["pesq_tse"] for r in results if r["pesq_tse"] is not None]
    pesqs_mix = [r["pesq_mixture"] for r in results if r["pesq_mixture"] is not None]

    print(f"\n{'='*60}")
    print(f"  TSE RESULTS (before RE-USE comparison)")
    print(f"{'='*60}")
    print(f"  Samples: {len(results)}")
    print(f"  SI-SDR Mixture:   mean={np.mean(sisdrs_mix):.2f}, median={np.median(sisdrs_mix):.2f}")
    print(f"  SI-SDR TSE:       mean={np.mean(sisdrs_tse):.2f}, median={np.median(sisdrs_tse):.2f}")
    print(f"  SI-SDR Improvement: {np.mean(sisdrs_tse) - np.mean(sisdrs_mix):.2f} dB")
    if pesqs_tse:
        print(f"  PESQ Mixture:     mean={np.mean(pesqs_mix):.3f}")
        print(f"  PESQ TSE:         mean={np.mean(pesqs_tse):.3f}")
    print(f"{'='*60}")

    print(f"\nAudio files saved to {audio_dir}/")
    print(f"  noisy_mixture/  -- {len(results)} input files for RE-USE")
    print(f"  clean_s1/       -- ground truth for evaluation")
    print(f"  tse_extracted/  -- TSE outputs")
    print(f"\nNext step: Upload noisy_mixture/ to Colab, run RE-USE, download enhanced/,")
    print(f"then run: python3 compare_tse_reuse.py --results_dir {args.output_dir}")


if __name__ == "__main__":
    main()
