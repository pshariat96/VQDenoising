import argparse
import csv
import os
import random
from collections import defaultdict

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
    si_sdr_val = 10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )
    return si_sdr_val.item()


def try_pesq(estimate_np, reference_np, sr=16000):
    try:
        from pesq import pesq as pesq_fn
        return pesq_fn(sr, reference_np, estimate_np, "wb")
    except Exception:
        return None


def discover_samples(librimix_dir):
    sep = os.path.join(librimix_dir, "sep_clean")
    mix_dir = os.path.join(sep, "mix_clean")
    s1_dir = os.path.join(sep, "s1")
    s2_dir = os.path.join(sep, "s2")

    mix_ids = {f.replace(".wav", "") for f in os.listdir(mix_dir) if f.endswith(".wav")}
    s1_ids = {f.replace(".wav", "") for f in os.listdir(s1_dir) if f.endswith(".wav")}
    s2_ids = {f.replace(".wav", "") for f in os.listdir(s2_dir) if f.endswith(".wav")}

    valid = sorted(mix_ids & s1_ids & s2_ids)
    return valid, mix_dir, s1_dir, s2_dir


def group_by_speaker(mixture_ids):
    spk_map = defaultdict(list)
    for mid in mixture_ids:
        spk1 = mid.split("-")[0]
        spk_map[spk1].append(mid)
    return spk_map


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0)


def prepare_enrollment(wav, sr=16000, duration=3.0):
    target_len = int(sr * duration)
    if wav.shape[-1] > target_len:
        start = random.randint(0, wav.shape[-1] - target_len)
        wav = wav[start : start + target_len]
    elif wav.shape[-1] < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[-1]))
    return wav


def extract_speaker(model, mixture_wav, enrollment_wav, config, device,
                    alpha=0.5, chunk_batch_size=16):
    sr = config["dataset"]["sample_rate"]
    n_fft = config["dataset"]["n_fft"]
    hop_length = config["dataset"]["hop_length"]
    win_length = config["dataset"]["win_length"]

    mixture_spec = stft_torch(mixture_wav, n_fft, hop_length, win_length).unsqueeze(0)
    enrollment_spec = (
        stft_torch(enrollment_wav, n_fft, hop_length, win_length)
        .unsqueeze(0)
        .to(device)
    )

    frames_per_chunk = sr * 3 // hop_length + 1
    mixture_chunks, orig_len = pad_and_reshape(mixture_spec, frames_per_chunk)
    num_chunks = mixture_chunks.shape[0]

    solver_method = config["solver"]["method"]
    solver_step = config["solver"]["test_step_size"]
    alpha_grid = torch.tensor([alpha, 1.0], device=device)

    all_outputs = []
    with torch.no_grad():
        for i in range(0, num_chunks, chunk_batch_size):
            batch = mixture_chunks[i : i + chunk_batch_size].to(device)
            bs = batch.shape[0]
            solver = ODESolver(velocity_model=model)
            out = solver.sample(
                time_grid=alpha_grid,
                x_init=batch.float(),
                method=solver_method,
                step_size=solver_step,
                enrollment=enrollment_spec.repeat(bs, 1, 1),
            )
            all_outputs.append(out.cpu())

    source_hat_spec = torch.cat(all_outputs, dim=0)
    source_hat_spec = reshape_and_remove_padding(source_hat_spec, orig_len)

    source_hat = istft_torch(
        source_hat_spec, n_fft, hop_length, win_length,
        length=mixture_wav.shape[-1],
    )

    max_val = source_hat.abs().max()
    if max_val > 1.0:
        source_hat = source_hat / max_val
    return source_hat


def parse_args():
    p = argparse.ArgumentParser(description="Wrong-enrollment experiment")
    p.add_argument("--librimix_dir", required=True,
                   help="Path to Libri2Mix_Official_Test root")
    p.add_argument("--config", default="config/config_FlowTSE_large_noisy.yaml")
    p.add_argument("--output_dir", default="wrong_enrollment_results")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--chunk_batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = parse_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = config["dataset"]["sample_rate"]

    valid_ids, mix_dir, s1_dir, s2_dir = discover_samples(args.librimix_dir)
    spk_map = group_by_speaker(valid_ids)
    all_speakers = sorted(spk_map.keys())

    print(f"Found {len(valid_ids)} complete samples across {len(all_speakers)} speakers")

    eligible = [mid for mid in valid_ids if len(spk_map[mid.split('-')[0]]) >= 2]
    print(f"Eligible samples (speaker has >=2 utterances): {len(eligible)}")

    selected = random.sample(eligible, min(args.n_samples, len(eligible)))
    print(f"Selected {len(selected)} samples for experiment\n")

    ckpt_path = config["eval"]["checkpoint"]
    print(f"Loading FlowTSE model from {ckpt_path}")
    model = load_flowtse_model(ckpt_path, config["model"], device)

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for idx, mix_id in enumerate(tqdm(selected, desc="Samples")):
        spk1 = mix_id.split("-")[0]
        spk2 = mix_id.split("_")[1].split("-")[0]

        sample_dir = os.path.join(args.output_dir, mix_id)
        os.makedirs(sample_dir, exist_ok=True)

        mixture_wav = load_audio(os.path.join(mix_dir, f"{mix_id}.wav"), sr)
        gt_wav = load_audio(os.path.join(s1_dir, f"{mix_id}.wav"), sr)

        same_spk_others = [m for m in spk_map[spk1] if m != mix_id]
        correct_enroll_id = random.choice(same_spk_others)
        correct_enroll_wav = prepare_enrollment(
            load_audio(os.path.join(s1_dir, f"{correct_enroll_id}.wav"), sr), sr
        )

        wrong_spk = random.choice([s for s in all_speakers if s != spk1])
        wrong_enroll_id = random.choice(spk_map[wrong_spk])
        wrong_enroll_wav = prepare_enrollment(
            load_audio(os.path.join(s1_dir, f"{wrong_enroll_id}.wav"), sr), sr
        )

        extracted_correct = extract_speaker(
            model, mixture_wav, correct_enroll_wav, config, device,
            alpha=args.alpha, chunk_batch_size=args.chunk_batch_size,
        )

        extracted_wrong = extract_speaker(
            model, mixture_wav, wrong_enroll_wav, config, device,
            alpha=args.alpha, chunk_batch_size=args.chunk_batch_size,
        )

        min_len = min(gt_wav.shape[-1], extracted_correct.shape[-1], extracted_wrong.shape[-1])
        gt_trimmed = gt_wav[:min_len]
        mix_trimmed = mixture_wav[:min_len]

        sisdr_correct = si_sdr(extracted_correct.squeeze()[:min_len], gt_trimmed)
        sisdr_wrong = si_sdr(extracted_wrong.squeeze()[:min_len], gt_trimmed)
        sisdr_mixture = si_sdr(mix_trimmed, gt_trimmed)

        pesq_correct = try_pesq(
            extracted_correct.squeeze()[:min_len].numpy(),
            gt_trimmed.numpy(), sr,
        )
        pesq_wrong = try_pesq(
            extracted_wrong.squeeze()[:min_len].numpy(),
            gt_trimmed.numpy(), sr,
        )

        results.append({
            "mixture_id": mix_id,
            "speaker1": spk1,
            "speaker2": spk2,
            "correct_enroll_id": correct_enroll_id,
            "correct_enroll_spk": spk1,
            "wrong_enroll_id": wrong_enroll_id,
            "wrong_enroll_spk": wrong_spk,
            "sisdr_mixture_dB": round(sisdr_mixture, 2),
            "sisdr_correct_dB": round(sisdr_correct, 2),
            "sisdr_wrong_dB": round(sisdr_wrong, 2),
            "sisdr_improvement_correct_dB": round(sisdr_correct - sisdr_mixture, 2),
            "sisdr_improvement_wrong_dB": round(sisdr_wrong - sisdr_mixture, 2),
            "pesq_correct": round(pesq_correct, 3) if pesq_correct is not None else None,
            "pesq_wrong": round(pesq_wrong, 3) if pesq_wrong is not None else None,
        })

        torchaudio.save(os.path.join(sample_dir, "mixture.wav"),
                        mixture_wav.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "ground_truth_s1.wav"),
                        gt_wav.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "correct_enrollment.wav"),
                        correct_enroll_wav.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "wrong_enrollment.wav"),
                        wrong_enroll_wav.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "extracted_correct.wav"),
                        extracted_correct, sr)
        torchaudio.save(os.path.join(sample_dir, "extracted_wrong.wav"),
                        extracted_wrong, sr)

        tqdm.write(
            f"  [{idx+1}/{len(selected)}] {mix_id}  "
            f"SI-SDR correct={sisdr_correct:+.1f} dB  "
            f"wrong={sisdr_wrong:+.1f} dB  "
            f"(mixture={sisdr_mixture:+.1f} dB)"
        )

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved per-sample metrics to {csv_path}")

    sisdr_c = [r["sisdr_correct_dB"] for r in results]
    sisdr_w = [r["sisdr_wrong_dB"] for r in results]
    sisdr_m = [r["sisdr_mixture_dB"] for r in results]
    imp_c = [r["sisdr_improvement_correct_dB"] for r in results]
    imp_w = [r["sisdr_improvement_wrong_dB"] for r in results]

    pesq_c = [r["pesq_correct"] for r in results if r["pesq_correct"] is not None]
    pesq_w = [r["pesq_wrong"] for r in results if r["pesq_wrong"] is not None]

    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def median(vals):
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return float("nan")
        return (s[n // 2] + s[(n - 1) // 2]) / 2

    summary_lines = [
        "=" * 60,
        "  WRONG ENROLLMENT EXPERIMENT -- SUMMARY",
        "=" * 60,
        f"  Samples evaluated:  {len(results)}",
        f"  Alpha (fixed):      {args.alpha}",
        f"  Solver:             {config['solver']['method']}, step={config['solver']['test_step_size']}",
        "",
        "  SI-SDR (dB)                   Mean      Median    Min       Max",
        "  " + "-" * 56,
        f"  Mixture (input)          {mean(sisdr_m):>8.2f}  {median(sisdr_m):>8.2f}  {min(sisdr_m):>8.2f}  {max(sisdr_m):>8.2f}",
        f"  Correct enrollment       {mean(sisdr_c):>8.2f}  {median(sisdr_c):>8.2f}  {min(sisdr_c):>8.2f}  {max(sisdr_c):>8.2f}",
        f"  Wrong enrollment         {mean(sisdr_w):>8.2f}  {median(sisdr_w):>8.2f}  {min(sisdr_w):>8.2f}  {max(sisdr_w):>8.2f}",
        "",
        f"  SI-SDR Improvement (dB)",
        f"  Correct enrollment       {mean(imp_c):>8.2f}  {median(imp_c):>8.2f}  {min(imp_c):>8.2f}  {max(imp_c):>8.2f}",
        f"  Wrong enrollment         {mean(imp_w):>8.2f}  {median(imp_w):>8.2f}  {min(imp_w):>8.2f}  {max(imp_w):>8.2f}",
    ]

    if pesq_c:
        summary_lines += [
            "",
            "  PESQ (wideband)          Mean      Median    Min       Max",
            "  " + "-" * 56,
            f"  Correct enrollment       {mean(pesq_c):>8.3f}  {median(pesq_c):>8.3f}  {min(pesq_c):>8.3f}  {max(pesq_c):>8.3f}",
            f"  Wrong enrollment         {mean(pesq_w):>8.3f}  {median(pesq_w):>8.3f}  {min(pesq_w):>8.3f}  {max(pesq_w):>8.3f}",
        ]

    wins_correct = sum(1 for c, w in zip(sisdr_c, sisdr_w) if c > w)
    ties = sum(1 for c, w in zip(sisdr_c, sisdr_w) if abs(c - w) < 0.1)
    summary_lines += [
        "",
        f"  Correct > Wrong:    {wins_correct}/{len(results)} samples ({100*wins_correct/len(results):.0f}%)",
        f"  Margin < 0.1 dB:    {ties}/{len(results)} samples",
        f"  Avg gap:            {mean(imp_c) - mean(imp_w):+.2f} dB in favor of correct enrollment",
        "=" * 60,
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
