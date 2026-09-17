import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from core.inference import (
    load_flowtse_model,
    load_t_predicter,
    load_audio,
    pad_and_reshape,
    reshape_and_remove_padding,
    estimate_alpha_per_chunk,
    parse_config,
)
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver


DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
CONFIG_PATH = "config/config_FlowTSE_large_noisy.yaml"
ORIGINAL_CKPT = "t_predictor_noisy.ckpt"
FINETUNED_CKPT = "t_predictor_target_absent_best.ckpt"
OUTPUT_DIR = "phase1b_eval_results"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

RECORDINGS = [
    {"file_id": "DH_EVAL_0011", "speakers": ["speaker27", "speaker28"]},
    {"file_id": "DH_EVAL_0012", "speakers": ["speaker29", "speaker30"]},
]


def parse_rttm(rttm_path):
    segments = defaultdict(list)
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            segments[speaker_id].append((start, start + duration))
    return dict(segments)


def chunk_has_target(chunk_start, chunk_end, speaker_segments, min_overlap_ratio=0.1):
    total_overlap = 0.0
    for seg_start, seg_end in speaker_segments:
        overlap_start = max(chunk_start, seg_start)
        overlap_end = min(chunk_end, seg_end)
        if overlap_end > overlap_start:
            total_overlap += overlap_end - overlap_start
    overlap_ratio = total_overlap / (chunk_end - chunk_start)
    return overlap_ratio >= min_overlap_ratio


def run_extraction(model, t_predicter, mixture_wav, enrollment_wav, config, device):
    sr = config["dataset"]["sample_rate"]
    n_fft = config["dataset"]["n_fft"]
    hop_length = config["dataset"]["hop_length"]
    win_length = config["dataset"]["win_length"]

    chunk_alphas = estimate_alpha_per_chunk(
        t_predicter, mixture_wav, enrollment_wav, CHUNK_SAMPLES, device
    )

    mixture_spec = stft_torch(mixture_wav, n_fft, hop_length, win_length).unsqueeze(0)
    enrollment_spec = stft_torch(enrollment_wav, n_fft, hop_length, win_length).unsqueeze(0).to(device)

    frames_per_chunk = CHUNK_SAMPLES // hop_length + 1
    mixture_chunks, orig_spec_len = pad_and_reshape(mixture_spec, frames_per_chunk)
    num_chunks = mixture_chunks.shape[0]

    all_outputs = []
    with torch.no_grad():
        for i in tqdm(range(num_chunks), desc="  Extracting", leave=False):
            chunk = mixture_chunks[i:i + 1].to(device)
            alpha_val = chunk_alphas[i] if i < len(chunk_alphas) else chunk_alphas[-1]

            solver = ODESolver(velocity_model=model)
            alpha_grid = torch.tensor([alpha_val, 1.0], device=device)
            out = solver.sample(
                time_grid=alpha_grid,
                x_init=chunk.float(),
                method="euler",
                step_size=1.0,
                enrollment=enrollment_spec,
            )
            all_outputs.append(out.cpu())

    source_hat_spec = torch.cat(all_outputs, dim=0)
    source_hat_spec = reshape_and_remove_padding(source_hat_spec, orig_spec_len)

    source_hat = istft_torch(
        source_hat_spec, n_fft, hop_length, win_length,
        length=mixture_wav.shape[-1],
    )

    max_val = source_hat.abs().max()
    if max_val > 1.0:
        source_hat = source_hat / max_val

    return source_hat, chunk_alphas


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = parse_config(CONFIG_PATH)
    t_pred_config = config["t_predicter"]

    print("Loading FlowTSE (UDiT) model...")
    udit = load_flowtse_model(config["eval"]["checkpoint"], config["model"], device)

    print(f"Loading ORIGINAL T-Predictor: {ORIGINAL_CKPT}")
    t_pred_orig = load_t_predicter(ORIGINAL_CKPT, t_pred_config, device)

    print(f"Loading FINE-TUNED T-Predictor: {FINETUNED_CKPT}")
    t_pred_ft = load_t_predicter(FINETUNED_CKPT, t_pred_config, device)

    summary_lines = ["Phase 1B Evaluation: Original vs Fine-Tuned T-Predictor\n"]
    all_results = []

    for rec in RECORDINGS:
        file_id = rec["file_id"]
        rttm_path = os.path.join(DIHARD_DIR, "rttm", f"{file_id}.rttm")
        flac_path = os.path.join(DIHARD_DIR, "flac", f"{file_id}.flac")

        print(f"\n{'='*60}")
        print(f"Recording: {file_id}")
        print(f"{'='*60}")

        speaker_segments = parse_rttm(rttm_path)

        for speaker in rec["speakers"]:
            print(f"\n  Speaker: {speaker}")

            enroll_path = os.path.join("dihard_results", file_id, speaker, "enrollment.wav")
            if not os.path.exists(enroll_path):
                print(f"    Enrollment not found, skipping")
                continue

            spk_dir = os.path.join(OUTPUT_DIR, file_id, speaker)
            os.makedirs(spk_dir, exist_ok=True)

            enrollment_wav = load_audio(enroll_path, SAMPLE_RATE)
            max_enroll = SAMPLE_RATE * 3
            if enrollment_wav.shape[-1] > max_enroll:
                enrollment_wav = enrollment_wav[:max_enroll]
            elif enrollment_wav.shape[-1] < max_enroll:
                enrollment_wav = F.pad(enrollment_wav, (0, max_enroll - enrollment_wav.shape[-1]))

            mixture_wav = load_audio(flac_path, SAMPLE_RATE)

            torchaudio.save(os.path.join(spk_dir, "mixture.wav"), mixture_wav.unsqueeze(0), SAMPLE_RATE)
            torchaudio.save(os.path.join(spk_dir, "enrollment.wav"), enrollment_wav.unsqueeze(0), SAMPLE_RATE)

            print(f"    Running with ORIGINAL T-Predictor...")
            ext_orig, alphas_orig = run_extraction(udit, t_pred_orig, mixture_wav, enrollment_wav, config, device)
            torchaudio.save(os.path.join(spk_dir, "extracted_original.wav"), ext_orig, SAMPLE_RATE)

            print(f"    Running with FINE-TUNED T-Predictor...")
            ext_ft, alphas_ft = run_extraction(udit, t_pred_ft, mixture_wav, enrollment_wav, config, device)
            torchaudio.save(os.path.join(spk_dir, "extracted_finetuned.wav"), ext_ft, SAMPLE_RATE)

            target_segs = speaker_segments.get(speaker, [])
            num_chunks = len(alphas_orig)
            labels = []
            for i in range(num_chunks):
                cs = i * CHUNK_SECONDS
                ce = cs + CHUNK_SECONDS
                labels.append(chunk_has_target(cs, ce, target_segs))

            present_idx = [i for i, l in enumerate(labels) if l]
            absent_idx = [i for i, l in enumerate(labels) if not l]

            alphas_orig_arr = np.array(alphas_orig[:num_chunks])
            alphas_ft_arr = np.array(alphas_ft[:num_chunks])
            labels_arr = np.array(labels)

            orig_present_mean = alphas_orig_arr[labels_arr].mean() if labels_arr.any() else 0
            orig_absent_mean = alphas_orig_arr[~labels_arr].mean() if (~labels_arr).any() else 0
            ft_present_mean = alphas_ft_arr[labels_arr].mean() if labels_arr.any() else 0
            ft_absent_mean = alphas_ft_arr[~labels_arr].mean() if (~labels_arr).any() else 0

            stats = (
                f"\n  {file_id} / {speaker}:\n"
                f"    Chunks: {num_chunks} total, {len(present_idx)} present, {len(absent_idx)} absent\n"
                f"    Original  -- present alpha: {orig_present_mean:.4f}, absent alpha: {orig_absent_mean:.4f}\n"
                f"    Finetuned -- present alpha: {ft_present_mean:.4f}, absent alpha: {ft_absent_mean:.4f}\n"
            )
            print(stats)
            summary_lines.append(stats)

            all_results.append({
                "file_id": file_id, "speaker": speaker,
                "num_chunks": num_chunks,
                "n_present": len(present_idx), "n_absent": len(absent_idx),
                "orig_present_alpha": orig_present_mean,
                "orig_absent_alpha": orig_absent_mean,
                "ft_present_alpha": ft_present_mean,
                "ft_absent_alpha": ft_absent_mean,
                "alphas_orig": alphas_orig_arr,
                "alphas_ft": alphas_ft_arr,
                "labels": labels_arr,
            })

    if all_results:
        all_orig_present = np.concatenate([r["alphas_orig"][r["labels"]] for r in all_results])
        all_orig_absent = np.concatenate([r["alphas_orig"][~r["labels"]] for r in all_results])
        all_ft_present = np.concatenate([r["alphas_ft"][r["labels"]] for r in all_results])
        all_ft_absent = np.concatenate([r["alphas_ft"][~r["labels"]] for r in all_results])

        summary_lines.append(f"\n{'='*60}")
        summary_lines.append(f"OVERALL (all speakers combined):")
        summary_lines.append(f"  Original  -- present: {all_orig_present.mean():.4f}, absent: {all_orig_absent.mean():.4f}")
        summary_lines.append(f"  Finetuned -- present: {all_ft_present.mean():.4f}, absent: {all_ft_absent.mean():.4f}")
        summary_lines.append(f"  Improvement in absent suppression: {all_orig_absent.mean():.4f} -> {all_ft_absent.mean():.4f}")

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
