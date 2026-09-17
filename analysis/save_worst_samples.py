import os
import random

import numpy as np
import pandas as pd
import torch
import torchaudio

from core.inference import parse_config, load_flowtse_model, pad_and_reshape, reshape_and_remove_padding
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver


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
        random.seed(42)
        start = random.randint(0, wav.shape[-1] - target_len)
        wav = wav[start:start + target_len]
    elif wav.shape[-1] < target_len:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[-1]))
    return wav


def add_wham_noise(clean_wav, noise_wav, noise_ratio):
    if noise_ratio == 0.0:
        return clean_wav.clone()
    n_samples = clean_wav.shape[-1]
    if noise_wav.shape[-1] >= n_samples:
        random.seed(42)
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


def extract_speaker(model, mixture_wav, enrollment_wav, config, device, alpha=0.5):
    sr = config["dataset"]["sample_rate"]
    n_fft = config["dataset"]["n_fft"]
    hop_length = config["dataset"]["hop_length"]
    win_length = config["dataset"]["win_length"]

    mixture_spec = stft_torch(mixture_wav, n_fft, hop_length, win_length).unsqueeze(0)
    enrollment_spec = stft_torch(enrollment_wav, n_fft, hop_length, win_length).unsqueeze(0).to(device)

    frames_per_chunk = sr * 3 // hop_length + 1
    mixture_chunks, orig_len = pad_and_reshape(mixture_spec, frames_per_chunk)
    num_chunks = mixture_chunks.shape[0]

    alpha_grid = torch.tensor([alpha, 1.0], device=device)
    all_outputs = []
    with torch.no_grad():
        for i in range(0, num_chunks, 16):
            batch = mixture_chunks[i:i + 16].to(device)
            bs = batch.shape[0]
            solver = ODESolver(velocity_model=model)
            out = solver.sample(
                time_grid=alpha_grid, x_init=batch.float(),
                method=config["solver"]["method"],
                step_size=config["solver"]["test_step_size"],
                enrollment=enrollment_spec.repeat(bs, 1, 1),
            )
            all_outputs.append(out.cpu())

    source_hat_spec = torch.cat(all_outputs, dim=0)
    source_hat_spec = reshape_and_remove_padding(source_hat_spec, orig_len)
    source_hat = istft_torch(source_hat_spec, n_fft, hop_length, win_length,
                             length=mixture_wav.shape[-1])
    if source_hat.abs().max() > 1.0:
        source_hat = source_hat / source_hat.abs().max()
    return source_hat


def main():
    results_dir = "noisy_enrollment_results"
    librimix_dir = os.environ.get("LIBRIMIX_DIR", "data/Libri2Mix")
    output_dir = os.path.join(results_dir, "worst_5_inspection")

    noisy_csv = os.path.join(results_dir, "noisy_40pct", "metrics.csv")
    clean_csv = os.path.join(results_dir, "clean", "metrics.csv")

    df_noisy = pd.read_csv(noisy_csv)
    df_clean = pd.read_csv(clean_csv)

    worst_5 = df_noisy.nsmallest(5, "sisdr_dB")
    print("Top 5 worst samples (40% noise):")
    print(worst_5[["mixture_id", "enroll_id", "noise_file", "sisdr_dB", "pesq"]].to_string())
    print()

    for _, row in worst_5.iterrows():
        mid = row["mixture_id"]
        clean_row = df_clean[df_clean["mixture_id"] == mid].iloc[0]
        print(f"  {mid}:  clean SI-SDR={clean_row['sisdr_dB']:.2f}  noisy SI-SDR={row['sisdr_dB']:.2f}  delta={row['sisdr_dB'] - clean_row['sisdr_dB']:.2f}")

    config = parse_config("config/config_FlowTSE_large_noisy.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sr = config["dataset"]["sample_rate"]

    ckpt_path = config["eval"]["checkpoint"]
    print(f"\nLoading model from {ckpt_path}")
    model = load_flowtse_model(ckpt_path, config["model"], device)

    sep = os.path.join(librimix_dir, "sep_clean")
    mix_dir = os.path.join(sep, "mix_clean")
    s1_dir = os.path.join(sep, "s1")
    noise_dir = os.path.join(sep, "noise")

    os.makedirs(output_dir, exist_ok=True)

    for rank, (_, row) in enumerate(worst_5.iterrows(), 1):
        mid = row["mixture_id"]
        enroll_id = row["enroll_id"]
        noise_file = row["noise_file"]

        sample_dir = os.path.join(output_dir, f"rank{rank}_{mid}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n[{rank}/5] {mid}")
        print(f"  enroll_id={enroll_id}, noise_file={noise_file}")
        print(f"  40% SI-SDR={row['sisdr_dB']:.2f}")

        mixture_wav = load_audio(os.path.join(mix_dir, f"{mid}.wav"), sr)
        gt_wav = load_audio(os.path.join(s1_dir, f"{mid}.wav"), sr)
        enroll_raw = load_audio(os.path.join(s1_dir, f"{enroll_id}.wav"), sr)
        noise_wav = load_audio(os.path.join(noise_dir, noise_file), sr)

        enrollment_clean = prepare_enrollment(enroll_raw, sr, 3.0)
        enrollment_noisy = add_wham_noise(enrollment_clean, noise_wav, 0.40)

        print("  Extracting with clean enrollment...")
        extracted_clean = extract_speaker(model, mixture_wav, enrollment_clean, config, device)

        print("  Extracting with noisy enrollment...")
        extracted_noisy = extract_speaker(model, mixture_wav, enrollment_noisy, config, device)

        torchaudio.save(os.path.join(sample_dir, "1_mixture.wav"), mixture_wav.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "2_ground_truth_s1.wav"), gt_wav.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "3_enrollment_clean.wav"), enrollment_clean.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "4_enrollment_noisy_40pct.wav"), enrollment_noisy.unsqueeze(0), sr)
        torchaudio.save(os.path.join(sample_dir, "5_extracted_clean_enroll.wav"), extracted_clean, sr)
        torchaudio.save(os.path.join(sample_dir, "6_extracted_noisy_enroll.wav"), extracted_noisy, sr)

        info_lines = [
            f"Mixture ID: {mid}",
            f"Enrollment ID: {enroll_id}",
            f"Noise file: {noise_file}",
            f"SI-SDR (clean enrollment): {df_clean[df_clean['mixture_id']==mid].iloc[0]['sisdr_dB']:.3f} dB",
            f"SI-SDR (40% noisy enrollment): {row['sisdr_dB']:.3f} dB",
            f"PESQ (40% noisy): {row['pesq']}",
        ]
        with open(os.path.join(sample_dir, "info.txt"), "w") as f:
            f.write("\n".join(info_lines))

        print(f"  Saved to {sample_dir}/")

    print(f"\nAll 5 worst samples saved to {output_dir}/")


if __name__ == "__main__":
    main()
