import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from core.inference import parse_config, load_flowtse_model, pad_and_reshape, reshape_and_remove_padding
from models.t_predicter import TPredicter
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver

DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
CONFIG_PATH = "config/config_FlowTSE_large_noisy.yaml"
OUTPUT_DIR = "multi_enrollment_dihard_results"
SAMPLE_RATE = 16000
CHUNK_SEC = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC

RECORDINGS = {
    "DH_EVAL_0011": ["speaker27", "speaker28"],
    "DH_EVAL_0012": ["speaker29", "speaker30"],
}


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


def load_t_predicter(ckpt_path, model_config, device):
    t_predicter = TPredicter(**model_config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    t_predicter.load_state_dict(state_dict)
    return t_predicter.eval().to(device)


def estimate_alpha_per_chunk(t_predicter, mixture_wav, enrollment_wav, device):
    total = mixture_wav.shape[-1]
    alphas = []
    with torch.no_grad():
        for start in range(0, total, CHUNK_SAMPLES):
            chunk = mixture_wav[start:start + CHUNK_SAMPLES]
            enr = enrollment_wav[:CHUNK_SAMPLES]
            if chunk.shape[-1] < CHUNK_SAMPLES:
                chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[-1]))
            if enr.shape[-1] < CHUNK_SAMPLES:
                enr = torch.nn.functional.pad(enr, (0, CHUNK_SAMPLES - enr.shape[-1]))
            alpha = t_predicter(
                chunk.unsqueeze(0).to(device),
                enr.unsqueeze(0).to(device),
            ).item()
            alphas.append(alpha)
    return alphas


def run_full_extraction(model, t_predicter, mixture_wav, enrollment_wav, config, device):
    n_fft = config["dataset"]["n_fft"]
    hop_length = config["dataset"]["hop_length"]
    win_length = config["dataset"]["win_length"]

    chunk_alphas = estimate_alpha_per_chunk(t_predicter, mixture_wav, enrollment_wav, device)

    mixture_spec = stft_torch(mixture_wav, n_fft, hop_length, win_length).unsqueeze(0)
    enrollment_spec = stft_torch(enrollment_wav, n_fft, hop_length, win_length).unsqueeze(0).to(device)

    frames_per_chunk = CHUNK_SAMPLES // hop_length + 1
    mixture_chunks, orig_spec_len = pad_and_reshape(mixture_spec, frames_per_chunk)
    num_chunks = mixture_chunks.shape[0]

    all_outputs = []
    with torch.no_grad():
        for i in tqdm(range(num_chunks), desc="    Extracting", leave=False):
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
    source_hat = istft_torch(source_hat_spec, n_fft, hop_length, win_length,
                             length=mixture_wav.shape[-1])
    max_val = source_hat.abs().max()
    if max_val > 1.0:
        source_hat = source_hat / max_val
    return source_hat


def extract_enrollment_clips(wav, sr, segments):
    seg_list = [(int(s * sr), int(e * sr), e - s) for s, e in segments]
    seg_list.sort(key=lambda x: -x[2])

    for start, end, dur in seg_list:
        if dur >= 6.0:
            clip_a = wav[start:start + sr * CHUNK_SEC]
            clip_b = wav[start + sr * CHUNK_SEC:start + sr * CHUNK_SEC * 2]
            if clip_b.shape[-1] >= sr * CHUNK_SEC:
                return clip_a, clip_b

    if len(seg_list) >= 2:
        s1, e1, _ = seg_list[0]
        s2, e2, _ = seg_list[1]
        clip_a = wav[s1:s1 + sr * CHUNK_SEC]
        clip_b = wav[s2:s2 + sr * CHUNK_SEC]
        if clip_a.shape[-1] < sr * CHUNK_SEC:
            clip_a = torch.nn.functional.pad(clip_a, (0, sr * CHUNK_SEC - clip_a.shape[-1]))
        if clip_b.shape[-1] < sr * CHUNK_SEC:
            clip_b = torch.nn.functional.pad(clip_b, (0, sr * CHUNK_SEC - clip_b.shape[-1]))
        return clip_a, clip_b

    s1, e1, _ = seg_list[0]
    clip = wav[s1:s1 + sr * CHUNK_SEC]
    if clip.shape[-1] < sr * CHUNK_SEC:
        clip = torch.nn.functional.pad(clip, (0, sr * CHUNK_SEC - clip.shape[-1]))
    return clip, clip


def si_sdr(estimate, reference):
    ref = reference.float()
    est = estimate.float()
    dot = torch.dot(est, ref)
    s_ref = dot / (torch.dot(ref, ref) + 1e-8) * ref
    noise = est - s_ref
    return 10 * torch.log10(torch.dot(s_ref, s_ref) / (torch.dot(noise, noise) + 1e-8) + 1e-8).item()


def main():
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = parse_config(CONFIG_PATH)
    device = "cpu"
    sr = config["dataset"]["sample_rate"]

    print("Loading FlowTSE (UDiT)...")
    model = load_flowtse_model(config["eval"]["checkpoint"], config["model"], device)

    print("Loading T-Predictor...")
    t_predicter = load_t_predicter("t_predictor_noisy.ckpt", config["t_predicter"], device)

    all_results = []

    for rec_id, speakers in RECORDINGS.items():
        audio_path = os.path.join(DIHARD_DIR, f"flac/{rec_id}.flac")
        rttm_path = os.path.join(DIHARD_DIR, f"rttm/{rec_id}.rttm")

        if not os.path.exists(audio_path):
            print(f"Skipping {rec_id}: audio not found")
            continue

        wav, file_sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)

        rttm = parse_rttm(rttm_path)
        print(f"\n{rec_id}: {wav.shape[-1]/sr:.1f}s")

        for spk in speakers:
            if spk not in rttm:
                print(f"  {spk}: not in RTTM")
                continue

            spk_dir = os.path.join(OUTPUT_DIR, rec_id, spk)
            os.makedirs(spk_dir, exist_ok=True)

            clip_a, clip_b = extract_enrollment_clips(wav, sr, rttm[spk])

            torchaudio.save(os.path.join(spk_dir, "mixture.wav"), wav.unsqueeze(0), sr)
            torchaudio.save(os.path.join(spk_dir, "enrollment_a.wav"), clip_a.unsqueeze(0), sr)
            torchaudio.save(os.path.join(spk_dir, "enrollment_b.wav"), clip_b.unsqueeze(0), sr)

            print(f"  {spk}: extracting with enrollment A...")
            ext_a = run_full_extraction(model, t_predicter, wav, clip_a, config, device)

            print(f"  {spk}: extracting with enrollment B...")
            ext_b = run_full_extraction(model, t_predicter, wav, clip_b, config, device)

            ext_avg = (ext_a + ext_b) / 2.0
            max_val = ext_avg.abs().max()
            if max_val > 1.0:
                ext_avg = ext_avg / max_val

            torchaudio.save(os.path.join(spk_dir, "extracted_single_a.wav"), ext_a, sr)
            torchaudio.save(os.path.join(spk_dir, "extracted_single_b.wav"), ext_b, sr)
            torchaudio.save(os.path.join(spk_dir, "extracted_average.wav"), ext_avg, sr)

            sdr_a = si_sdr(ext_a.squeeze(), wav)
            sdr_b = si_sdr(ext_b.squeeze(), wav)
            sdr_avg = si_sdr(ext_avg.squeeze(), wav)

            result = {
                "recording": rec_id, "speaker": spk,
                "si_sdr_single_a": sdr_a, "si_sdr_single_b": sdr_b,
                "si_sdr_average": sdr_avg,
            }
            all_results.append(result)

            print(f"  {spk}: SI-SDR(vs mix) -- A: {sdr_a:.2f}, B: {sdr_b:.2f}, Avg: {sdr_avg:.2f}")
            print(f"  {spk}: saved all audio files")

    summary = ["Multi-Enrollment DIHARD Test", f"Recordings: {len(RECORDINGS)}, Speakers: {len(all_results)}", ""]
    summary.append("SI-SDR (vs mixture):")
    for r in all_results:
        summary.append(f"  {r['recording']}/{r['speaker']}: "
                       f"A={r['si_sdr_single_a']:.2f}, B={r['si_sdr_single_b']:.2f}, "
                       f"Avg={r['si_sdr_average']:.2f}")

    means_a = np.mean([r["si_sdr_single_a"] for r in all_results])
    means_b = np.mean([r["si_sdr_single_b"] for r in all_results])
    means_avg = np.mean([r["si_sdr_average"] for r in all_results])
    summary.append(f"\nMean SI-SDR: Single A={means_a:.2f}, Single B={means_b:.2f}, Average={means_avg:.2f}")
    summary.append(f"Avg improvement over single: {means_avg - (means_a + means_b)/2:.2f} dB")

    summary_text = "\n".join(summary)
    print(f"\n{summary_text}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
