import os
import random
from collections import defaultdict

import torch
import torchaudio
from tqdm import tqdm

from core.inference import (
    parse_config, load_flowtse_model, load_speaker_gate,
    pad_and_reshape, reshape_and_remove_padding,
)
from models.t_predicter import TPredicter
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver

DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
CONFIG_PATH = "config/config_FlowTSE_large_noisy.yaml"
GATE_CKPT = "best_unfrozen.ckpt"
GATE_THRESHOLD = 0.5
OUTPUT_DIR = "tse_dihard_results"
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


def run_gate_per_chunk(gate, mixture_wav, enrollment_wav, device, threshold):
    total = mixture_wav.shape[-1]
    presence = []
    probs = []
    with torch.no_grad():
        for start in range(0, total, CHUNK_SAMPLES):
            chunk = mixture_wav[start:start + CHUNK_SAMPLES]
            if chunk.shape[-1] < CHUNK_SAMPLES:
                chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[-1]))
            prob = gate.predict(
                chunk.unsqueeze(0).to(device),
                enrollment_wav[:CHUNK_SAMPLES].unsqueeze(0).to(device),
            ).item()
            probs.append(prob)
            presence.append(prob >= threshold)
    return presence, probs


def run_extraction(model, t_predicter, gate, mixture_wav, enrollment_wav, config, device, threshold):
    n_fft = config["dataset"]["n_fft"]
    hop_length = config["dataset"]["hop_length"]
    win_length = config["dataset"]["win_length"]

    chunk_alphas = estimate_alpha_per_chunk(t_predicter, mixture_wav, enrollment_wav, device)
    chunk_presence, chunk_probs = run_gate_per_chunk(gate, mixture_wav, enrollment_wav, device, threshold)

    n_present = sum(chunk_presence)
    n_absent = len(chunk_presence) - n_present
    print(f"    Gate: {n_present} present, {n_absent} absent (threshold={threshold})")

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
    ext_wav = istft_torch(source_hat_spec, n_fft, hop_length, win_length,
                          length=mixture_wav.shape[-1])
    max_val = ext_wav.abs().max()
    if max_val > 1.0:
        ext_wav = ext_wav / max_val

    gated_wav = ext_wav.clone()
    for ci, present in enumerate(chunk_presence):
        if not present:
            s = ci * CHUNK_SAMPLES
            e = min(s + CHUNK_SAMPLES, gated_wav.shape[-1])
            gated_wav[..., s:e] = 0.0

    return ext_wav, gated_wav, chunk_presence, chunk_probs


def extract_enrollment(wav, sr, segments):
    seg_list = [(int(s * sr), int(e * sr), e - s) for s, e in segments]
    seg_list.sort(key=lambda x: -x[2])
    start, end, dur = seg_list[0]
    clip = wav[start:start + sr * CHUNK_SEC]
    if clip.shape[-1] < CHUNK_SAMPLES:
        clip = torch.nn.functional.pad(clip, (0, CHUNK_SAMPLES - clip.shape[-1]))
    return clip


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

    print(f"Loading Speaker Gate: {GATE_CKPT}")
    gate = load_speaker_gate(GATE_CKPT, device)

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

            enrollment = extract_enrollment(wav, sr, rttm[spk])

            torchaudio.save(os.path.join(spk_dir, "mixture.wav"), wav.unsqueeze(0), sr)
            torchaudio.save(os.path.join(spk_dir, "enrollment.wav"), enrollment.unsqueeze(0), sr)

            print(f"  {spk}: running TSE with gate...")
            ext_ungated, ext_gated, presence, probs = run_extraction(
                model, t_predicter, gate, wav, enrollment, config, device, GATE_THRESHOLD
            )

            torchaudio.save(os.path.join(spk_dir, "extracted.wav"), ext_ungated, sr)
            torchaudio.save(os.path.join(spk_dir, "extracted_gated.wav"), ext_gated, sr)

            dur = wav.shape[-1] / sr
            print(f"  {spk}: saved extracted.wav + extracted_gated.wav ({dur:.1f}s)")

    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
