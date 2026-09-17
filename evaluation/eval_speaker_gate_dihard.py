import os
import re
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from core.speaker_gate import SpeakerVerificationGate

DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
GATE_CKPT = "best_unfrozen.ckpt"
GATE_CONFIG = {"C": 1024}
OUTPUT_DIR = "speaker_gate_dihard_eval"
SAMPLE_RATE = 16000
CHUNK_SEC = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SEC

RECORDINGS = {
    "DH_EVAL_0011": {
        "audio": os.path.join(DIHARD_DIR, "flac/DH_EVAL_0011.flac"),
        "rttm": os.path.join(DIHARD_DIR, "rttm/DH_EVAL_0011.rttm"),
        "speakers": ["speaker27", "speaker28"],
    },
    "DH_EVAL_0012": {
        "audio": os.path.join(DIHARD_DIR, "flac/DH_EVAL_0012.flac"),
        "rttm": os.path.join(DIHARD_DIR, "rttm/DH_EVAL_0012.rttm"),
        "speakers": ["speaker29", "speaker30"],
    },
}


def load_gate(ckpt_path, config, device):
    model = SpeakerVerificationGate(**config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def parse_rttm(rttm_path):
    segments = defaultdict(list)
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            speaker = parts[7]
            start = float(parts[3])
            dur = float(parts[4])
            segments[speaker].append((start, start + dur))
    return segments


def is_speaker_present(speaker_segments, chunk_start, chunk_end, min_overlap=0.5):
    overlap = 0.0
    for seg_start, seg_end in speaker_segments:
        o_start = max(chunk_start, seg_start)
        o_end = min(chunk_end, seg_end)
        if o_end > o_start:
            overlap += o_end - o_start
    return overlap >= min_overlap


def extract_enrollment(wav, sr, segments):
    seg_list = [(int(s * sr), int(e * sr), e - s) for s, e in segments]
    seg_list.sort(key=lambda x: -x[2])
    start, end, dur = seg_list[0]
    clip = wav[start:start + sr * CHUNK_SEC]
    if clip.shape[-1] < CHUNK_SAMPLES:
        clip = torch.nn.functional.pad(clip, (0, CHUNK_SAMPLES - clip.shape[-1]))
    return clip


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading gate: {GATE_CKPT}")
    gate = load_gate(GATE_CKPT, GATE_CONFIG, device)

    all_results = []

    for rec_id, rec_info in RECORDINGS.items():
        audio_path = rec_info["audio"]
        rttm_path = rec_info["rttm"]
        target_speakers = rec_info["speakers"]

        if not os.path.exists(audio_path):
            print(f"Skipping {rec_id}: audio not found")
            continue
        if not os.path.exists(rttm_path):
            print(f"Skipping {rec_id}: RTTM not found")
            continue

        print(f"\nProcessing {rec_id}")
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        total_samples = wav.shape[-1]
        duration = total_samples / SAMPLE_RATE
        print(f"  Duration: {duration:.1f}s, {total_samples} samples")

        rttm_segments = parse_rttm(rttm_path)
        print(f"  RTTM speakers: {sorted(rttm_segments.keys())}")
        print(f"  Target speakers: {target_speakers}")

        for spk in target_speakers:
            if spk not in rttm_segments:
                print(f"  Skipping {spk}: not in RTTM")
                continue

            spk_segments = rttm_segments[spk]
            enr_wav = extract_enrollment(wav, SAMPLE_RATE, spk_segments)
            print(f"  {spk}: extracted enrollment from longest segment")

            n_chunks = total_samples // CHUNK_SAMPLES

            for ci in range(n_chunks):
                chunk_start_sec = ci * CHUNK_SEC
                chunk_end_sec = (ci + 1) * CHUNK_SEC
                chunk_start_samp = ci * CHUNK_SAMPLES
                chunk_wav = wav[chunk_start_samp:chunk_start_samp + CHUNK_SAMPLES]

                if chunk_wav.shape[-1] < CHUNK_SAMPLES:
                    chunk_wav = torch.nn.functional.pad(
                        chunk_wav, (0, CHUNK_SAMPLES - chunk_wav.shape[-1])
                    )

                present = is_speaker_present(spk_segments, chunk_start_sec, chunk_end_sec)

                with torch.no_grad():
                    prob = gate.predict(
                        chunk_wav.unsqueeze(0), enr_wav.unsqueeze(0)
                    ).item()

                all_results.append({
                    "recording": rec_id,
                    "speaker": spk,
                    "chunk": ci,
                    "present": present,
                    "prob": prob,
                })

            n_present = sum(1 for r in all_results
                           if r["recording"] == rec_id and r["speaker"] == spk and r["present"])
            n_absent = sum(1 for r in all_results
                          if r["recording"] == rec_id and r["speaker"] == spk and not r["present"])
            print(f"  {spk}: {n_present} present, {n_absent} absent chunks")

    if not all_results:
        print("No results to analyze")
        return

    labels = np.array([1.0 if r["present"] else 0.0 for r in all_results])
    probs = np.array([r["prob"] for r in all_results])
    preds = (probs > 0.5).astype(float)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec_score = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    present_probs = probs[labels == 1]
    absent_probs = probs[labels == 0]

    summary = []
    summary.append("Speaker Verification Gate - DIHARD Evaluation")
    summary.append(f"Total chunks: {len(labels)} ({int(labels.sum())} present, {int((1-labels).sum())} absent)\n")
    summary.append(f"Accuracy:  {acc:.4f}")
    summary.append(f"Precision: {prec:.4f}")
    summary.append(f"Recall:    {rec_score:.4f}")
    summary.append(f"F1 Score:  {f1:.4f}")
    summary.append(f"AUC-ROC:   {auc:.4f}\n")
    summary.append(f"Present P(present) mean: {present_probs.mean():.4f} +/- {present_probs.std():.4f}")
    summary.append(f"Absent  P(present) mean: {absent_probs.mean():.4f} +/- {absent_probs.std():.4f}")
    summary.append(f"\nComparison with Phase 1A post-processing (F1 = 0.71)")
    summary.append(f"Gate F1: {f1:.4f}")

    summary.append("\nPer-speaker breakdown:")
    for rec_id in RECORDINGS:
        for spk in sorted(set(r["speaker"] for r in all_results if r["recording"] == rec_id)):
            spk_results = [r for r in all_results if r["recording"] == rec_id and r["speaker"] == spk]
            spk_labels = np.array([1.0 if r["present"] else 0.0 for r in spk_results])
            spk_probs = np.array([r["prob"] for r in spk_results])
            spk_preds = (spk_probs > 0.5).astype(float)
            spk_f1 = f1_score(spk_labels, spk_preds, zero_division=0)
            n_p = int(spk_labels.sum())
            n_a = len(spk_labels) - n_p
            summary.append(f"  {rec_id}/{spk}: F1={spk_f1:.3f} ({n_p} present, {n_a} absent)")

    summary_text = "\n".join(summary)
    print(f"\n{summary_text}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)

    with open(os.path.join(OUTPUT_DIR, "predictions.csv"), "w") as f:
        f.write("recording,speaker,chunk,present,prob\n")
        for r in all_results:
            f.write(f"{r['recording']},{r['speaker']},{r['chunk']},"
                    f"{1 if r['present'] else 0},{r['prob']:.6f}\n")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
