import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from core.inference import load_t_predicter


DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
RESULTS_ROOT = "dihard_results"
OUTPUT_DIR = "similarity_postprocess_results"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

T_PREDICTER_CKPT = "t_predictor_noisy.ckpt"
T_PREDICTER_CONFIG = {"C": 1024}

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


def compute_embedding(ecapa_model, waveform, device):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    with torch.no_grad():
        emb = ecapa_model(waveform.to(device), aug=False)
    return F.normalize(emb, dim=-1).cpu()


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ECAPA-TDNN from TPredicter checkpoint...")
    t_predicter = load_t_predicter(T_PREDICTER_CKPT, T_PREDICTER_CONFIG, device)
    ecapa = t_predicter.ecapa_tdnn
    ecapa.eval()

    all_data = []

    for rec in RECORDINGS:
        file_id = rec["file_id"]
        rttm_path = os.path.join(DIHARD_DIR, "rttm", f"{file_id}.rttm")
        flac_path = os.path.join(DIHARD_DIR, "flac", f"{file_id}.flac")

        print(f"\n=== {file_id} ===")
        speaker_segments = parse_rttm(rttm_path)
        audio, sr = torchaudio.load(flac_path)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.squeeze(0)
        total_duration = audio.shape[-1] / SAMPLE_RATE

        for speaker in rec["speakers"]:
            print(f"\n  Speaker: {speaker}")

            enroll_path = os.path.join(RESULTS_ROOT, file_id, speaker, "enrollment.wav")
            extracted_path = os.path.join(RESULTS_ROOT, file_id, speaker, "extracted_fixed.wav")

            if not os.path.exists(enroll_path) or not os.path.exists(extracted_path):
                print(f"    Missing files, skipping")
                continue

            enroll_wav, _ = torchaudio.load(enroll_path)
            enroll_wav = enroll_wav.squeeze(0)
            extracted_wav, _ = torchaudio.load(extracted_path)
            extracted_wav = extracted_wav.squeeze(0)

            enroll_emb = compute_embedding(ecapa, enroll_wav, device)

            num_chunks = (extracted_wav.shape[-1] + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
            target_segs = speaker_segments.get(speaker, [])

            for i in range(num_chunks):
                start_sample = i * CHUNK_SAMPLES
                end_sample = min(start_sample + CHUNK_SAMPLES, extracted_wav.shape[-1])
                chunk = extracted_wav[start_sample:end_sample]

                if chunk.shape[-1] < CHUNK_SAMPLES:
                    chunk = F.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[-1]))

                chunk_start_sec = start_sample / SAMPLE_RATE
                chunk_end_sec = end_sample / SAMPLE_RATE

                is_present = chunk_has_target(chunk_start_sec, chunk_end_sec, target_segs)

                chunk_emb = compute_embedding(ecapa, chunk, device)
                cos_sim = F.cosine_similarity(enroll_emb, chunk_emb, dim=-1).item()

                chunk_energy = chunk.pow(2).mean().sqrt().item()

                all_data.append({
                    "file_id": file_id,
                    "speaker": speaker,
                    "chunk_idx": i,
                    "chunk_start": chunk_start_sec,
                    "chunk_end": chunk_end_sec,
                    "target_present": is_present,
                    "cosine_similarity": cos_sim,
                    "chunk_energy_rms": chunk_energy,
                })

            present_count = sum(1 for d in all_data if d["speaker"] == speaker and d["file_id"] == file_id and d["target_present"])
            absent_count = sum(1 for d in all_data if d["speaker"] == speaker and d["file_id"] == file_id and not d["target_present"])
            print(f"    Chunks: {present_count} present, {absent_count} absent")

    import csv
    csv_path = os.path.join(OUTPUT_DIR, "chunk_similarities.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
        writer.writeheader()
        writer.writerows(all_data)
    print(f"\nSaved {len(all_data)} chunk records to {csv_path}")

    present_sims = [d["cosine_similarity"] for d in all_data if d["target_present"]]
    absent_sims = [d["cosine_similarity"] for d in all_data if not d["target_present"]]

    print(f"\n=== SIMILARITY STATISTICS ===")
    print(f"Target present:  n={len(present_sims)}, mean={np.mean(present_sims):.4f}, "
          f"median={np.median(present_sims):.4f}, std={np.std(present_sims):.4f}")
    print(f"Target absent:   n={len(absent_sims)}, mean={np.mean(absent_sims):.4f}, "
          f"median={np.median(absent_sims):.4f}, std={np.std(absent_sims):.4f}")

    all_sims = present_sims + absent_sims
    all_labels = [1] * len(present_sims) + [0] * len(absent_sims)

    thresholds = np.linspace(min(all_sims) - 0.01, max(all_sims) + 0.01, 200)
    precisions, recalls, f1s = [], [], []
    for thresh in thresholds:
        tp = sum(1 for s, l in zip(all_sims, all_labels) if s < thresh and l == 0)
        fp = sum(1 for s, l in zip(all_sims, all_labels) if s < thresh and l == 1)
        fn = sum(1 for s, l in zip(all_sims, all_labels) if s >= thresh and l == 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]

    print(f"\n=== GENERATING GATED OUTPUTS (threshold={best_thresh:.4f}) ===")
    for rec in RECORDINGS:
        file_id = rec["file_id"]
        for speaker in rec["speakers"]:
            extracted_path = os.path.join(RESULTS_ROOT, file_id, speaker, "extracted_fixed.wav")
            if not os.path.exists(extracted_path):
                continue

            extracted_wav, sr = torchaudio.load(extracted_path)
            extracted_wav = extracted_wav.squeeze(0)

            speaker_chunks = [d for d in all_data
                              if d["file_id"] == file_id and d["speaker"] == speaker]

            gated = extracted_wav.clone()
            silenced_count = 0
            for cd in speaker_chunks:
                if cd["cosine_similarity"] < best_thresh:
                    start = int(cd["chunk_start"] * SAMPLE_RATE)
                    end = min(int(cd["chunk_end"] * SAMPLE_RATE), gated.shape[-1])
                    gated[start:end] = 0.0
                    silenced_count += 1

            out_path = os.path.join(OUTPUT_DIR, file_id, speaker)
            os.makedirs(out_path, exist_ok=True)
            gated_path = os.path.join(out_path, "extracted_gated.wav")
            torchaudio.save(gated_path, gated.unsqueeze(0), SAMPLE_RATE)

            total_chunks = len(speaker_chunks)
            print(f"  {file_id}/{speaker}: silenced {silenced_count}/{total_chunks} chunks -> {gated_path}")

    print(f"\n{'='*65}")
    print(f"  PHASE 1A RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  Total chunks analyzed: {len(all_data)}")
    print(f"  Target present: {len(present_sims)}, Target absent: {len(absent_sims)}")
    print(f"  Best threshold: {best_thresh:.4f}")
    print(f"  At best threshold:")
    print(f"    Precision: {precisions[best_idx]:.3f}")
    print(f"    Recall:    {recalls[best_idx]:.3f}")
    print(f"    F1:        {f1s[best_idx]:.3f}")
    print(f"{'='*65}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Phase 1A: Speaker Similarity Post-Processing\n\n")
        f.write(f"Total chunks: {len(all_data)}\n")
        f.write(f"Target present: {len(present_sims)}, Target absent: {len(absent_sims)}\n\n")
        f.write(f"Present similarity: mean={np.mean(present_sims):.4f}, median={np.median(present_sims):.4f}, std={np.std(present_sims):.4f}\n")
        f.write(f"Absent similarity:  mean={np.mean(absent_sims):.4f}, median={np.median(absent_sims):.4f}, std={np.std(absent_sims):.4f}\n\n")
        f.write(f"Best threshold: {best_thresh:.4f}\n")
        f.write(f"Precision: {precisions[best_idx]:.3f}\n")
        f.write(f"Recall:    {recalls[best_idx]:.3f}\n")
        f.write(f"F1:        {f1s[best_idx]:.3f}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
