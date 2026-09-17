import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from tqdm import tqdm

from core.speaker_gate import SpeakerVerificationGate

LIBRIMIX_DIR = os.environ.get("LIBRIMIX_DIR", "data/Libri2Mix") + "/sep_clean"
GATE_CKPT = "best_unfrozen.ckpt"
GATE_CONFIG = {"C": 1024}
OUTPUT_DIR = "speaker_gate_librimix_eval"
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = SAMPLE_RATE * 3
N_SAMPLES = 500
SEED = 42


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


def load_audio_segment(path, max_samples=None):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if max_samples is not None:
        if wav.shape[-1] > max_samples:
            wav = wav[:max_samples]
        elif wav.shape[-1] < max_samples:
            wav = torch.nn.functional.pad(wav, (0, max_samples - wav.shape[-1]))
    return wav


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mix_dir = os.path.join(LIBRIMIX_DIR, "mix_both")
    s1_dir = os.path.join(LIBRIMIX_DIR, "s1")
    s2_dir = os.path.join(LIBRIMIX_DIR, "s2")

    mix_files = sorted([f for f in os.listdir(mix_dir) if f.endswith(".wav")])
    print(f"Total mix_both files: {len(mix_files)}")

    spk_to_mixtures = defaultdict(list)
    all_speakers = set()
    for f in mix_files:
        mix_id = f.replace(".wav", "")
        utt1, utt2 = mix_id.split("_")
        spk1 = utt1.split("-")[0]
        spk2 = utt2.split("-")[0]
        spk_to_mixtures[spk1].append((mix_id, "s1"))
        spk_to_mixtures[spk2].append((mix_id, "s2"))
        all_speakers.add(spk1)
        all_speakers.add(spk2)

    all_speakers = sorted(all_speakers)
    print(f"Unique speakers: {len(all_speakers)}")

    eligible = []
    for f in mix_files:
        mix_id = f.replace(".wav", "")
        utt1, utt2 = mix_id.split("_")
        spk1 = utt1.split("-")[0]
        spk2 = utt2.split("-")[0]
        if len(spk_to_mixtures[spk1]) >= 2 and len(spk_to_mixtures[spk2]) >= 2:
            eligible.append(mix_id)

    selected = random.sample(eligible, min(N_SAMPLES, len(eligible)))
    print(f"Selected: {len(selected)}")

    print(f"\nLoading gate: {GATE_CKPT}")
    gate = load_gate(GATE_CKPT, GATE_CONFIG, device)

    all_labels = []
    all_probs = []
    results = []

    for mix_id in tqdm(selected, desc="Evaluating"):
        utt1, utt2 = mix_id.split("_")
        spk1 = utt1.split("-")[0]
        spk2 = utt2.split("-")[0]

        mix_path = os.path.join(mix_dir, f"{mix_id}.wav")
        if not os.path.exists(mix_path):
            continue
        try:
            mix_wav = load_audio_segment(mix_path, SEGMENT_SAMPLES)
        except Exception:
            continue

        other_mixes_spk1 = [
            (mid, pos) for mid, pos in spk_to_mixtures[spk1] if mid != mix_id
        ]
        if not other_mixes_spk1:
            continue
        enr_mid, enr_pos = random.choice(other_mixes_spk1)
        enr_dir = s1_dir if enr_pos == "s1" else s2_dir
        enr_path = os.path.join(enr_dir, f"{enr_mid}.wav")
        if not os.path.exists(enr_path):
            continue
        try:
            enr_correct = load_audio_segment(enr_path, SEGMENT_SAMPLES)
        except Exception:
            continue

        wrong_speakers = [s for s in all_speakers if s != spk1 and s != spk2]
        random.shuffle(wrong_speakers)
        enr_wrong = None
        for ws in wrong_speakers:
            for wmid, wpos in spk_to_mixtures[ws]:
                wd = s1_dir if wpos == "s1" else s2_dir
                wp = os.path.join(wd, f"{wmid}.wav")
                if os.path.exists(wp):
                    try:
                        enr_wrong = load_audio_segment(wp, SEGMENT_SAMPLES)
                    except Exception:
                        continue
                    break
            if enr_wrong is not None:
                break
        if enr_wrong is None:
            continue

        with torch.no_grad():
            prob_present = gate.predict(
                mix_wav.unsqueeze(0), enr_correct.unsqueeze(0)
            ).item()
            prob_absent = gate.predict(
                mix_wav.unsqueeze(0), enr_wrong.unsqueeze(0)
            ).item()

        all_labels.extend([1.0, 0.0])
        all_probs.extend([prob_present, prob_absent])
        results.append({
            "mix_id": mix_id,
            "prob_present": prob_present,
            "prob_absent": prob_absent,
        })

    print(f"\nCompleted {len(results)} evaluations")

    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds_05 = (probs > 0.5).astype(float)

    acc = accuracy_score(labels, preds_05)
    prec = precision_score(labels, preds_05, zero_division=0)
    rec = recall_score(labels, preds_05, zero_division=0)
    f1 = f1_score(labels, preds_05, zero_division=0)
    auc = roc_auc_score(labels, probs)

    present_probs = probs[labels == 1]
    absent_probs = probs[labels == 0]

    summary = []
    summary.append("Speaker Verification Gate - LibriMix Evaluation")
    summary.append(f"Samples: {len(results)} mixtures ({len(labels)} total pairs)\n")
    summary.append(f"Accuracy:  {acc:.4f}")
    summary.append(f"Precision: {prec:.4f}")
    summary.append(f"Recall:    {rec:.4f}")
    summary.append(f"F1 Score:  {f1:.4f}")
    summary.append(f"AUC-ROC:   {auc:.4f}\n")
    summary.append(f"Present P(present) mean: {present_probs.mean():.4f} +/- {present_probs.std():.4f}")
    summary.append(f"Absent  P(present) mean: {absent_probs.mean():.4f} +/- {absent_probs.std():.4f}")
    summary.append(f"Separation: {present_probs.mean() - absent_probs.mean():.4f}")

    summary_text = "\n".join(summary)
    print(f"\n{summary_text}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)

    with open(os.path.join(OUTPUT_DIR, "predictions.csv"), "w") as f:
        f.write("mix_id,prob_present,prob_absent\n")
        for r in results:
            f.write(f"{r['mix_id']},{r['prob_present']:.6f},{r['prob_absent']:.6f}\n")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
