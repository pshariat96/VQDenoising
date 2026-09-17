import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from core.inference import load_t_predicter

LIBRIMIX_DIR = os.environ.get("LIBRIMIX_DIR", "data/Libri2Mix") + "/sep_clean"
ORIGINAL_CKPT = "t_predictor_noisy.ckpt"
FINETUNED_CKPT = "t_predictor_target_absent_best.ckpt"
T_PRED_CONFIG = {"C": 1024}
OUTPUT_DIR = "phase1b_librimix_eval"
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = SAMPLE_RATE * 3
N_SAMPLES = 500
SEED = 42


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


def predict_alpha(t_predicter, mixture_wav, enrollment_wav, device):
    mix_in = mixture_wav.unsqueeze(0).to(device)
    enr_in = enrollment_wav.unsqueeze(0).to(device)
    with torch.no_grad():
        alpha = t_predicter(mix_in, enr_in, aug=False)
    return alpha.item()


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

    print(f"Eligible mixtures (both speakers have >=2 appearances): {len(eligible)}")
    selected = random.sample(eligible, min(N_SAMPLES, len(eligible)))
    print(f"Selected: {len(selected)}")

    print(f"\nLoading ORIGINAL T-Predictor: {ORIGINAL_CKPT}")
    t_pred_orig = load_t_predicter(ORIGINAL_CKPT, T_PRED_CONFIG, device)

    print(f"Loading FINE-TUNED T-Predictor: {FINETUNED_CKPT}")
    t_pred_ft = load_t_predicter(FINETUNED_CKPT, T_PRED_CONFIG, device)

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
        wrong_spk = None
        for ws in wrong_speakers:
            for wmid, wpos in spk_to_mixtures[ws]:
                wd = s1_dir if wpos == "s1" else s2_dir
                wp = os.path.join(wd, f"{wmid}.wav")
                if os.path.exists(wp):
                    try:
                        enr_wrong = load_audio_segment(wp, SEGMENT_SAMPLES)
                        wrong_spk = ws
                    except Exception:
                        continue
                    break
            if enr_wrong is not None:
                break
        if enr_wrong is None:
            continue

        orig_present = predict_alpha(t_pred_orig, mix_wav, enr_correct, device)
        orig_absent = predict_alpha(t_pred_orig, mix_wav, enr_wrong, device)
        ft_present = predict_alpha(t_pred_ft, mix_wav, enr_correct, device)
        ft_absent = predict_alpha(t_pred_ft, mix_wav, enr_wrong, device)

        results.append({
            "mix_id": mix_id,
            "spk_present": spk1,
            "spk_absent": wrong_spk,
            "orig_present": orig_present,
            "orig_absent": orig_absent,
            "ft_present": ft_present,
            "ft_absent": ft_absent,
        })

    print(f"\nCompleted {len(results)} evaluations")

    csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    with open(csv_path, "w") as f:
        f.write("mix_id,spk_present,spk_absent,orig_present,orig_absent,ft_present,ft_absent\n")
        for r in results:
            f.write(f"{r['mix_id']},{r['spk_present']},{r['spk_absent']},"
                    f"{r['orig_present']:.6f},{r['orig_absent']:.6f},"
                    f"{r['ft_present']:.6f},{r['ft_absent']:.6f}\n")

    orig_present = np.array([r["orig_present"] for r in results])
    orig_absent = np.array([r["orig_absent"] for r in results])
    ft_present = np.array([r["ft_present"] for r in results])
    ft_absent = np.array([r["ft_absent"] for r in results])

    summary = []
    summary.append("Phase 1B LibriMix Target-Absent Evaluation")
    summary.append(f"Samples: {len(results)}\n")
    summary.append("Alpha predictions (mean +/- std):")
    summary.append(f"  Original  -- present: {orig_present.mean():.4f} +/- {orig_present.std():.4f}")
    summary.append(f"  Original  -- absent:  {orig_absent.mean():.4f} +/- {orig_absent.std():.4f}")
    summary.append(f"  Finetuned -- present: {ft_present.mean():.4f} +/- {ft_present.std():.4f}")
    summary.append(f"  Finetuned -- absent:  {ft_absent.mean():.4f} +/- {ft_absent.std():.4f}")
    summary.append("")
    summary.append("Separation (present - absent mean):")
    summary.append(f"  Original:  {orig_present.mean() - orig_absent.mean():.4f}")
    summary.append(f"  Finetuned: {ft_present.mean() - ft_absent.mean():.4f}")
    summary.append("")

    for thresh in [0.1, 0.2, 0.3]:
        orig_pct = (orig_absent < thresh).mean() * 100
        ft_pct = (ft_absent < thresh).mean() * 100
        summary.append(f"  Absent alpha < {thresh}: Original {orig_pct:.1f}%, Finetuned {ft_pct:.1f}%")

    summary.append("")
    summary.append("DIHARD comparison (from Phase 1B eval):")
    summary.append("  Original absent alpha on DIHARD:  0.637")
    summary.append("  Finetuned absent alpha on DIHARD: 0.582 (delta = -0.055)")

    summary_text = "\n".join(summary)
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n{summary_text}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
