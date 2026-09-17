import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from collections import Counter

VOICE_QUALITY_LABELS = [
    'shrill', 'nasal', 'deep',
    'silky', 'husky', 'raspy', 'guttural', 'vocal-fry',
    'booming', 'authoritative', 'loud', 'hushed', 'soft',
    'crisp', 'slurred', 'lisp', 'stammering',
    'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated', 'hesitant',
]

SR = 16000
MAX_AUDIO_LENGTH = 15 * SR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def load_model():
    try:
        from src.model.voice_quality.whisper_voice_quality import WhisperWrapper
    except ImportError:
        vox_profile_path = Path(__file__).parent / "vox-profile-release"
        if vox_profile_path.exists():
            sys.path.insert(0, str(vox_profile_path))
            from src.model.voice_quality.whisper_voice_quality import WhisperWrapper
        else:
            print("ERROR: vox-profile-release not found.")
            print("  git clone https://github.com/tiantiaf0627/vox-profile-release.git")
            print("  cd vox-profile-release && pip install -e .")
            sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"  Loading model on: {device}")
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-voice-quality").to(device)
    model.eval()
    return model, device


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.squeeze(0)
    if wav.shape[0] < 3 * SR:
        return None
    return wav[:MAX_AUDIO_LENGTH]


def predict(model, device, audio, threshold=0.5):
    data = audio.unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(data, return_feature=False)
    probs = nn.Sigmoid()(torch.tensor(logits)).cpu().numpy()[0]
    labels = [VOICE_QUALITY_LABELS[i] for i, p in enumerate(probs) if p > threshold]
    return labels


def main():
    args = parse_args()
    base_dir = Path(__file__).parent
    output_dir = base_dir / "reuse_experiment" / "output" / "librimix"

    scenarios = {
        "Clean GT": output_dir / "clean_gt",
        "Enrollment": output_dir / "enrollment",
        "Mixture": output_dir / "before",
        "Mixture+Reuse": output_dir / "after_reuse",
    }

    available = {k: v for k, v in scenarios.items() if v.exists()}
    file_sets = [set(f.name for f in path.glob("*.wav")) for path in available.values()]
    common_files = sorted(set.intersection(*file_sets))[:args.n_samples]

    print("=" * 75)
    print("  Voice Quality: Before vs After RE-USE")
    print("=" * 75)
    print(f"  Model: whisper-large-v3-voice-quality")
    print(f"  Samples: {len(common_files)}, Threshold: {args.threshold}")
    print(f"  Conditions: {list(available.keys())}")
    print()

    model, device = load_model()
    print()

    all_results = {sn: {} for sn in available}

    for i, fname in enumerate(common_files):
        print(f"  ┌─ Sample {i+1}: {fname}")

        for scenario_name, scenario_dir in available.items():
            fpath = scenario_dir / fname
            audio = load_audio(str(fpath))
            if audio is None:
                print(f"  │  {scenario_name:<13}: (too short)")
                all_results[scenario_name][fname] = []
                continue

            labels = predict(model, device, audio, args.threshold)
            all_results[scenario_name][fname] = labels
            labels_str = ", ".join(labels) if labels else "(none)"
            print(f"  │  {scenario_name:<13}: {labels_str}")

        print(f"  └{'─'*70}")
        print()

    print("=" * 75)
    print("  SUMMARY: Label counts across all samples")
    print("=" * 75)

    label_counts = {sn: Counter() for sn in available}
    for sn in available:
        for fname in common_files:
            for lbl in all_results[sn].get(fname, []):
                label_counts[sn][lbl] += 1

    print(f"\n  {'Label':<15}", end="")
    for sn in available:
        print(f" {sn:>13}", end="")
    print()
    print(f"  {'─'*15}", end="")
    for _ in available:
        print(f" {'─'*13}", end="")
    print()

    all_seen = set()
    for counts in label_counts.values():
        all_seen.update(counts.keys())

    for lbl in VOICE_QUALITY_LABELS:
        if lbl not in all_seen:
            continue
        print(f"  {lbl:<15}", end="")
        for sn in available:
            print(f" {label_counts[sn].get(lbl, 0):>13}", end="")
        print()

    print(f"\n\n{'='*75}")
    print("  MIXTURE vs MIXTURE+REUSE: What changed?")
    print("=" * 75)

    changed = 0
    for fname in common_files:
        before = set(all_results.get("Mixture", {}).get(fname, []))
        after = set(all_results.get("Mixture+Reuse", {}).get(fname, []))
        if before != after:
            changed += 1
            added = after - before
            removed = before - after
            print(f"\n  {fname}:")
            if removed:
                print(f"    - removed: {', '.join(removed)}")
            if added:
                print(f"    + added:   {', '.join(added)}")

    if changed == 0:
        print(f"\n  No voice quality changes detected across {len(common_files)} samples!")
    else:
        print(f"\n  Changed: {changed}/{len(common_files)} samples")

    print(f"\n\n{'='*75}")
    print("  CLEAN GT vs MIXTURE+REUSE: Does RE-USE recover original voice quality?")
    print("=" * 75)

    matched = 0
    for fname in common_files:
        clean = set(all_results.get("Clean GT", {}).get(fname, []))
        after = set(all_results.get("Mixture+Reuse", {}).get(fname, []))
        if clean == after:
            matched += 1
        else:
            diff_added = after - clean
            diff_removed = clean - after
            print(f"\n  {fname}:")
            print(f"    Clean GT:      {', '.join(sorted(clean)) or '(none)'}")
            print(f"    Mixture+Reuse: {', '.join(sorted(after)) or '(none)'}")

    print(f"\n  Matching: {matched}/{len(common_files)} samples have identical voice quality labels")

    save_path = output_dir / "voice_quality_results.json"
    with open(save_path, 'w') as f:
        json.dump({
            "config": {"threshold": args.threshold, "n_samples": len(common_files)},
            "results": {sn: {fname: lbls for fname, lbls in res.items()} for sn, res in all_results.items()},
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
