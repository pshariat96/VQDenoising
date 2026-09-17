import os
import sys
import json
import time

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RECORDINGS = [
    {
        "name": "DH_EVAL_0011",
        "before": os.path.join(BASE_DIR, "reuse_experiment/output/dihard/DH_EVAL_0011/before.wav"),
        "after": os.path.join(BASE_DIR, "reuse_experiment/output/dihard/DH_EVAL_0011/after_reuse.wav"),
    },
    {
        "name": "DH_EVAL_0012",
        "before": os.path.join(BASE_DIR, "reuse_experiment/output/dihard/DH_EVAL_0012/before.wav"),
        "after": os.path.join(BASE_DIR, "reuse_experiment/output/dihard/DH_EVAL_0012/after_reuse.wav"),
    },
    {
        "name": "YouTube",
        "before": os.path.join(BASE_DIR, "reuse_experiment/output/youtube/before.wav"),
        "after": os.path.join(BASE_DIR, "reuse_experiment/output/youtube/after_reuse.wav"),
    },
]

OUTPUT_DIR = os.path.join(BASE_DIR, "reuse_experiment/output/dihard_youtube_voice_quality")
TOP_N = 15
CHUNK_SEC = 15
MIN_CHUNK_SEC = 3
SR = 16000

VOICE_QUALITY_LABELS = [
    'shrill', 'nasal', 'deep',
    'silky', 'husky', 'raspy', 'guttural', 'vocal-fry',
    'booming', 'authoritative', 'loud', 'hushed', 'soft',
    'crisp', 'slurred', 'lisp', 'stammering',
    'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated', 'hesitant',
]


def load_model():
    vox_path = os.path.join(BASE_DIR, "vox-profile-release")
    if not os.path.exists(os.path.join(vox_path, "src")):
        print(f"ERROR: {vox_path}/src not found. Clone it first:")
        print(f"  git clone https://github.com/tiantiaf0627/vox-profile-release.git {vox_path}")
        sys.exit(1)

    sys.path.insert(0, vox_path)

    from src.model.voice_quality import whisper_voice_quality as vq_mod
    orig_fwd = vq_mod.WhisperEncoderLayer.forward

    def patched_fwd(self, hidden_states, attention_mask=None, layer_head_mask=None, output_attentions=False, **kwargs):
        return orig_fwd(self, hidden_states, attention_mask, layer_head_mask, output_attentions)

    vq_mod.WhisperEncoderLayer.forward = patched_fwd

    from src.model.voice_quality.whisper_voice_quality import WhisperWrapper

    device = torch.device("cpu")
    print(f"Loading model on {device}...")
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-voice-quality").float().to(device)
    model.eval()
    print("Model loaded.")
    return model, device


def load_audio(path):
    data, sr = sf.read(path, dtype='float32')
    wav = torch.from_numpy(data)
    if wav.dim() > 1:
        wav = wav.mean(dim=-1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def chunk_audio(wav, chunk_samples, min_samples):
    chunks = []
    for start in range(0, len(wav), chunk_samples):
        end = min(start + chunk_samples, len(wav))
        if end - start >= min_samples:
            chunks.append((start, end, wav[start:end]))
    return chunks


def predict(model, device, audio, threshold=0.5):
    data = audio.unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(data, return_feature=False)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    return [VOICE_QUALITY_LABELS[i] for i, p in enumerate(probs) if p > threshold]


def format_time(samples):
    sec = samples / SR
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, device = load_model()

    chunk_samples = CHUNK_SEC * SR
    min_samples = MIN_CHUNK_SEC * SR

    all_results = []

    for rec in RECORDINGS:
        name = rec["name"]
        if not os.path.exists(rec["before"]):
            print(f"\n  {name}: before.wav not found, skipping")
            continue
        if not os.path.exists(rec["after"]):
            print(f"\n  {name}: after_reuse.wav not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        before_wav = load_audio(rec["before"])
        after_wav = load_audio(rec["after"])

        dur_before = len(before_wav) / SR
        dur_after = len(after_wav) / SR
        print(f"  Before: {dur_before:.1f}s | After: {dur_after:.1f}s")

        min_len = min(len(before_wav), len(after_wav))
        before_wav = before_wav[:min_len]
        after_wav = after_wav[:min_len]

        before_chunks = chunk_audio(before_wav, chunk_samples, min_samples)
        after_chunks = chunk_audio(after_wav, chunk_samples, min_samples)

        n_chunks = min(len(before_chunks), len(after_chunks))
        print(f"  Chunks: {n_chunks}")

        start_time = time.time()
        for i in range(n_chunks):
            b_start, b_end, b_audio = before_chunks[i]
            a_start, a_end, a_audio = after_chunks[i]

            before_labels = predict(model, device, b_audio)
            after_labels = predict(model, device, a_audio)

            added = sorted(set(after_labels) - set(before_labels))
            removed = sorted(set(before_labels) - set(after_labels))
            distance = len(added) + len(removed)

            result = {
                "recording": name,
                "chunk_idx": i,
                "start_sample": b_start,
                "end_sample": b_end,
                "time_range": f"{format_time(b_start)}–{format_time(b_end)}",
                "before_labels": before_labels,
                "after_labels": after_labels,
                "added": added,
                "removed": removed,
                "label_distance": distance,
            }
            all_results.append(result)

            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (n_chunks - i - 1) / rate
                print(f"  [{i+1}/{n_chunks}] {rate:.1f} chunks/sec, ETA: {eta:.0f}s")

        elapsed = time.time() - start_time
        print(f"  Done: {n_chunks} chunks in {elapsed:.0f}s")

    all_results.sort(key=lambda x: x["label_distance"], reverse=True)

    results_path = os.path.join(OUTPUT_DIR, "dihard_youtube_voice_quality.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {results_path}")

    top_dir = os.path.join(OUTPUT_DIR, "top_15_worst")
    os.makedirs(top_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TOP {TOP_N} CHUNKS WITH BIGGEST VOICE QUALITY CHANGE")
    print(f"{'='*60}")

    for rank, result in enumerate(all_results[:TOP_N]):
        rec_name = result["recording"]
        time_range = result["time_range"]
        before_labels = result["before_labels"]
        after_labels = result["after_labels"]
        added = result["added"]
        removed = result["removed"]
        distance = result["label_distance"]

        print(f"\n  #{rank+1}: {rec_name} @ {time_range} (distance={distance})")
        print(f"    Before: {', '.join(before_labels) if before_labels else '(none)'}")
        print(f"    After:  {', '.join(after_labels) if after_labels else '(none)'}")
        if added:
            print(f"    + Added:   {', '.join(added)}")
        if removed:
            print(f"    - Removed: {', '.join(removed)}")

        chunk_name = f"{rank+1:02d}_{rec_name}_{time_range.replace('–', '_to_').replace(':', '')}"
        chunk_dir = os.path.join(top_dir, chunk_name)
        os.makedirs(chunk_dir, exist_ok=True)

        rec_info = next(r for r in RECORDINGS if r["name"] == rec_name)
        before_wav = load_audio(rec_info["before"])
        after_wav = load_audio(rec_info["after"])

        s = result["start_sample"]
        e = result["end_sample"]
        sf.write(os.path.join(chunk_dir, "before.wav"), before_wav[s:e].numpy(), SR)
        sf.write(os.path.join(chunk_dir, "after_reuse.wav"), after_wav[s:e].numpy(), SR)

    print(f"\n{'='*60}")
    print(f"  Audio chunks saved to: {top_dir}")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY PER RECORDING")
    print(f"{'='*60}")
    from collections import Counter
    for rec in RECORDINGS:
        name = rec["name"]
        rec_results = [r for r in all_results if r["recording"] == name]
        if not rec_results:
            continue
        n = len(rec_results)
        avg_dist = sum(r["label_distance"] for r in rec_results) / n
        zero_change = sum(1 for r in rec_results if r["label_distance"] == 0)

        added_counter = Counter()
        removed_counter = Counter()
        for r in rec_results:
            added_counter.update(r["added"])
            removed_counter.update(r["removed"])

        print(f"\n  {name} ({n} chunks):")
        print(f"    Avg label distance: {avg_dist:.1f}")
        print(f"    Unchanged chunks:   {zero_change}/{n} ({zero_change/n*100:.0f}%)")
        print(f"    Top added:")
        for lbl, cnt in added_counter.most_common(5):
            print(f"      + {lbl:<15} {cnt}/{n} ({cnt/n*100:.0f}%)")
        print(f"    Top removed:")
        for lbl, cnt in removed_counter.most_common(5):
            print(f"      - {lbl:<15} {cnt}/{n} ({cnt/n*100:.0f}%)")


if __name__ == "__main__":
    main()
