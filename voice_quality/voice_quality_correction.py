import os
import sys
import argparse
import json

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SR = 16000
CHUNK_SEC = 15

VOICE_QUALITY_LABELS = [
    'shrill', 'nasal', 'deep',
    'silky', 'husky', 'raspy', 'guttural', 'vocal-fry',
    'booming', 'authoritative', 'loud', 'hushed', 'soft',
    'crisp', 'slurred', 'lisp', 'stammering',
    'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated', 'hesitant',
]


def load_voice_quality_model():
    vox_path = os.path.join(BASE_DIR, "vox-profile-release")
    sys.path.insert(0, vox_path)

    from src.model.voice_quality import whisper_voice_quality as vq_mod
    orig_fwd = vq_mod.WhisperEncoderLayer.forward
    def patched_fwd(self, hidden_states, attention_mask=None, layer_head_mask=None, output_attentions=False, **kwargs):
        return orig_fwd(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    vq_mod.WhisperEncoderLayer.forward = patched_fwd

    from src.model.voice_quality.whisper_voice_quality import WhisperWrapper

    device = torch.device("cpu")
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-voice-quality").float().to(device)
    model.eval()
    return model, device


def get_probs(model, device, audio):
    if audio.shape[0] < 3 * SR:
        return None
    audio = audio[:CHUNK_SEC * SR]
    data = audio.unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(data, return_feature=False)
    return torch.sigmoid(logits).cpu().numpy()[0]


def get_labels(probs, threshold=0.5):
    return [VOICE_QUALITY_LABELS[i] for i, p in enumerate(probs) if p > threshold]


def voice_quality_distance(probs_a, probs_b):
    return float(np.sum(np.abs(probs_a - probs_b)))


def label_set_distance(probs_a, probs_b, threshold=0.5):
    labels_a = set(get_labels(probs_a, threshold))
    labels_b = set(get_labels(probs_b, threshold))
    return len(labels_a.symmetric_difference(labels_b))


def optimize_mixing_weight(model, device, noisy_chunk, enhanced_chunk, ref_probs, n_steps=11):
    best_alpha = 1.0
    best_dist = float('inf')
    best_probs = None

    alphas = np.linspace(0, 1, n_steps)

    for alpha in alphas:
        corrected = alpha * enhanced_chunk + (1 - alpha) * noisy_chunk
        probs = get_probs(model, device, corrected)
        if probs is None:
            continue
        dist = voice_quality_distance(probs, ref_probs)
        if dist < best_dist:
            best_dist = dist
            best_alpha = alpha
            best_probs = probs

    return best_alpha, best_dist, best_probs


def correct_audio(model, device, noisy_wav, enhanced_wav, ref_probs, chunk_samples):
    min_len = min(len(noisy_wav), len(enhanced_wav))
    noisy_wav = noisy_wav[:min_len]
    enhanced_wav = enhanced_wav[:min_len]

    corrected = torch.zeros_like(noisy_wav)
    alphas = []
    n_chunks = (min_len + chunk_samples - 1) // chunk_samples

    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, min_len)

        noisy_chunk = noisy_wav[start:end]
        enhanced_chunk = enhanced_wav[start:end]

        if end - start < 3 * SR:
            corrected[start:end] = enhanced_chunk
            alphas.append(1.0)
            continue

        alpha, dist, _ = optimize_mixing_weight(
            model, device, noisy_chunk, enhanced_chunk, ref_probs
        )
        corrected[start:end] = alpha * enhanced_chunk + (1 - alpha) * noisy_chunk
        alphas.append(alpha)

    return corrected, alphas


def main():
    parser = argparse.ArgumentParser(description="Voice-quality-guided correction for SE output")
    parser.add_argument("--enrollment", required=True, help="Clean enrollment audio (reference voice quality)")
    parser.add_argument("--noisy", required=True, help="Noisy input audio")
    parser.add_argument("--enhanced", required=True, help="RE-USE enhanced audio")
    parser.add_argument("--output", required=True, help="Output corrected audio path")
    parser.add_argument("--n_steps", type=int, default=11, help="Number of alpha values to search")
    args = parser.parse_args()

    print("Loading voice quality model...")
    model, device = load_voice_quality_model()

    print("Loading audio...")
    enrollment_wav = torch.from_numpy(sf.read(args.enrollment, dtype='float32')[0])
    noisy_wav = torch.from_numpy(sf.read(args.noisy, dtype='float32')[0])
    enhanced_wav = torch.from_numpy(sf.read(args.enhanced, dtype='float32')[0])

    if enrollment_wav.dim() > 1:
        enrollment_wav = enrollment_wav.mean(dim=-1)
    if noisy_wav.dim() > 1:
        noisy_wav = noisy_wav.mean(dim=-1)
    if enhanced_wav.dim() > 1:
        enhanced_wav = enhanced_wav.mean(dim=-1)

    print("Getting reference voice quality from enrollment...")
    ref_probs = get_probs(model, device, enrollment_wav)
    ref_labels = get_labels(ref_probs)
    print(f"  Reference labels: {ref_labels}")

    chunk_samples = CHUNK_SEC * SR
    print(f"Optimizing mixing weights ({args.n_steps} alpha values per chunk)...")
    corrected_wav, alphas = correct_audio(
        model, device, noisy_wav, enhanced_wav, ref_probs, chunk_samples
    )

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    sf.write(args.output, corrected_wav.numpy(), SR)
    print(f"\nCorrected audio saved to: {args.output}")

    print(f"\nPer-chunk mixing weights (alpha: 1=fully enhanced, 0=fully noisy):")
    for i, alpha in enumerate(alphas):
        print(f"  Chunk {i}: alpha={alpha:.2f}")
    print(f"\n  Mean alpha: {np.mean(alphas):.2f}")
    print(f"  Chunks favoring noisy (alpha<0.5): {sum(1 for a in alphas if a < 0.5)}/{len(alphas)}")
    print(f"  Chunks favoring enhanced (alpha>0.5): {sum(1 for a in alphas if a > 0.5)}/{len(alphas)}")

    print(f"\nVoice quality comparison:")
    for name, wav in [("Noisy", noisy_wav), ("Enhanced", enhanced_wav), ("Corrected", corrected_wav)]:
        probs = get_probs(model, device, wav[:CHUNK_SEC * SR])
        if probs is not None:
            labels = get_labels(probs)
            dist = voice_quality_distance(probs, ref_probs)
            ldist = label_set_distance(probs, ref_probs)
            print(f"  {name:10s}: labels={labels}")
            print(f"             continuous_dist={dist:.2f}, label_dist={ldist}")


if __name__ == "__main__":
    main()
