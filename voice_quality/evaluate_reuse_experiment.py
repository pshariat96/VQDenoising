import os
import csv
import torch
import torchaudio
import numpy as np
from collections import defaultdict

OUTPUT_DIR = "reuse_experiment/output"
SR = 16000


def si_sdr(estimate, reference):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target
    return (10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )).item()


def rms(wav):
    return torch.sqrt(torch.mean(wav ** 2)).item()


def correlation(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def load_mono(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0), target_sr


def evaluate_librimix():
    lm_dir = os.path.join(OUTPUT_DIR, "librimix")
    if not os.path.exists(lm_dir):
        print("  LibriMix results not found")
        return None

    csv_path = os.path.join(OUTPUT_DIR, "librimix_reuse_metrics.csv")
    if os.path.exists(csv_path):
        print(f"  Found pre-computed metrics: {csv_path}")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            results = list(reader)
        for r in results:
            for k in r:
                if k != "mixture_id" and r[k] is not None and r[k] != "":
                    try:
                        r[k] = float(r[k])
                    except (ValueError, TypeError):
                        pass
        return results

    before_dir = os.path.join(lm_dir, "before")
    after_dir = os.path.join(lm_dir, "after_reuse")
    gt_dir = os.path.join(lm_dir, "clean_gt")

    if not all(os.path.exists(d) for d in [before_dir, after_dir, gt_dir]):
        print("  LibriMix output directories incomplete")
        return None

    try:
        from pesq import pesq as pesq_fn
        has_pesq = True
    except ImportError:
        has_pesq = False

    try:
        from pystoi import stoi as stoi_fn
        has_stoi = True
    except ImportError:
        has_stoi = False

    results = []
    for fname in sorted(os.listdir(before_dir)):
        if not fname.endswith(".wav"):
            continue
        mix_id = fname.replace(".wav", "")

        before_wav, _ = load_mono(os.path.join(before_dir, fname))
        after_wav, _ = load_mono(os.path.join(after_dir, fname))
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue
        gt_wav, _ = load_mono(gt_path)

        min_len = min(before_wav.shape[-1], after_wav.shape[-1], gt_wav.shape[-1])
        b = before_wav[:min_len]
        a = after_wav[:min_len]
        g = gt_wav[:min_len]

        row = {
            "mixture_id": mix_id,
            "sisdr_before": round(si_sdr(b, g), 3),
            "sisdr_after_reuse": round(si_sdr(a, g), 3),
        }
        if has_pesq:
            try:
                row["pesq_before"] = round(pesq_fn(SR, g.numpy(), b.numpy(), "wb"), 3)
                row["pesq_after_reuse"] = round(pesq_fn(SR, g.numpy(), a.numpy(), "wb"), 3)
            except Exception:
                pass
        if has_stoi:
            try:
                row["stoi_before"] = round(stoi_fn(g.numpy(), b.numpy(), SR, extended=False), 4)
                row["stoi_after_reuse"] = round(stoi_fn(g.numpy(), a.numpy(), SR, extended=False), 4)
            except Exception:
                pass
        results.append(row)
    return results


def evaluate_dihard():
    dh_dir = os.path.join(OUTPUT_DIR, "dihard")
    if not os.path.exists(dh_dir):
        print("  DIHARD results not found")
        return None

    results = []
    for rec_id in sorted(os.listdir(dh_dir)):
        rec_dir = os.path.join(dh_dir, rec_id)
        if not os.path.isdir(rec_dir):
            continue
        before_path = os.path.join(rec_dir, "before.wav")
        after_path = os.path.join(rec_dir, "after_reuse.wav")
        if not all(os.path.exists(p) for p in [before_path, after_path]):
            continue

        b, _ = load_mono(before_path)
        a, _ = load_mono(after_path)
        min_len = min(b.shape[-1], a.shape[-1])
        b, a = b[:min_len], a[:min_len]

        results.append({
            "recording": rec_id,
            "duration_s": round(min_len / SR, 1),
            "rms_before": round(rms(b), 4),
            "rms_after": round(rms(a), 4),
            "correlation": round(correlation(b, a), 4),
        })
    return results


def evaluate_youtube():
    yt_dir = os.path.join(OUTPUT_DIR, "youtube")
    if not os.path.exists(yt_dir):
        print("  YouTube results not found")
        return None

    before_path = os.path.join(yt_dir, "before.wav")
    after_path = os.path.join(yt_dir, "after_reuse.wav")
    if not all(os.path.exists(p) for p in [before_path, after_path]):
        return None

    b, _ = load_mono(before_path)
    a, _ = load_mono(after_path)
    min_len = min(b.shape[-1], a.shape[-1])
    b, a = b[:min_len], a[:min_len]

    return {
        "duration_s": round(min_len / SR, 1),
        "rms_before": round(rms(b), 4),
        "rms_after": round(rms(a), 4),
        "correlation": round(correlation(b, a), 4),
    }


def main():
    print("=" * 70)
    print("  RE-USE Speech Enhancement Experiment Results")
    print("=" * 70)

    print("\n--- LibriMix (50 samples, has clean ground truth) ---")
    lm = evaluate_librimix()
    if lm:
        sisdrs_b = [r["sisdr_before"] for r in lm]
        sisdrs_a = [r["sisdr_after_reuse"] for r in lm]
        print(f"  SI-SDR Before:      mean={np.mean(sisdrs_b):.2f}  std={np.std(sisdrs_b):.2f}")
        print(f"  SI-SDR After:       mean={np.mean(sisdrs_a):.2f}  std={np.std(sisdrs_a):.2f}")
        print(f"  SI-SDR Improvement: {np.mean(sisdrs_a) - np.mean(sisdrs_b):+.2f} dB")

        pesqs_b = [r["pesq_before"] for r in lm if "pesq_before" in r]
        pesqs_a = [r["pesq_after_reuse"] for r in lm if "pesq_after_reuse" in r]
        if pesqs_b:
            print(f"  PESQ Before:        mean={np.mean(pesqs_b):.3f}")
            print(f"  PESQ After:         mean={np.mean(pesqs_a):.3f}")
            print(f"  PESQ Improvement:   {np.mean(pesqs_a) - np.mean(pesqs_b):+.3f}")

        stois_b = [r["stoi_before"] for r in lm if "stoi_before" in r]
        stois_a = [r["stoi_after_reuse"] for r in lm if "stoi_after_reuse" in r]
        if stois_b:
            print(f"  STOI Before:        mean={np.mean(stois_b):.4f}")
            print(f"  STOI After:         mean={np.mean(stois_a):.4f}")
            print(f"  STOI Improvement:   {np.mean(stois_a) - np.mean(stois_b):+.4f}")

    print("\n--- DIHARD (DH_EVAL_0011 + 0012, no clean GT) ---")
    dh = evaluate_dihard()
    if dh:
        for r in dh:
            print(f"  {r['recording']} ({r['duration_s']}s):")
            print(f"    RMS  before={r['rms_before']:.4f}  after={r['rms_after']:.4f}")
            print(f"    Correlation: {r['correlation']:.4f}")

    print("\n--- YouTube Documentary (no clean GT) ---")
    yt = evaluate_youtube()
    if yt:
        print(f"  Duration: {yt['duration_s']}s")
        print(f"  RMS  before={yt['rms_before']:.4f}  after={yt['rms_after']:.4f}")
        print(f"  Correlation: {yt['correlation']:.4f}")

    print(f"\n{'='*70}")
    print("  Audio files saved in reuse_experiment/output/ for listening:")
    print("    youtube/   -> before.wav, after_reuse.wav")
    print("    dihard/    -> {rec_id}/before.wav, after_reuse.wav")
    print("    librimix/  -> before/, after_reuse/, clean_gt/, enrollment/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
