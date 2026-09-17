import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torchaudio


DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
FILE_ID = "DH_EVAL_0012"

FLAC_PATH = os.path.join(DIHARD_DIR, "flac", f"{FILE_ID}.flac")
RTTM_PATH = os.path.join(DIHARD_DIR, "rttm", f"{FILE_ID}.rttm")
RESULTS_DIR = os.path.join("dihard_results", FILE_ID)

SAMPLE_RATE = 16000


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


def load_audio(path):
    audio, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.squeeze(0).numpy()


def rms(signal):
    return np.sqrt(np.mean(signal ** 2))


def energy_db(signal):
    r = rms(signal)
    if r < 1e-10:
        return -100.0
    return 20 * np.log10(r)


def active_ratio(signal, threshold_db=-40):
    frame_len = int(0.025 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    n_frames = max(1, (len(signal) - frame_len) // hop)
    active = 0
    for i in range(n_frames):
        frame = signal[i * hop: i * hop + frame_len]
        if energy_db(frame) > threshold_db:
            active += 1
    return active / n_frames


def spectral_centroid(signal, sr):
    spec = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    if spec.sum() < 1e-10:
        return 0.0
    return np.sum(freqs * spec) / np.sum(spec)


def signal_stats(signal, label):
    return {
        "label": label,
        "rms": rms(signal),
        "energy_db": energy_db(signal),
        "peak": np.max(np.abs(signal)),
        "active_ratio": active_ratio(signal),
        "spectral_centroid": spectral_centroid(signal, SAMPLE_RATE),
    }


def format_stats_table(stats_list):
    lines = []
    lines.append("| Signal | RMS | Energy (dB) | Peak | Active % | Spectral Centroid (Hz) |")
    lines.append("|--------|-----|-------------|------|----------|------------------------|")
    for s in stats_list:
        lines.append(
            f"| {s['label']} | {s['rms']:.4f} | {s['energy_db']:.1f} | "
            f"{s['peak']:.4f} | {s['active_ratio']*100:.1f}% | {s['spectral_centroid']:.0f} |"
        )
    return "\n".join(lines)


def main():
    print(f"Generating report for {FILE_ID}...")

    speaker_segments = parse_rttm(RTTM_PATH)
    speakers = sorted(speaker_segments.keys())
    mixture = load_audio(FLAC_PATH)
    total_duration = len(mixture) / SAMPLE_RATE

    report = []
    report.append(f"# AD-FlowTSE Extraction Report -- {FILE_ID}")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    report.append("## Recording Info\n")
    report.append(f"- **File**: `{FILE_ID}.flac`")
    report.append(f"- **Duration**: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    report.append(f"- **Sample rate**: {SAMPLE_RATE} Hz")
    report.append(f"- **Speakers**: {len(speakers)} -- {', '.join(speakers)}")
    report.append(f"- **Model**: FlowTSE (noisy) with UDiT backbone")
    report.append(f"- **Alpha modes**: fixed (0.5) and adaptive (TPredicter)\n")

    report.append("## Speaker Activity Summary\n")
    report.append("| Speaker | Segments | Total Speech (s) | Avg Segment (s) | Longest (s) |")
    report.append("|---------|----------|-------------------|------------------|-------------|")
    for speaker in speakers:
        segs = speaker_segments[speaker]
        durations = [e - s for s, e in segs]
        report.append(
            f"| {speaker} | {len(segs)} | {sum(durations):.1f} | "
            f"{np.mean(durations):.1f} | {max(durations):.1f} |"
        )
    report.append("")

    for speaker in speakers:
        spk_dir = os.path.join(RESULTS_DIR, speaker)
        fixed_path = os.path.join(spk_dir, "extracted_fixed.wav")
        adaptive_path = os.path.join(spk_dir, "extracted_adaptive.wav")
        enrollment_path = os.path.join(spk_dir, "enrollment.wav")

        missing = [p for p in [fixed_path, adaptive_path, enrollment_path]
                   if not os.path.exists(p)]
        if missing:
            report.append(f"## {speaker}\n")
            report.append(f"**Skipped** -- missing files: {missing}\n")
            continue

        extracted_fixed = load_audio(fixed_path)
        extracted_adaptive = load_audio(adaptive_path)
        enrollment = load_audio(enrollment_path)

        report.append(f"## {speaker}\n")

        report.append(f"**Enrollment**: {len(enrollment)/SAMPLE_RATE:.1f}s segment "
                      f"(RMS={rms(enrollment):.4f})\n")

        report.append("### Signal Statistics\n")
        stats = [
            signal_stats(mixture, "Mixture (original)"),
            signal_stats(extracted_fixed, "Extracted (fixed alpha=0.5)"),
            signal_stats(extracted_adaptive, "Extracted (adaptive)"),
        ]
        report.append(format_stats_table(stats))
        report.append("")

        mix_e = energy_db(mixture)
        fix_e = energy_db(extracted_fixed)
        ada_e = energy_db(extracted_adaptive)
        report.append("### Energy Change from Mixture\n")
        report.append(f"- Fixed alpha:    **{fix_e - mix_e:+.1f} dB**")
        report.append(f"- Adaptive alpha: **{ada_e - mix_e:+.1f} dB**\n")

        min_len = min(len(extracted_fixed), len(extracted_adaptive))
        corr = np.corrcoef(
            extracted_fixed[:min_len], extracted_adaptive[:min_len]
        )[0, 1]
        report.append(f"### Fixed vs Adaptive Correlation\n")
        report.append(f"Pearson correlation: **{corr:.4f}**")
        report.append("(1.0 = identical outputs, lower = more divergence between modes)\n")

    report.append("## Notes\n")
    report.append("- No ground-truth clean source is available for DIHARD, so standard "
                  "metrics (SI-SDR, PESQ, STOI) cannot be computed against a reference.")
    report.append("- The model was trained on LibriMix (synthetic 2-speaker, read speech). "
                  "Performance on real-world conversational DIHARD data may differ.")
    report.append("- **Active %** measures the fraction of 25ms frames above -40 dB. "
                  "A lower value after extraction suggests background suppression.")
    report.append("- **Spectral centroid** shift indicates whether the model altered the "
                  "frequency balance of the extracted speech.")
    report.append("- Listen to the output audio files for subjective quality assessment.\n")

    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
