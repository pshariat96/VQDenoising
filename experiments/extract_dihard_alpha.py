import argparse
import os
import subprocess
import sys
from collections import defaultdict

import torchaudio

DIHARD_DIR = os.environ.get("DIHARD_DIR", "data/dihard")
CONFIG_PATH = "config/config_FlowTSE_large_noisy.yaml"
T_PREDICTER_CKPT = "t_predictor_noisy.ckpt"
OUTPUT_ROOT = "dihard_results"
SAMPLE_RATE = 16000
ENROLLMENT_DURATION = 3.0
CHUNK_SECONDS = 3.0

SWEEP_RUNS = [
    {"tag": "euler_step1p0", "solver_method": "euler", "test_step_size": 1.0},
    {"tag": "euler_step0p5", "solver_method": "euler", "test_step_size": 0.5},
    {"tag": "euler_step0p25", "solver_method": "euler", "test_step_size": 0.25},
    {"tag": "midpoint_step0p5", "solver_method": "midpoint", "test_step_size": 0.5},
    {"tag": "midpoint_step0p25", "solver_method": "midpoint", "test_step_size": 0.25},
]


def find_upstream_root():
    for root in sys.path:
        if os.path.exists(os.path.join(root or ".", "inference.py")):
            return root or "."
    raise SystemExit("Could not find inference.py. Clone AD-FlowTSE and add it to PYTHONPATH.")


def parse_rttm(rttm_path):
    segments = defaultdict(list)
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            segments[parts[7]].append((start, start + duration))
    return dict(segments)


def find_non_overlapping_segments(target_segments, other_segments, min_duration):
    all_others = sorted([seg for segs in other_segments for seg in segs])

    clean = []
    for t_start, t_end in target_segments:
        sub_segments = [(t_start, t_end)]
        for o_start, o_end in all_others:
            new_sub = []
            for s_start, s_end in sub_segments:
                if o_end <= s_start or o_start >= s_end:
                    new_sub.append((s_start, s_end))
                else:
                    if s_start < o_start:
                        new_sub.append((s_start, o_start))
                    if o_end < s_end:
                        new_sub.append((o_end, s_end))
            sub_segments = new_sub
        clean.extend(sub_segments)

    clean = [(s, e) for s, e in clean if (e - s) >= min_duration]
    clean.sort(key=lambda x: -(x[1] - x[0]))
    return clean


def extract_enrollment(audio, sr, segment_start, duration):
    start_sample = int(segment_start * sr)
    end_sample = start_sample + int(duration * sr)
    return audio[:, start_sample:end_sample]


def enrollment_quality_score(enrollment):
    rms = float(enrollment.pow(2).mean().sqrt())
    peak = float(enrollment.abs().max())
    return rms * (rms / max(peak, 1e-6))


def choose_best_enrollment_start(audio, sr, candidate_segments, duration):
    stride_sec = 0.25
    best = None

    for seg_start, seg_end in candidate_segments:
        seg_duration = seg_end - seg_start
        if seg_duration < duration:
            continue

        max_offset = seg_duration - duration
        if max_offset <= 1e-6:
            starts = [seg_start]
        else:
            n_steps = int(max_offset / stride_sec) + 1
            starts = [seg_start + i * stride_sec for i in range(n_steps + 1)]
            starts.append(seg_end - duration)

        for start in starts:
            start = min(start, seg_end - duration)
            enrollment = extract_enrollment(audio, sr, start, duration)
            score = enrollment_quality_score(enrollment)
            if best is None or score > best["score"]:
                best = {
                    "start": start,
                    "end": start + duration,
                    "score": score,
                    "rms": float(enrollment.pow(2).mean().sqrt()),
                    "peak": float(enrollment.abs().max()),
                }

    return best


def select_enrollments(audio, sr, speaker_segments):
    speakers = sorted(speaker_segments.keys())
    selected = {}

    for speaker in speakers:
        others = [speaker_segments[s] for s in speakers if s != speaker]
        clean_segs = find_non_overlapping_segments(
            speaker_segments[speaker], others, min_duration=ENROLLMENT_DURATION
        )

        if not clean_segs:
            print(f"  {speaker}: no non-overlapping segment >= {ENROLLMENT_DURATION}s, "
                  "falling back to longest available")
            clean_segs = sorted(speaker_segments[speaker], key=lambda x: -(x[1] - x[0]))
            if not clean_segs:
                print(f"  {speaker}: no segments at all, skipping")
                continue

        best_crop = choose_best_enrollment_start(audio, sr, clean_segs, ENROLLMENT_DURATION)
        if best_crop is None:
            print(f"  {speaker}: no valid {ENROLLMENT_DURATION}s crop, skipping")
            continue

        print(f"  {speaker}: enrollment = {best_crop['start']:.2f}s - {best_crop['end']:.2f}s "
              f"(rms={best_crop['rms']:.5f}, peak={best_crop['peak']:.5f})")
        selected[speaker] = extract_enrollment(
            audio, sr, best_crop["start"], ENROLLMENT_DURATION
        )

    return selected


def build_runs(enrollments, sr, output_root, solver=None):
    solver_args = []
    if solver is not None:
        solver_args = [
            "--solver_method", solver["solver_method"],
            "--test_step_size", str(solver["test_step_size"]),
        ]

    runs = []
    for speaker, enrollment in enrollments.items():
        speaker_dir = os.path.join(output_root, speaker)
        os.makedirs(speaker_dir, exist_ok=True)

        enroll_path = os.path.join(speaker_dir, "enrollment.wav")
        torchaudio.save(enroll_path, enrollment, sr)

        runs.append({
            "speaker": speaker,
            "label": "fixed (alpha=0.5)",
            "enrollment": enroll_path,
            "output": os.path.join(speaker_dir, "extracted_fixed.wav"),
            "args": ["--alpha", "0.5"] + solver_args,
        })
        runs.append({
            "speaker": speaker,
            "label": "adaptive (TPredicter)",
            "enrollment": enroll_path,
            "output": os.path.join(speaker_dir, "extracted_adaptive.wav"),
            "args": ["--t_predicter_ckpt", T_PREDICTER_CKPT] + solver_args,
        })
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file-id", default="DH_EVAL_0012")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    upstream = find_upstream_root()
    config_path = CONFIG_PATH
    if not os.path.exists(config_path):
        config_path = os.path.join(upstream, CONFIG_PATH)

    flac_path = os.path.join(DIHARD_DIR, "flac", f"{args.file_id}.flac")
    rttm_path = os.path.join(DIHARD_DIR, "rttm", f"{args.file_id}.rttm")

    print(f"Loading {flac_path}")
    audio, sr = torchaudio.load(flac_path)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    print(f"  {audio.shape[-1] / sr:.1f}s at {sr} Hz")

    speaker_segments = parse_rttm(rttm_path)
    print(f"Speakers: {', '.join(sorted(speaker_segments))}")

    enrollments = select_enrollments(audio, sr, speaker_segments)
    if not enrollments:
        raise SystemExit("No usable enrollments found.")

    record_root = os.path.join(OUTPUT_ROOT, args.file_id)
    if args.sweep:
        runs = []
        for sweep in SWEEP_RUNS:
            sweep_dir = os.path.join(record_root, "sweeps", sweep["tag"])
            for run in build_runs(enrollments, sr, sweep_dir, solver=sweep):
                run["label"] = f"{run['label']} -- {sweep['tag']}"
                runs.append(run)
    else:
        runs = build_runs(enrollments, sr, record_root)

    print()
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['speaker']} -- {run['label']}")
        print(f"  {run['output']}")

        cmd = [
            sys.executable, os.path.join(upstream, "inference.py"),
            "--config", config_path,
            "--mixture", flac_path,
            "--enrollment", run["enrollment"],
            "--output", run["output"],
            "--chunk_seconds", str(CHUNK_SECONDS),
        ] + run["args"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  failed (exit {result.returncode})")
            print(f"  {result.stderr[-500:]}")
        else:
            print("  done")

    print(f"\nResults in {record_root}/")


if __name__ == "__main__":
    main()
