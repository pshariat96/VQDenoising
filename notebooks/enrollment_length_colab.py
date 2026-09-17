import os
import shutil

os.chdir("/content")

print("=== [1/4] Installing dependencies ===")
os.system("pip install -q torch torchaudio einops pesq pystoi soundfile scipy pandas pyyaml torchdiffeq")

print("=== [2/4] Cloning repos ===")
PROJECT_DIR = "/content/AD-FlowTSE"
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
os.system(f"git clone --recursive https://github.com/aleXiehta/AD-FlowTSE.git {PROJECT_DIR} > /dev/null 2>&1")
os.system("pip install -q flow_matching > /dev/null 2>&1")


init_file = f"{PROJECT_DIR}/models/__init__.py"
with open(init_file, "w") as f:
    f.write("""from .udit.udit import UDiT
from models.ecapa_tdnn import ECAPA_TDNN

try:
    from .speakerbeam.td_speakerbeam import TimeDomainSpeakerBeam
    from .spex_plus.spex_plus import SpEx_Plus
except ImportError:
    pass
""")

print("=== [3/4] Generating LibriMix test set ===")
BUILD_DIR = "/content/librimix_build"
SOURCES_DIR = f"{BUILD_DIR}/sources"
os.makedirs(SOURCES_DIR, exist_ok=True)


LS_TAR = f"{SOURCES_DIR}/test-clean.tar.gz"
if not os.path.exists(f"{SOURCES_DIR}/LibriSpeech/test-clean"):
    if not os.path.exists(LS_TAR):
        print("  Downloading LibriSpeech test-clean (~350MB)...")
        os.system(f"wget -q -c https://www.openslr.org/resources/12/test-clean.tar.gz -P {SOURCES_DIR}")
    print("  Extracting...")
    os.system(f"tar -xzf {LS_TAR} -C {SOURCES_DIR}")


WHAM_ZIP = f"{SOURCES_DIR}/wham_noise.zip"
if not os.path.exists(f"{SOURCES_DIR}/wham_noise/tt"):
    if not os.path.exists(WHAM_ZIP):
        print("  Downloading WHAM noise (~18GB, takes a few minutes)...")
        os.system(f"wget -c https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip -P {SOURCES_DIR}")
    print("  Extracting test noise (tt/) and metadata...")
    os.system(f'unzip -q -o {WHAM_ZIP} "wham_noise/tt/*" "wham_noise/metadata/*" -d {SOURCES_DIR}')


LIBRIMIX_REPO = f"{BUILD_DIR}/LibriMix"
if not os.path.exists(LIBRIMIX_REPO):
    os.system(f"git clone https://github.com/JorisCos/LibriMix.git {LIBRIMIX_REPO} > /dev/null 2>&1")
os.system(f"pip install -q -r {LIBRIMIX_REPO}/requirements.txt > /dev/null 2>&1")


OUTPUT_DIR = f"{BUILD_DIR}/output"
LS_PATH = f"{SOURCES_DIR}/LibriSpeech"
WHAM_PATH = f"{SOURCES_DIR}/wham_noise"

os.chdir(LIBRIMIX_REPO)
os.system(f"""python scripts/create_librimix_from_metadata.py \
    --librispeech_dir {LS_PATH} \
    --wham_dir {WHAM_PATH} \
    --metadata_dir metadata/Libri2Mix \
    --librimix_outdir {OUTPUT_DIR} \
    --n_src 2 \
    --freqs 16k \
    --modes min \
    --types mix_clean mix_both""")


DATASET_DIR = "/content/Libri2Mix_Test"
GENERATED_AUDIO = f"{OUTPUT_DIR}/Libri2Mix/wav16k/min/test"
os.makedirs(f"{DATASET_DIR}/sep_clean", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/metadata", exist_ok=True)

for subdir in ["mix_clean", "mix_both", "s1", "s2", "noise"]:
    src = os.path.join(GENERATED_AUDIO, subdir)
    dst = os.path.join(DATASET_DIR, "sep_clean", subdir)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  {subdir}: {len(os.listdir(dst))} files")

shutil.copy(
    f"{LIBRIMIX_REPO}/metadata/Libri2Mix/libri2mix_test-clean.csv",
    f"{DATASET_DIR}/metadata/test.csv",
)


for d in ["mix_clean", "s1", "s2"]:
    n = len([f for f in os.listdir(f"{DATASET_DIR}/sep_clean/{d}") if f.endswith(".wav")])
    print(f"  Verified {d}: {n} files")

print("=== [4/4] Mounting Drive for checkpoint ===")
from google.colab import drive
drive.mount("/content/drive")

CKPT_SRC = "/content/drive/MyDrive/libri2mix_noisy.ckpt"
CKPT_DST = f"{PROJECT_DIR}/libri2mix_noisy.ckpt"
if os.path.exists(CKPT_SRC):
    shutil.copy(CKPT_SRC, CKPT_DST)
    print(f"  Checkpoint copied to {CKPT_DST}")
else:
    print(f"  WARNING: Checkpoint not found at {CKPT_SRC}")
    print("  Please upload libri2mix_noisy.ckpt to your Drive root and re-run this cell.")

print("\nSetup complete!")


import os
import sys
import csv
import random
import yaml
from collections import defaultdict

import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm

os.chdir("/content/AD-FlowTSE")
sys.path.insert(0, "/content/AD-FlowTSE")

from models.udit.udit import UDiT
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver


DATASET_DIR = "/content/Libri2Mix_Test"
CONFIG_PATH = "/content/AD-FlowTSE/config/config_FlowTSE_large_noisy.yaml"
CKPT_PATH = "/content/AD-FlowTSE/libri2mix_noisy.ckpt"
OUTPUT_DIR = "/content/enrollment_length_results"
N_SAMPLES = 200
SEED = 42
ALPHA = 0.5
CHUNK_BATCH_SIZE = 64
ENROLLMENT_LENGTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]

random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SR = config["dataset"]["sample_rate"]
N_FFT = config["dataset"]["n_fft"]
HOP_LENGTH = config["dataset"]["hop_length"]
WIN_LENGTH = config["dataset"]["win_length"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def load_flowtse_model(ckpt_path, model_config, device):
    model = UDiT(**model_config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def pad_and_reshape(tensor, multiple):
    n, d, l = tensor.shape
    padding_length = (multiple - (l % multiple)) % multiple
    padded = torch.nn.functional.pad(tensor, (0, padding_length))
    chunks = torch.cat(torch.chunk(padded, padded.shape[-1] // multiple, dim=-1), dim=0)
    return chunks, l


def reshape_and_remove_padding(tensor, original_length):
    n_k, d, multiple = tensor.shape
    n = original_length // multiple + (1 if original_length % multiple != 0 else 0)
    combined = torch.cat(torch.chunk(tensor, n, dim=0), dim=-1)
    return combined[:, :, :original_length]


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0)


def si_sdr(estimate, reference):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    dot = torch.sum(estimate * reference)
    s_target = dot * reference / (torch.sum(reference ** 2) + 1e-8)
    e_noise = estimate - s_target
    return (10 * torch.log10(
        torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + 1e-8) + 1e-8
    )).item()


def try_pesq(estimate_np, reference_np, sr=16000):
    try:
        from pesq import pesq as pesq_fn
        return pesq_fn(sr, reference_np, estimate_np, "wb")
    except Exception:
        return None


def extract_speaker(model, mixture_wav, enrollment_wav, device,
                    alpha=0.5, chunk_batch_size=64):
    mixture_spec = stft_torch(mixture_wav, N_FFT, HOP_LENGTH, WIN_LENGTH).unsqueeze(0)
    enrollment_spec = (
        stft_torch(enrollment_wav, N_FFT, HOP_LENGTH, WIN_LENGTH)
        .unsqueeze(0).to(device)
    )

    frames_per_chunk = SR * 3 // HOP_LENGTH + 1
    mixture_chunks, orig_len = pad_and_reshape(mixture_spec, frames_per_chunk)
    num_chunks = mixture_chunks.shape[0]

    solver_method = config["solver"]["method"]
    solver_step = config["solver"]["test_step_size"]
    alpha_grid = torch.tensor([alpha, 1.0], device=device)

    all_outputs = []
    with torch.no_grad():
        for i in range(0, num_chunks, chunk_batch_size):
            batch = mixture_chunks[i:i + chunk_batch_size].to(device)
            bs = batch.shape[0]
            solver = ODESolver(velocity_model=model)
            out = solver.sample(
                time_grid=alpha_grid,
                x_init=batch.float(),
                method=solver_method,
                step_size=solver_step,
                enrollment=enrollment_spec.repeat(bs, 1, 1),
            )
            all_outputs.append(out.cpu())

    source_hat_spec = torch.cat(all_outputs, dim=0)
    source_hat_spec = reshape_and_remove_padding(source_hat_spec, orig_len)
    source_hat = istft_torch(source_hat_spec, N_FFT, HOP_LENGTH, WIN_LENGTH,
                             length=mixture_wav.shape[-1])
    max_val = source_hat.abs().max()
    if max_val > 1.0:
        source_hat = source_hat / max_val
    return source_hat


sep = os.path.join(DATASET_DIR, "sep_clean")
mix_dir = os.path.join(sep, "mix_clean")
s1_dir = os.path.join(sep, "s1")
s2_dir = os.path.join(sep, "s2")

mix_ids = {f.replace(".wav", "") for f in os.listdir(mix_dir) if f.endswith(".wav")}
s1_ids = {f.replace(".wav", "") for f in os.listdir(s1_dir) if f.endswith(".wav")}
s2_ids = {f.replace(".wav", "") for f in os.listdir(s2_dir) if f.endswith(".wav")}
valid_ids = sorted(mix_ids & s1_ids & s2_ids)

spk_map = defaultdict(list)
for mid in valid_ids:
    spk_map[mid.split("-")[0]].append(mid)

eligible = [mid for mid in valid_ids if len(spk_map[mid.split("-")[0]]) >= 2]
print(f"Total valid samples: {len(valid_ids)}")
print(f"Eligible (speaker has >=2 utterances): {len(eligible)}")

selected = random.sample(eligible, min(N_SAMPLES, len(eligible)))
print(f"Selected {len(selected)} samples for experiment")


enroll_source_info = {}
for mix_id in tqdm(selected, desc="Scanning enrollment sources"):
    spk1 = mix_id.split("-")[0]
    others = [m for m in spk_map[spk1] if m != mix_id]
    enroll_id = random.choice(others)
    enroll_path = os.path.join(s1_dir, f"{enroll_id}.wav")
    enroll_wav = load_audio(enroll_path, SR)
    enroll_source_info[mix_id] = {
        "enroll_id": enroll_id,
        "enroll_wav": enroll_wav,
        "max_seconds": enroll_wav.shape[-1] / SR,
    }


print(f"\nLoading FlowTSE model from {CKPT_PATH}...")
model = load_flowtse_model(CKPT_PATH, config["model"], device)
print("Model loaded!\n")


all_results = []

for enroll_secs in ENROLLMENT_LENGTHS:
    enroll_samples = int(SR * enroll_secs)
    eligible_for_length = [
        mid for mid in selected
        if enroll_source_info[mid]["enroll_wav"].shape[-1] >= enroll_samples
    ]
    print(f"\n--- Enrollment length: {enroll_secs}s ({len(eligible_for_length)}/{len(selected)} eligible) ---")

    if len(eligible_for_length) == 0:
        print("  No eligible samples, skipping.")
        continue

    for mix_id in tqdm(eligible_for_length, desc=f"{enroll_secs}s"):

        mixture_wav = load_audio(os.path.join(mix_dir, f"{mix_id}.wav"), SR)
        gt_wav = load_audio(os.path.join(s1_dir, f"{mix_id}.wav"), SR)

        raw_enroll = enroll_source_info[mix_id]["enroll_wav"]
        enrollment_wav = raw_enroll[:enroll_samples]

        if enrollment_wav.shape[-1] < SR * 3:
            enrollment_wav = torch.nn.functional.pad(
                enrollment_wav, (0, SR * 3 - enrollment_wav.shape[-1])
            )

        extracted = extract_speaker(
            model, mixture_wav, enrollment_wav, device,
            alpha=ALPHA, chunk_batch_size=CHUNK_BATCH_SIZE,
        )

        min_len = min(gt_wav.shape[-1], extracted.shape[-1])
        gt_t = gt_wav[:min_len]
        ext_t = extracted.squeeze()[:min_len]

        sisdr_val = si_sdr(ext_t, gt_t)
        pesq_val = try_pesq(ext_t.numpy(), gt_t.numpy(), SR)

        all_results.append({
            "mixture_id": mix_id,
            "enrollment_seconds": enroll_secs,
            "sisdr_dB": round(sisdr_val, 3),
            "pesq": round(pesq_val, 3) if pesq_val is not None else None,
        })

    length_results = [r for r in all_results if r["enrollment_seconds"] == enroll_secs]
    sisdrs = [r["sisdr_dB"] for r in length_results]
    pesqs = [r["pesq"] for r in length_results if r["pesq"] is not None]
    print(f"  SI-SDR: mean={np.mean(sisdrs):.2f}, median={np.median(sisdrs):.2f}")
    if pesqs:
        print(f"  PESQ:   mean={np.mean(pesqs):.3f}, median={np.median(pesqs):.3f}")


csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f"\nSaved {len(all_results)} results to {csv_path}")


import pandas as pd

df = pd.read_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))


agg = df.groupby("enrollment_seconds").agg(
    sisdr_mean=("sisdr_dB", "mean"),
    sisdr_std=("sisdr_dB", "std"),
    sisdr_median=("sisdr_dB", "median"),
    pesq_mean=("pesq", "mean"),
    pesq_std=("pesq", "std"),
    pesq_median=("pesq", "median"),
    n_samples=("sisdr_dB", "count"),
).reset_index()

print("\nAggregated Results:")
print(agg.to_string(index=False))




summary_lines = [
    "=" * 65,
    "  ENROLLMENT LENGTH vs QUALITY -- SUMMARY",
    "=" * 65,
    f"  Total samples selected: {N_SAMPLES}",
    f"  Alpha (fixed): {ALPHA}",
    f"  Solver: {config['solver']['method']}, step={config['solver']['test_step_size']}",
    "",
    f"  {'Length':>6s}  {'N':>5s}  {'SI-SDR Mean':>11s}  {'SI-SDR Med':>10s}  {'PESQ Mean':>9s}  {'PESQ Med':>8s}",
    "  " + "-" * 58,
]
for _, row in agg.iterrows():
    pesq_m = f"{row['pesq_mean']:.3f}" if pd.notna(row["pesq_mean"]) else "N/A"
    pesq_med = f"{row['pesq_median']:.3f}" if pd.notna(row["pesq_median"]) else "N/A"
    summary_lines.append(
        f"  {row['enrollment_seconds']:>5.1f}s  {int(row['n_samples']):>5d}  "
        f"{row['sisdr_mean']:>10.2f}  {row['sisdr_median']:>10.2f}  "
        f"{pesq_m:>9s}  {pesq_med:>8s}"
    )
summary_lines.append("=" * 65)

summary_text = "\n".join(summary_lines)
print(summary_text)

summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved summary to {summary_path}")


DRIVE_SAVE = "/content/drive/MyDrive/enrollment_length_results"
os.makedirs(DRIVE_SAVE, exist_ok=True)
for fname in ["metrics.csv", "summary.txt"]:
    src = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DRIVE_SAVE, fname))
print(f"Results also saved to {DRIVE_SAVE}")
