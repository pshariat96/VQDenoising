from google.colab import drive
drive.mount('/content/drive')

import torch
print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")


import os, sys, math, torch, torch.nn as nn, torchaudio, numpy as np, random
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(repo_id="nvidia/RE-USE", local_dir="./REUSE", local_dir_use_symlinks=False)

sys.path.insert(0, './REUSE')
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config, pad_or_trim_to_match

RELU = nn.ReLU()
device = torch.device('cuda')

config_path = hf_hub_download(repo_id='nvidia/RE-USE', filename='config.json')
cfg = load_config(config_path)
n_fft = cfg['stft_cfg']['n_fft']
hop_size = cfg['stft_cfg']['hop_size']
win_size = cfg['stft_cfg']['win_size']
compress_factor = cfg['model_cfg']['compress_factor']
sampling_rate = cfg['stft_cfg']['sampling_rate']

SE_model = SEMamba.from_pretrained('nvidia/RE-USE', cfg=cfg).to(device)
SE_model.eval()
print("RE-USE loaded.")

def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1

def enhance_short(wav_tensor, sr, model, device):
    wav = wav_tensor.to(device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    n_fft_s = make_even(n_fft * sr // sampling_rate)
    hop_s = make_even(hop_size * sr // sampling_rate)
    win_s = make_even(win_size * sr // sampling_rate)
    with torch.no_grad():
        noisy_mag, noisy_pha, _ = mag_phase_stft(wav, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s, compress_factor=compress_factor, center=True, addeps=False)
        amp_g, pha_g, _ = model(noisy_mag, noisy_pha)
        mag = torch.expm1(RELU(amp_g))
        zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0
        audio_g = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, compress_factor)
        audio_g = pad_or_trim_to_match(wav.detach(), audio_g, pad_value=1e-8)
    return audio_g.cpu()


from google.colab import drive
drive.mount('/content/drive', force_remount=False)

DRIVE_BASE = '/content/drive/MyDrive/Libri2Mix_Official_Test/sep_clean'
DRIVE_S1_CLEAN = os.path.join(DRIVE_BASE, 's1')
DRIVE_S1_NOISY_20 = os.path.join(DRIVE_BASE, 's1_noisy_20')
DRIVE_S1_NOISY_40 = os.path.join(DRIVE_BASE, 's1_noisy_40')

OUTPUT_DIR = '/content/voice_quality_reuse_output'

random.seed(42)
clean_ids = sorted(f.replace('.wav', '') for f in os.listdir(DRIVE_S1_CLEAN) if f.endswith('.wav'))
noisy20_ids = set(f.replace('.wav', '') for f in os.listdir(DRIVE_S1_NOISY_20) if f.endswith('.wav'))
noisy40_ids = set(f.replace('.wav', '') for f in os.listdir(DRIVE_S1_NOISY_40) if f.endswith('.wav'))

common = [x for x in clean_ids if x in noisy20_ids and x in noisy40_ids]
selected = random.sample(common, 10)
print(f"Selected 10 samples: {selected[:3]}...")

CONDITIONS = {
    'clean_s1': DRIVE_S1_CLEAN,
    's1_noisy_20': DRIVE_S1_NOISY_20,
    's1_noisy_40': DRIVE_S1_NOISY_40,
}

for condition, src_dir in CONDITIONS.items():
    before_dir = os.path.join(OUTPUT_DIR, condition, 'before')
    after_dir = os.path.join(OUTPUT_DIR, condition, 'after_reuse')
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    print(f"\nProcessing {condition}...")
    for mix_id in tqdm(selected):
        src_path = os.path.join(src_dir, f"{mix_id}.wav")
        if not os.path.exists(src_path):
            print(f"  SKIP: {mix_id}")
            continue

        wav, sr = torchaudio.load(src_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        enhanced = enhance_short(wav, sr, SE_model, device)

        torchaudio.save(os.path.join(before_dir, f"{mix_id}.wav"), wav, sr)
        torchaudio.save(os.path.join(after_dir, f"{mix_id}.wav"), enhanced, sr)


MIX_DIR = '/content/drive/MyDrive/Libri2Mix_Official_Test/mix_both'
if os.path.exists(MIX_DIR):
    mix_out = os.path.join(OUTPUT_DIR, 'mixture')
    os.makedirs(mix_out, exist_ok=True)
    for mix_id in selected[:10]:
        src = os.path.join(MIX_DIR, f"{mix_id}.wav")
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(mix_out, f"{mix_id}.wav"))
    print(f"\nCopied {len(os.listdir(mix_out))} mixture files")

print(f"\nDone! Output at: {OUTPUT_DIR}")
print(f"Total files: {sum(len(os.listdir(os.path.join(r,d))) for r,ds,_ in os.walk(OUTPUT_DIR) for d in ds if os.path.isdir(os.path.join(r,d)))}")


import shutil

manifest_path = os.path.join(OUTPUT_DIR, 'selected_ids.txt')
with open(manifest_path, 'w') as f:
    for mid in selected:
        f.write(f"{mid}\n")

DRIVE_SAVE = '/content/drive/MyDrive/AD-FlowTSE_checkpoints/voice_quality_reuse_output'
if os.path.exists(DRIVE_SAVE):
    shutil.rmtree(DRIVE_SAVE)
shutil.copytree(OUTPUT_DIR, DRIVE_SAVE)
print(f"Saved to Drive: {DRIVE_SAVE}")
print(f"\nDownload and place at: AD-FlowTSE/reuse_libri1mix_experiment/output/")
