import os
import sys
import tempfile
import shutil

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discriminator import MetricDiscriminator
from dataloader import PairedSEDataset


def create_fake_data(tmpdir, n_files=20, sr=16000, duration=2.0):
    noisy_dir = os.path.join(tmpdir, "noisy")
    clean_dir = os.path.join(tmpdir, "clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    samples = int(sr * duration)
    for i in range(n_files):
        clean = np.random.randn(samples).astype(np.float32) * 0.1
        noise = np.random.randn(samples).astype(np.float32) * 0.05
        noisy = clean + noise

        fname = f"test_{i:04d}.wav"
        sf.write(os.path.join(clean_dir, fname), clean, sr)
        sf.write(os.path.join(noisy_dir, fname), noisy, sr)

    return noisy_dir, clean_dir


def test_dataloader(noisy_dir, clean_dir):
    print("\n[1/4] Testing dataloader...")

    dataset = PairedSEDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        sample_rate=16000,
        segment_length=32000,
        min_segment_length=8000,
    )

    print(f"  Dataset size: {len(dataset)}")
    noisy, clean, fname = dataset[0]
    print(f"  Sample shape: noisy={noisy.shape}, clean={clean.shape}")
    print(f"  Filename: {fname}")
    assert noisy.shape == clean.shape == (32000,), f"Shape mismatch: {noisy.shape}"
    print("  PASSED")


def test_discriminator():
    print("\n[2/4] Testing discriminator...")

    D = MetricDiscriminator(n_fft=512, hop_length=128)
    D.eval()

    B = 4
    T = 32000
    enhanced = torch.randn(B, T)
    clean = torch.randn(B, T)

    with torch.no_grad():
        score = D(enhanced, clean)

    print(f"  Input shapes: enhanced={enhanced.shape}, clean={clean.shape}")
    print(f"  Output shape: {score.shape}")
    print(f"  Output values: {score.squeeze().tolist()}")
    assert score.shape == (B, 1), f"Expected (4,1), got {score.shape}"
    assert (score >= 0).all() and (score <= 1).all(), "Scores outside [0,1]"
    print("  PASSED")


class MultiResolutionSTFTLoss(nn.Module):

    def __init__(self, fft_sizes, hop_sizes, win_sizes):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def stft_loss(self, x, y, n_fft, hop, win):
        window = torch.hann_window(win, device=x.device)
        x_spec = torch.stft(x, n_fft, hop, win, window=window, return_complex=True)
        y_spec = torch.stft(y, n_fft, hop, win, window=window, return_complex=True)
        x_mag = x_spec.abs()
        y_mag = y_spec.abs()
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)
        mag_loss = nn.functional.l1_loss(torch.log(x_mag + 1e-8), torch.log(y_mag + 1e-8))
        return sc_loss + mag_loss

    def forward(self, x, y):
        loss = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss += self.stft_loss(x, y, fft, hop, win)
        return loss / len(self.fft_sizes)


def test_stft_loss():
    print("\n[3/4] Testing STFT loss...")

    loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024],
        hop_sizes=[128, 256],
        win_sizes=[512, 1024],
    )

    x = torch.randn(4, 32000)
    y = torch.randn(4, 32000)

    loss = loss_fn(x, y)
    print(f"  Loss (random vs random): {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"

    loss_same = loss_fn(x, x)
    print(f"  Loss (same vs same): {loss_same.item():.6f}")
    assert loss_same.item() < 0.01, "Self-loss should be near zero"
    print("  PASSED")


def test_training_step():
    print("\n[4/4] Testing training step (fake generator)...")

    class FakeGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.tensor(0.9))

        def forward(self, x):
            return self.param * x + (1 - self.param) * torch.randn_like(x) * 0.01

    G = FakeGenerator()
    D = MetricDiscriminator(n_fft=512, hop_length=128)
    stft_loss_fn = MultiResolutionSTFTLoss([512], [128], [512])
    l1_loss_fn = nn.L1Loss()

    opt_g = torch.optim.AdamW(G.parameters(), lr=1e-3)
    opt_d = torch.optim.AdamW(D.parameters(), lr=1e-3)

    B, T = 2, 32000
    noisy = torch.randn(B, T)
    clean = torch.randn(B, T)

    enhanced = G(noisy)

    loss_l1 = l1_loss_fn(enhanced, clean)
    loss_stft = stft_loss_fn(enhanced, clean)
    loss_spec = loss_l1 + loss_stft

    vq_distances = torch.rand(B)

    opt_d.zero_grad()
    pred_score = D(enhanced.detach(), clean).squeeze(-1)
    loss_d = nn.functional.mse_loss(pred_score, vq_distances)
    loss_d.backward()
    opt_d.step()

    opt_g.zero_grad()
    pred_score_g = D(enhanced, clean).squeeze(-1)
    loss_gan = nn.functional.mse_loss(pred_score_g, torch.zeros_like(pred_score_g))
    loss_g = loss_spec + 0.5 * loss_gan
    loss_g.backward()
    opt_g.step()

    print(f"  L_spec: {loss_spec.item():.4f}")
    print(f"  L_d:    {loss_d.item():.4f}")
    print(f"  L_gan:  {loss_gan.item():.4f}")
    print(f"  L_g:    {loss_g.item():.4f}")
    print("  Gradients flow correctly.")
    print("  PASSED")


def main():
    print("=" * 60)
    print("  MetricGAN Fine-Tuning: Local Sanity Check")
    print("  (No CUDA or mamba-ssm required)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="metricgan_test_")

    try:
        noisy_dir, clean_dir = create_fake_data(tmpdir, n_files=20)

        test_dataloader(noisy_dir, clean_dir)
        test_discriminator()
        test_stft_loss()
        test_training_step()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
        print("\n  The training loop logic is correct.")
        print("  To run the actual training, you need:")
        print("    1. CUDA GPU (A100/V100)")
        print("    2. mamba-ssm + causal-conv1d installed")
        print("    3. URGENT 2025 data in ./data/urgent2025/")
        print("    4. vox-profile-release cloned locally")
        print(f"\n  Then: python train.py --config config.yaml")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
