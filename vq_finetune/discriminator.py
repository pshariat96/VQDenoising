# MetricGAN discriminator, after Fu et al., ICML 2019 (arxiv.org/abs/1905.04874).
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class MetricDiscriminator(nn.Module):

    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_freq = n_fft // 2 + 1

        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 15, kernel_size=(5, 5), padding=2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(15, 25, kernel_size=(7, 7), padding=3)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(25, 40, kernel_size=(9, 9), padding=4)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(40, 50, kernel_size=(11, 11), padding=5)),
            nn.LeakyReLU(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_layers = nn.Sequential(
            spectral_norm(nn.Linear(50, 50)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(50, 10)),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def compute_spectrogram(self, waveform):
        window = torch.hann_window(self.n_fft, device=waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        return spec.abs()

    def forward(self, enhanced_wav, clean_wav):
        enhanced_spec = self.compute_spectrogram(enhanced_wav)
        clean_spec = self.compute_spectrogram(clean_wav)

        min_frames = min(enhanced_spec.shape[-1], clean_spec.shape[-1])
        enhanced_spec = enhanced_spec[..., :min_frames]
        clean_spec = clean_spec[..., :min_frames]

        x = torch.stack([enhanced_spec, clean_spec], dim=1)

        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        score = self.fc_layers(x)
        return score
