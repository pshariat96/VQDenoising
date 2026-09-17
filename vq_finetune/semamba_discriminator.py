# Ported from RoyChao19477/SEMamba (models/discriminator.py, models/lsigmoid.py).
import numpy as np
import torch
import torch.nn as nn
from pesq import pesq
from joblib import Parallel, delayed


class LearnableSigmoid1D(nn.Module):

    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def pesq_loss(clean, noisy, sr=16000):
    try:
        score = pesq(sr, clean, noisy, "wb")
    except Exception:
        score = -1
    return score


def batch_pesq(clean, noisy, num_workers=8):
    scores = Parallel(n_jobs=num_workers)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    scores = np.array(scores)
    if -1 in scores:
        return None
    scores = (scores - 1) / 3.5
    return torch.FloatTensor(scores)


class MetricDiscriminator(nn.Module):

    def __init__(self, dim=16, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),
            nn.utils.spectral_norm(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True),
            nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim * 8, dim * 4)),
            nn.Dropout(0.3),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Linear(dim * 4, 1)),
            LearnableSigmoid1D(1),
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)
