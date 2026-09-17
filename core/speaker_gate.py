import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ecapa_tdnn import ECAPA_TDNN


class SpeakerVerificationGate(nn.Module):
    def __init__(self, C=1024):
        super().__init__()
        self.ecapa_tdnn = ECAPA_TDNN(C=C)
        self.classifier = nn.Sequential(
            nn.Linear(192 * 2 + 1, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, mixture, enrollment, aug=False):
        mix_emb = self.ecapa_tdnn(mixture, aug)
        enr_emb = self.ecapa_tdnn(enrollment, aug)

        mix_norm = F.normalize(mix_emb, dim=-1)
        enr_norm = F.normalize(enr_emb, dim=-1)
        cosine = (mix_norm * enr_norm).sum(dim=-1, keepdim=True)

        features = torch.cat([mix_emb, enr_emb, cosine], dim=-1)
        logits = self.classifier(features).squeeze(-1)
        return logits

    def predict(self, mixture, enrollment):
        with torch.no_grad():
            logits = self.forward(mixture, enrollment, aug=False)
            return torch.sigmoid(logits)
