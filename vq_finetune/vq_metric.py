import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf


VOICE_QUALITY_LABELS = [
    'shrill', 'nasal', 'deep',
    'silky', 'husky', 'raspy', 'guttural', 'vocal-fry',
    'booming', 'authoritative', 'loud', 'hushed', 'soft',
    'crisp', 'slurred', 'lisp', 'stammering',
    'singsong', 'pitchy', 'flowing', 'monotone', 'staccato',
    'punctuated', 'enunciated', 'hesitant',
]

NUM_LABELS = len(VOICE_QUALITY_LABELS)
MAX_DISTANCE = NUM_LABELS


class VoiceQualityMetric:

    def __init__(self, vox_profile_path, model_id="tiantiaf/whisper-large-v3-voice-quality",
                 device="cuda", sample_rate=16000):
        self.device = torch.device(device)
        self.sample_rate = sample_rate

        sys.path.insert(0, os.path.abspath(vox_profile_path))
        from src.model.voice_quality.whisper_voice_quality import WhisperWrapper

        self.model = WhisperWrapper.from_pretrained(model_id).float().to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def get_probs(self, waveform):
        if waveform.shape[0] < self.sample_rate * 1:
            return None

        max_len = 15 * self.sample_rate
        waveform = waveform[:max_len]

        data = waveform.unsqueeze(0).float().to(self.device)
        logits = self.model(data, return_feature=False)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs

    @torch.no_grad()
    def compute_distance(self, enhanced_wav, clean_wav):
        probs_enhanced = self.get_probs(enhanced_wav)
        probs_clean = self.get_probs(clean_wav)

        if probs_enhanced is None or probs_clean is None:
            return None

        l1_dist = np.abs(probs_enhanced - probs_clean).sum()
        normalized = l1_dist / MAX_DISTANCE
        return float(normalized)

    @torch.no_grad()
    def compute_batch_distance(self, enhanced_batch, clean_batch):
        B = enhanced_batch.shape[0]
        distances = []

        for i in range(B):
            dist = self.compute_distance(enhanced_batch[i].cpu(), clean_batch[i].cpu())
            if dist is None:
                dist = 0.5
            distances.append(dist)

        return torch.tensor(distances, dtype=torch.float32)

    def _setup_diff(self):
        if getattr(self, "_diff_ready", False):
            return
        fe = self.model.feature_extractor
        mf = torch.as_tensor(np.asarray(fe.mel_filters), dtype=torch.float32)

        n_mels = int(getattr(fe, "feature_size", 128))
        if mf.shape[0] != n_mels and mf.shape[1] == n_mels:
            mf = mf.t().contiguous()
        self._mel_filters = mf.to(self.device)
        self._n_fft = int(getattr(fe, "n_fft", 400))
        self._hop = int(getattr(fe, "hop_length", 160))
        self._chunk_samples = 15 * self.sample_rate
        self._window = torch.hann_window(self._n_fft, periodic=True).to(self.device)

        enc = self.model.backbone_model.encoder
        enc.embed_positions = enc.embed_positions.from_pretrained(
            self.model.embed_positions[:750].to(self.device))
        enc.to(self.device)

        for m in enc.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        for m in self.model.model_seq.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        for m in self.model.output_layer.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

        self._grad_ckpt = False
        try:
            enc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            enc.train()
            self._grad_ckpt = True
        except Exception:
            try:
                enc.gradient_checkpointing = True
                enc.train()
                self._grad_ckpt = True
            except Exception:
                enc.eval()
        self.model.model_seq.eval()
        self.model.output_layer.eval()
        self._diff_ready = True

    def _logmel_diff(self, wav):
        self._setup_diff()
        n = self._chunk_samples
        T = wav.shape[-1]
        if T < n:
            wav = F.pad(wav, (0, n - T))
        else:
            wav = wav[..., :n]
        stft = torch.stft(wav, self._n_fft, hop_length=self._hop, window=self._window,
                          center=True, return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        mel = self._mel_filters @ mag
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _encode_diff(self, wav):
        mel = self._logmel_diff(wav)
        enc = self.model.backbone_model.encoder
        return enc(mel, output_hidden_states=True).hidden_states

    def logits_diff(self, wav):
        hidden = self._encode_diff(wav)
        feats = hidden[-1].transpose(1, 2)
        feats = self.model.model_seq(feats).transpose(1, 2)
        pooled = feats.mean(dim=1)
        return self.model.output_layer(pooled)

    def _maybe_eot(self, enhanced, clean, eot):
        if not eot:
            return enhanced, clean

        shift = int(torch.randint(0, max(1, self.sample_rate // 10), (1,)).item())
        return torch.roll(enhanced, shift, dims=-1), torch.roll(clean, shift, dims=-1)

    def perceptual_loss(self, enhanced, clean, mode="feat", layers=None, eot=True):
        self._setup_diff()
        enhanced, clean = self._maybe_eot(enhanced, clean, eot)
        if mode == "out":
            le = self.logits_diff(enhanced)
            with torch.no_grad():
                lc = self.logits_diff(clean)
            return (torch.sigmoid(le) - torch.sigmoid(lc)).abs().mean()

        he = self._encode_diff(enhanced)
        with torch.no_grad():
            hc = self._encode_diff(clean)
        if layers is None:
            L = len(he)
            layers = sorted({L // 3, (2 * L) // 3, L - 1})
        loss = 0.0
        for l in layers:
            loss = loss + (he[l] - hc[l]).abs().mean()
        return loss / len(layers)

    @torch.no_grad()
    def validate_diff_mel(self, sample_wav, tol=1e-3):
        self._setup_diff()
        wav = sample_wav.to(self.device).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ours = self._logmel_diff(wav[:1])[0]
        fe = self.model.feature_extractor
        ref = fe(wav[0].detach().cpu().numpy(), return_tensors="pt",
                 sampling_rate=self.sample_rate,
                 max_length=self._chunk_samples).input_features[0].to(self.device)
        t = min(ours.shape[-1], ref.shape[-1])
        max_diff = (ours[..., :t] - ref[..., :t]).abs().max().item()
        return max_diff, (max_diff < tol)

    @torch.no_grad()
    def compute_label_distance(self, enhanced_wav, clean_wav, threshold=0.5):
        probs_enhanced = self.get_probs(enhanced_wav)
        probs_clean = self.get_probs(clean_wav)

        if probs_enhanced is None or probs_clean is None:
            return None, [], []

        labels_enhanced = set(
            VOICE_QUALITY_LABELS[i] for i, p in enumerate(probs_enhanced) if p > threshold
        )
        labels_clean = set(
            VOICE_QUALITY_LABELS[i] for i, p in enumerate(probs_clean) if p > threshold
        )

        added = sorted(labels_enhanced - labels_clean)
        removed = sorted(labels_clean - labels_enhanced)
        distance = len(added) + len(removed)

        return distance, added, removed
