import os
import csv
import argparse

import numpy as np
import yaml
import torch
import librosa
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import wilcoxon


from evaluate import build_test_dataset, load_ft_model
from train import load_reuse_model, enhance_batch, si_sdr
from vq_metric import VoiceQualityMetric

try:
    from pesq import pesq as _pesq
except Exception:
    _pesq = None


class DNSMOS:
    SR = 16000
    INPUT_LENGTH = 9.01

    def __init__(self, primary_path, p808_path, use_gpu=False):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        prov = ['CPUExecutionProvider']
        if use_gpu and 'CUDAExecutionProvider' in avail:
            prov = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(primary_path, providers=prov)
        self.p808 = ort.InferenceSession(p808_path, providers=prov)
        self.in_name = self.sess.get_inputs()[0].name
        self.p808_in = self.p808.get_inputs()[0].name

    def _melspec(self, audio, n_mels=120, frame_size=320, hop=160):
        m = librosa.feature.melspectrogram(
            y=audio, sr=self.SR, n_fft=frame_size + 1, hop_length=hop, n_mels=n_mels)
        return ((librosa.power_to_db(m, ref=np.max) + 40) / 40).T

    @staticmethod
    def _polyfit(sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def __call__(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n = int(self.INPUT_LENGTH * self.SR)
        while len(audio) < n:
            audio = np.append(audio, audio)
        hops = int(np.floor(len(audio) / self.SR) - self.INPUT_LENGTH) + 1
        S, B, O, P = [], [], [], []
        for i in range(max(hops, 1)):
            seg = audio[int(i * self.SR): int((i + self.INPUT_LENGTH) * self.SR)]
            if len(seg) < n:
                continue
            feat = seg.astype('float32')[np.newaxis, :]
            p808_feat = self._melspec(seg[:-160]).astype('float32')[np.newaxis, :, :]
            p808_mos = self.p808.run(None, {self.p808_in: p808_feat})[0][0][0]
            sig_raw, bak_raw, ovr_raw = self.sess.run(None, {self.in_name: feat})[0][0]
            s, b, o = self._polyfit(sig_raw, bak_raw, ovr_raw)
            S.append(s); B.append(b); O.append(o); P.append(p808_mos)
        if not O:
            return dict(SIG=np.nan, BAK=np.nan, OVRL=np.nan, P808=np.nan)
        return dict(SIG=float(np.mean(S)), BAK=float(np.mean(B)),
                    OVRL=float(np.mean(O)), P808=float(np.mean(P)))


def _to16k(x, sr):
    x = np.asarray(x, dtype=np.float32)
    return x if sr == 16000 else librosa.resample(x, orig_sr=sr, target_sr=16000)


def _pesq_wb(clean16k, deg16k):
    if _pesq is None:
        return np.nan
    L = min(len(clean16k), len(deg16k))
    try:
        return float(_pesq(16000, clean16k[:L], deg16k[:L], 'wb'))
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--ckpt', action='append', required=True,
                    help='LABEL=PATH to a checkpoint (repeatable).')
    ap.add_argument('--dnsmos-dir', default='../urgent2025_challenge/DNSMOS/DNSMOS')
    ap.add_argument('--out-dir', default='./eval_independent')
    ap.add_argument('--batch-size', type=int, default=1)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sr = config['data']['sample_rate']
    print(f"Device: {device}, eval sample_rate={sr}")

    ckpts = []
    for spec in args.ckpt:
        if '=' not in spec:
            raise SystemExit(f"--ckpt must be LABEL=PATH, got: {spec}")
        label, path = spec.split('=', 1)
        ckpts.append((label, path))

    primary = os.path.join(args.dnsmos_dir, 'sig_bak_ovr.onnx')
    p808 = os.path.join(args.dnsmos_dir, 'model_v8.onnx')
    for p in (primary, p808):
        if not os.path.exists(p):
            raise SystemExit(
                f"DNSMOS model missing: {p}\n"
                f"Run: (cd ../urgent2025_challenge && bash utils/download_dnsmos_onnx.sh)")

    dataset = build_test_dataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=config['training'].get('num_workers', 2),
                        pin_memory=True)
    print(f"Validation utterances: {len(dataset)} (deterministic crops)")

    pretrained_id = config['generator']['pretrained']
    print("Loading frozen RE-USE...")
    frozen, reuse_cfg = load_reuse_model(pretrained_id, device)
    frozen.eval()
    print("Loading voice-quality metric (trained judge)...")
    vq_metric = VoiceQualityMetric(vox_profile_path='./vox-profile-release',
                                   model_id=config['vq_metric']['model_id'],
                                   device=str(device), sample_rate=sr)
    print("Loading DNSMOS (independent judge)...")
    dnsmos = DNSMOS(primary, p808, use_gpu=(device.type == 'cuda'))

    arms = {}
    for label, path in ckpts:
        print(f"Loading arm '{label}': {path}")
        ft, _ = load_ft_model(pretrained_id, path, device)
        arms[label] = ft

    conditions = ['noisy', 'frozen'] + [l for l, _ in ckpts]

    rec = {c: {m: {} for m in ('vq', 'ovrl', 'sig', 'bak', 'p808', 'pesq', 'sisdr')}
           for c in conditions}
    clean_dnsmos = {}

    @torch.no_grad()
    def run():
        for noisy, clean, uids in tqdm(loader, desc="independent-eval"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            waves = {'noisy': noisy,
                     'frozen': enhance_batch(frozen, reuse_cfg, noisy, device)}
            for label, ft in arms.items():
                waves[label] = enhance_batch(ft, reuse_cfg, noisy, device)
            for i, uid in enumerate(uids):
                clean_np = clean[i].cpu().numpy()
                clean16 = _to16k(clean_np, sr)
                clean_dnsmos[uid] = dnsmos(clean16)['OVRL']
                for c in conditions:
                    w = waves[c]
                    L = min(w.shape[-1], clean.shape[-1])
                    w_i = w[i, :L]

                    d = vq_metric.compute_distance(w_i.cpu(), clean[i, :L].cpu())
                    rec[c]['vq'][uid] = d if d is not None else np.nan

                    w16 = _to16k(w_i.cpu().numpy(), sr)
                    ds = dnsmos(w16)
                    rec[c]['ovrl'][uid] = ds['OVRL']; rec[c]['sig'][uid] = ds['SIG']
                    rec[c]['bak'][uid] = ds['BAK']; rec[c]['p808'][uid] = ds['P808']

                    rec[c]['pesq'][uid] = _pesq_wb(clean16, w16)
                    rec[c]['sisdr'][uid] = si_sdr(
                        w_i.cpu().float(), clean[i, :L].cpu().float()).item()
    run()

    def col(c, m):
        v = np.array([x for x in rec[c][m].values() if x == x])
        return v

    print("\n=== Per-condition means (n=%d) ===" % len(dataset))
    hdr = f"{'condition':<10} {'VQ↓':>8} {'DNSMOS_OVRL↑':>13} {'SIG↑':>7} {'BAK↑':>7} {'P808↑':>7} {'PESQ↑':>7} {'SI-SDR↑':>8}"
    print(hdr); print('-' * len(hdr))
    clean_ovrl = np.nanmean(list(clean_dnsmos.values()))
    print(f"{'clean':<10} {'0.0000':>8} {clean_ovrl:>13.3f} {'-':>7} {'-':>7} {'-':>7} {'4.500':>7} {'inf':>8}")
    for c in conditions:
        vq = np.nanmean(list(rec[c]['vq'].values()))
        print(f"{c:<10} {vq:>8.4f} {np.nanmean(list(rec[c]['ovrl'].values())):>13.3f} "
              f"{np.nanmean(list(rec[c]['sig'].values())):>7.3f} {np.nanmean(list(rec[c]['bak'].values())):>7.3f} "
              f"{np.nanmean(list(rec[c]['p808'].values())):>7.3f} {np.nanmean(list(rec[c]['pesq'].values())):>7.3f} "
              f"{np.nanmean(list(rec[c]['sisdr'].values())):>8.2f}")

    def paired(metric, name, higher_better=True):
        print(f"\n=== Paired Wilcoxon: {name} (higher=better={higher_better}) ===")
        for a in range(len(conditions)):
            for b in range(a + 1, len(conditions)):
                ca, cb = conditions[a], conditions[b]
                shared = sorted(set(rec[ca][metric]) & set(rec[cb][metric]))
                va = np.array([rec[ca][metric][u] for u in shared])
                vb = np.array([rec[cb][metric][u] for u in shared])
                mask = (va == va) & (vb == vb)
                va, vb = va[mask], vb[mask]
                try:
                    _, p = wilcoxon(va, vb)
                except ValueError:
                    p = float('nan')
                better = ca if (va.mean() > vb.mean()) == higher_better else cb
                sig = "SIGNIFICANT" if (p == p and p < 0.05) else "n.s."
                print(f"  {ca} ({va.mean():.3f}) vs {cb} ({vb.mean():.3f}) "
                      f"-> better={better}, p={p:.3g} ({sig})")

    paired('ovrl', 'DNSMOS OVRL', higher_better=True)
    paired('pesq', 'PESQ', higher_better=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'independent_per_utterance.csv')
    all_uids = sorted({u for c in conditions for u in rec[c]['ovrl']})
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['uid', 'clean_ovrl']
        for c in conditions:
            cols += [f'{c}_{m}' for m in ('vq', 'ovrl', 'sig', 'bak', 'p808', 'pesq', 'sisdr')]
        w.writerow(cols)
        for u in all_uids:
            row = [u, clean_dnsmos.get(u, '')]
            for c in conditions:
                row += [rec[c][m].get(u, '') for m in ('vq', 'ovrl', 'sig', 'bak', 'p808', 'pesq', 'sisdr')]
            w.writerow(row)
    print(f"\nWrote {csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()
