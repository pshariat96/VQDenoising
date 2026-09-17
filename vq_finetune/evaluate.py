import os
import sys
import csv
import argparse

import numpy as np
import yaml
import torch
import soundfile as sf
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import wilcoxon

from dataloader import URGENTMetaDataset
from vq_metric import VoiceQualityMetric
from train import load_reuse_model, enhance_batch, si_sdr


def build_test_dataset(config):
    d = config['data']
    meta = d.get('test_meta_tsv') or d['train_meta_tsv']
    return URGENTMetaDataset(
        meta_tsv=meta,
        sample_rate=d['sample_rate'],
        segment_length=d['segment_length'],
        min_segment_length=d['min_segment_length'],
        data_root=d.get('data_root') or None,
        max_files=d.get('max_files', None),
        deterministic_crop=True,
    )


def load_ft_model(pretrained_id, ckpt_path, device):
    model, cfg = load_reuse_model(pretrained_id, device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get('generator_state_dict', state)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate_checkpoint(frozen, ft, reuse_cfg, vq_metric, loader, device, n_examples):
    rows = []
    examples = []
    for noisy, clean, uids in tqdm(loader, desc="eval", leave=False):
        noisy = noisy.to(device)
        clean = clean.to(device)
        enh_base = enhance_batch(frozen, reuse_cfg, noisy, device)
        enh_ft = enhance_batch(ft, reuse_cfg, noisy, device)
        base_vq = vq_metric.compute_batch_distance(enh_base, clean)
        ft_vq = vq_metric.compute_batch_distance(enh_ft, clean)
        for i, uid in enumerate(uids):
            L = min(enh_base.shape[-1], enh_ft.shape[-1], clean.shape[-1])
            base_sisdr = si_sdr(enh_base[i, :L].cpu().float(), clean[i, :L].cpu().float()).item()
            ft_sisdr = si_sdr(enh_ft[i, :L].cpu().float(), clean[i, :L].cpu().float()).item()
            delta = base_vq[i].item() - ft_vq[i].item()
            rows.append(dict(uid=uid, base_vq=base_vq[i].item(), ft_vq=ft_vq[i].item(),
                             delta_vq=delta, base_sisdr=base_sisdr, ft_sisdr=ft_sisdr))
            if n_examples > 0:
                examples.append((delta, uid,
                                 noisy[i, :L].cpu().numpy(), clean[i, :L].cpu().numpy(),
                                 enh_base[i, :L].cpu().numpy(), enh_ft[i, :L].cpu().numpy()))
    return rows, examples


def summarize(label, rows):
    base = np.array([r['base_vq'] for r in rows])
    ft = np.array([r['ft_vq'] for r in rows])
    delta = base - ft
    pct_improved = float((delta > 0).mean() * 100)
    try:
        stat, p = wilcoxon(base, ft)
    except ValueError:
        stat, p = float('nan'), float('nan')
    print(f"\n=== {label} (n={len(rows)}) ===")
    print(f"  VQ distance   frozen={base.mean():.4f}  ft={ft.mean():.4f}  "
          f"Δ={delta.mean():+.4f}  ({pct_improved:.0f}% improved)")
    print(f"  SI-SDR        frozen={np.mean([r['base_sisdr'] for r in rows]):.2f} dB  "
          f"ft={np.mean([r['ft_sisdr'] for r in rows]):.2f} dB")
    sig = "SIGNIFICANT" if (p == p and p < 0.05) else "n.s."
    print(f"  Wilcoxon (ft vs frozen): p={p:.3g}  -> {sig}")
    return dict(label=label, n=len(rows), base_vq=base.mean(), ft_vq=ft.mean(),
                delta=delta.mean(), pct_improved=pct_improved, p=p)


def save_examples(label, examples, out_dir, sr, n_examples):
    if n_examples <= 0 or not examples:
        return
    examples.sort(key=lambda e: e[0])
    worst = examples[:n_examples]
    best = examples[-n_examples:][::-1]
    ex_dir = os.path.join(out_dir, label, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    for tag, group in (("best", best), ("worst", worst)):
        for rank, (delta, uid, noisy, clean, enh_base, enh_ft) in enumerate(group):
            safe = str(uid).replace('/', '_')
            stem = f"{tag}{rank+1}_d{delta:+.4f}_{safe}"
            sf.write(os.path.join(ex_dir, f"{stem}_noisy.wav"), noisy, sr)
            sf.write(os.path.join(ex_dir, f"{stem}_clean.wav"), clean, sr)
            sf.write(os.path.join(ex_dir, f"{stem}_frozen.wav"), enh_base, sr)
            sf.write(os.path.join(ex_dir, f"{stem}_finetuned.wav"), enh_ft, sr)
    print(f"  saved {n_examples} best + {n_examples} worst examples to {ex_dir}")


def save_csv(label, rows, out_dir):
    os.makedirs(os.path.join(out_dir, label), exist_ok=True)
    path = os.path.join(out_dir, label, "per_utterance.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--ckpt', action='append', required=True,
                    help='LABEL=PATH to a best_model.pt (repeatable).')
    ap.add_argument('--out-dir', default='./eval')
    ap.add_argument('--examples', type=int, default=5,
                    help='Save this many best + worst improved utterances per arm.')
    ap.add_argument('--batch-size', type=int, default=1)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpts = []
    for spec in args.ckpt:
        if '=' not in spec:
            raise SystemExit(f"--ckpt must be LABEL=PATH, got: {spec}")
        label, path = spec.split('=', 1)
        ckpts.append((label, path))

    sr = config['data']['sample_rate']
    dataset = build_test_dataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=config['training'].get('num_workers', 2),
                        pin_memory=True)
    print(f"Validation utterances: {len(dataset)} (deterministic crops)")

    pretrained_id = config['generator']['pretrained']
    print("Loading frozen RE-USE (baseline)...")
    frozen, reuse_cfg = load_reuse_model(pretrained_id, device)
    frozen.eval()

    print("Loading voice quality metric...")
    vq_metric = VoiceQualityMetric(
        vox_profile_path='./vox-profile-release',
        model_id=config['vq_metric']['model_id'],
        device=str(device), sample_rate=sr,
    )

    summaries = []
    ft_vq_by_label = {}
    for label, path in ckpts:
        print(f"\nLoading fine-tuned checkpoint '{label}': {path}")
        ft, _ = load_ft_model(pretrained_id, path, device)
        rows, examples = evaluate_checkpoint(
            frozen, ft, reuse_cfg, vq_metric, loader, device, args.examples)
        summaries.append(summarize(label, rows))
        save_csv(label, rows, args.out_dir)
        save_examples(label, examples, args.out_dir, sr, args.examples)
        ft_vq_by_label[label] = {r['uid']: r['ft_vq'] for r in rows}
        del ft
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if len(ckpts) >= 2:
        print("\n=== Cross-arm paired Wilcoxon (lower VQ = better) ===")
        labels = [l for l, _ in ckpts]
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                la, lb = labels[a], labels[b]
                shared = sorted(set(ft_vq_by_label[la]) & set(ft_vq_by_label[lb]))
                va = np.array([ft_vq_by_label[la][u] for u in shared])
                vb = np.array([ft_vq_by_label[lb][u] for u in shared])
                try:
                    _, p = wilcoxon(va, vb)
                except ValueError:
                    p = float('nan')
                winner = la if va.mean() < vb.mean() else lb
                sig = "SIGNIFICANT" if (p == p and p < 0.05) else "n.s."
                print(f"  {la} (VQ {va.mean():.4f}) vs {lb} (VQ {vb.mean():.4f}) "
                      f"-> lower={winner}, p={p:.3g} ({sig})")

    print("\nDone.")


if __name__ == '__main__':
    main()
