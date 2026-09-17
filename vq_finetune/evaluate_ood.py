import os
import csv
import glob
import argparse

import numpy as np
import yaml
import torch
import librosa
from tqdm import tqdm
from scipy.stats import wilcoxon


from train import load_reuse_model, enhance_batch
from evaluate import load_ft_model
from vq_metric import VoiceQualityMetric


def make_dnsmos(device):
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as f
    dev = "cuda" if device.type == "cuda" else "cpu"
    def run(wav):
        out = f(wav, 16000, False, device=dev)
        out = out.detach().cpu().numpy().reshape(-1)
        return float(out[0]), float(out[1]), float(out[2]), float(out[3])
    return run


def make_nisqa():
    from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as f
    def run(wav):
        out = f(wav, 16000).detach().cpu().numpy().reshape(-1)
        return tuple(float(x) for x in out[:5])
    return run


def make_utmos(device):
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(device).eval()
    @torch.no_grad()
    def run(wav):
        w = wav.unsqueeze(0).to(device) if wav.dim() == 1 else wav.to(device)
        return float(predictor(w, 16000).item())
    return run


def make_spksim(device):
    from speechbrain.inference.speaker import EncoderClassifier
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})
    @torch.no_grad()
    def embed(wav):
        w = wav.unsqueeze(0).to(device) if wav.dim() == 1 else wav.to(device)
        return clf.encode_batch(w).squeeze().detach().cpu().numpy()
    return embed


def make_whisper(model_name, device):
    import re
    import whisper
    import jiwer
    model = whisper.load_model(model_name, device=device)

    def norm(t):
        t = re.sub(r"[^\w\s]", "", (t or "").lower())
        return re.sub(r"\s+", " ", t).strip()

    @torch.no_grad()
    def transcribe(wav):
        audio = wav.detach().cpu().numpy().astype(np.float32)
        return model.transcribe(audio, fp16=(device.type == "cuda"))["text"].strip()

    def _score(fn, ref, hyp):
        r, h = norm(ref), norm(hyp)
        if not r:
            return np.nan
        try:
            return float(fn(r, h))
        except Exception:
            return np.nan
    return transcribe, (lambda ref, hyp: _score(jiwer.cer, ref, hyp)),           (lambda ref, hyp: _score(jiwer.wer, ref, hyp))


def _cos(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_wavs(ood_dir):
    files = []
    for e in ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg"):
        files += glob.glob(os.path.join(ood_dir, "**", e), recursive=True)
    return sorted(set(files))


def load_clip(path, sr, clip_seconds, min_seconds=1.0):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    if clip_seconds and clip_seconds > 0:
        wav = wav[: int(clip_seconds * sr)]
    if len(wav) < int(min_seconds * sr):
        return None
    return torch.from_numpy(np.ascontiguousarray(wav)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ood-dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=PATH to a fine-tuned checkpoint (repeatable).")
    ap.add_argument("--out-dir", default="./eval_ood")
    ap.add_argument("--clip-seconds", type=float, default=15.0)
    ap.add_argument("--whisper-model", default="base",
                    help="openai-whisper size for CER/WER (tiny/base/small/medium).")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-vq", action="store_true", help="skip the project VQ judge.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["data"]["sample_rate"]
    print(f"Device: {device}, sample_rate={sr}, clip={args.clip_seconds}s")

    ckpts = []
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        ckpts.append((label, path))

    wavs = find_wavs(args.ood_dir)
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        raise SystemExit(f"No audio found under {args.ood_dir}")
    print(f"Out-of-domain clips: {len(wavs)}")

    pretrained_id = config["generator"]["pretrained"]
    print("Loading frozen RE-USE...")
    frozen, reuse_cfg = load_reuse_model(pretrained_id, device)
    frozen.eval()
    arms = {}
    for label, path in ckpts:
        print(f"Loading arm '{label}': {path}")
        arms[label], _ = load_ft_model(pretrained_id, path, device)

    def _try(name, fn):
        try:
            m = fn(); print(f"  [ok] {name}"); return m
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}"); return None
    print("Loading metric back-ends...")
    dnsmos = _try("DNSMOS (torchmetrics)", lambda: make_dnsmos(device))
    nisqa = _try("NISQA v2.0 (torchmetrics)", make_nisqa)
    utmos = _try("UTMOS22 (SpeechMOS)", lambda: make_utmos(device))
    spk_embed = _try("SpkSim (SpeechBrain ECAPA)", lambda: make_spksim(device))
    whisper_pack = _try(f"Whisper-{args.whisper_model} + jiwer",
                        lambda: make_whisper(args.whisper_model, device))
    vq = None
    if not args.no_vq:
        vq = _try("VQ judge (whisper-large-v3-voice-quality)",
                  lambda: VoiceQualityMetric(vox_profile_path="./vox-profile-release",
                                             model_id=config["vq_metric"]["model_id"],
                                             device=str(device), sample_rate=sr))

    conditions = ["noisy", "frozen"] + [l for l, _ in ckpts]
    metrics = ("dns_ovrl", "dns_sig", "dns_bak", "dns_p808",
               "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
               "utmos", "spksim", "cer", "wer",
               "vq_act", "drift_noisy", "drift_frozen")
    rec = {c: {m: {} for m in metrics} for c in conditions}

    @torch.no_grad()
    def run():
        for path in tqdm(wavs, desc="ood-eval"):
            clip = load_clip(path, sr, args.clip_seconds)
            if clip is None:
                continue
            uid = os.path.relpath(path, args.ood_dir)
            noisy = clip.unsqueeze(0).to(device)
            outs = {"noisy": clip, "frozen": enhance_batch(frozen, reuse_cfg, noisy, device)[0].cpu()}
            for label, ft in arms.items():
                outs[label] = enhance_batch(ft, reuse_cfg, noisy, device)[0].cpu()

            ref_txt = whisper_pack[0](outs["noisy"]) if whisper_pack else None
            noisy_emb = spk_embed(outs["noisy"]) if spk_embed else None
            probs = {c: (vq.get_probs(outs[c]) if vq else None) for c in conditions}
            for c in conditions:
                w = outs[c]
                if dnsmos:
                    p808, sig, bak, ovrl = dnsmos(w)
                    rec[c]["dns_p808"][uid]=p808; rec[c]["dns_sig"][uid]=sig
                    rec[c]["dns_bak"][uid]=bak;  rec[c]["dns_ovrl"][uid]=ovrl
                if nisqa:
                    mos, noi, dis, col, loud = nisqa(w)
                    rec[c]["nisqa_mos"][uid]=mos; rec[c]["nisqa_noi"][uid]=noi
                    rec[c]["nisqa_dis"][uid]=dis; rec[c]["nisqa_col"][uid]=col
                    rec[c]["nisqa_loud"][uid]=loud
                if utmos:
                    rec[c]["utmos"][uid] = utmos(w)
                if spk_embed:
                    rec[c]["spksim"][uid] = _cos(spk_embed(w), noisy_emb)
                if whisper_pack:
                    hyp = whisper_pack[0](w)
                    rec[c]["cer"][uid] = whisper_pack[1](ref_txt, hyp)
                    rec[c]["wer"][uid] = whisper_pack[2](ref_txt, hyp)
                if vq:
                    rec[c]["vq_act"][uid] = float(np.mean(probs[c])) if probs[c] is not None else np.nan
                    dn = np.abs(probs[c]-probs["noisy"]).sum()/25 if (probs[c] is not None and probs["noisy"] is not None) else np.nan
                    df = np.abs(probs[c]-probs["frozen"]).sum()/25 if (probs[c] is not None and probs["frozen"] is not None) else np.nan
                    rec[c]["drift_noisy"][uid] = dn
                    rec[c]["drift_frozen"][uid] = df
    run()

    def mean(c, m):
        v = [x for x in rec[c][m].values() if x == x]
        return np.mean(v) if v else np.nan
    n = max(len(rec["noisy"][m]) for m in metrics)
    print(f"\n=== Out-of-domain no-reference means (n={n} clips) ===")
    cols = [("DNSMOS_OVRL","dns_ovrl"), ("NISQA_MOS","nisqa_mos"), ("UTMOS","utmos"),
            ("SpkSim","spksim"), ("CER","cer"), ("WER","wer"), ("VQact","vq_act"),
            ("driftFrz","drift_frozen")]
    hdr = f"{'condition':<9}" + "".join(f"{h:>12}" for h,_ in cols)
    print(hdr); print("-"*len(hdr))
    for c in conditions:
        print(f"{c:<9}" + "".join(f"{mean(c,k):>12.4f}" for _,k in cols))

    def paired(metric, name, higher_better):
        print(f"\n=== Paired Wilcoxon: {name} ===")
        for i in range(len(conditions)):
            for j in range(i+1, len(conditions)):
                ca, cb = conditions[i], conditions[j]
                sh = sorted(set(rec[ca][metric]) & set(rec[cb][metric]))
                va = np.array([rec[ca][metric][u] for u in sh])
                vb = np.array([rec[cb][metric][u] for u in sh])
                mask = (va==va)&(vb==vb); va, vb = va[mask], vb[mask]
                if len(va) < 1 or np.allclose(va, vb):
                    continue
                try: _, p = wilcoxon(va, vb)
                except ValueError: p = float("nan")
                better = ca if (va.mean() > vb.mean()) == higher_better else cb
                sig = "SIGNIFICANT" if (p==p and p < 0.05) else "n.s."
                print(f"  {ca} ({va.mean():.3f}) vs {cb} ({vb.mean():.3f}) -> better={better}, p={p:.3g} ({sig})")

    if dnsmos: paired("dns_ovrl", "DNSMOS OVRL (higher=better)", True)
    if nisqa:  paired("nisqa_mos", "NISQA MOS (higher=better)", True)
    if utmos:  paired("utmos", "UTMOS (higher=better)", True)
    if spk_embed: paired("spksim", "Speaker similarity vs input (higher=better)", True)
    if whisper_pack: paired("cer", "CER vs input transcript (lower=better)", False)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "ood_per_clip.csv")
    all_uids = sorted({u for c in conditions for m in metrics for u in rec[c][m]})
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); head = ["uid"]
        for c in conditions:
            head += [f"{c}_{m}" for m in metrics]
        w.writerow(head)
        for u in all_uids:
            row = [u]
            for c in conditions:
                row += [rec[c][m].get(u, "") for m in metrics]
            w.writerow(row)
    print(f"\nWrote {csv_path}\nDone.")


if __name__ == "__main__":
    main()
