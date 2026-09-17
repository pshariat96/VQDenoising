import os
import sys
import argparse
import math

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from discriminator import MetricDiscriminator as VQDistanceDiscriminator
from semamba_discriminator import MetricDiscriminator as SEMambaMetricDiscriminator, batch_pesq
from semamba_loss import semamba_generator_loss_terms
from vq_metric import VoiceQualityMetric
from dataloader import create_dataloaders


ARMS = {
    "A0": dict(loss_family="stft",    use_pesq_metric=False, use_vq=False, direct_vq=None),
    "A1": dict(loss_family="stft",    use_pesq_metric=False, use_vq=True,  direct_vq=None),
    "B0": dict(loss_family="semamba", use_pesq_metric=True,  use_vq=False, direct_vq=None),
    "B1": dict(loss_family="semamba", use_pesq_metric=False, use_vq=True,  direct_vq=None),
    "B2": dict(loss_family="semamba", use_pesq_metric=True,  use_vq=True,  direct_vq=None),

    "C1": dict(loss_family="semamba", use_pesq_metric=False, use_vq=False, direct_vq="out"),
    "C2": dict(loss_family="semamba", use_pesq_metric=False, use_vq=False, direct_vq="feat"),
}


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
        mag_loss = F.l1_loss(torch.log(x_mag + 1e-8), torch.log(y_mag + 1e-8))

        return sc_loss + mag_loss

    def forward(self, x, y):
        loss = 0.0
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss += self.stft_loss(x, y, fft, hop, win)
        return loss / len(self.fft_sizes)


def load_reuse_model(pretrained_id, device):
    from huggingface_hub import snapshot_download

    reuse_dir = snapshot_download(repo_id=pretrained_id, local_dir="./REUSE")

    for pkg in ("utils", "models"):
        pkg_dir = os.path.join(reuse_dir, pkg)
        init_py = os.path.join(pkg_dir, "__init__.py")
        if os.path.isdir(pkg_dir) and not os.path.exists(init_py):
            open(init_py, "a").close()
    for _name in list(sys.modules):
        if _name in ("utils", "models") or _name.startswith(("utils.", "models.")):
            del sys.modules[_name]
    if reuse_dir in sys.path:
        sys.path.remove(reuse_dir)
    sys.path.insert(0, reuse_dir)

    from models.generator_SEMamba_time_d4 import SEMamba
    from utils.util import load_config

    config_path = hf_hub_download(repo_id=pretrained_id, filename='config.json')
    cfg = load_config(config_path)

    model = SEMamba.from_pretrained(pretrained_id, cfg=cfg).to(device)
    model.train()

    return model, cfg


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def _sfi_stft_params(cfg, sr=16000):
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    n_fft_s = make_even(n_fft * sr // sampling_rate)
    hop_s = make_even(hop_size * sr // sampling_rate)
    win_s = make_even(win_size * sr // sampling_rate)
    return n_fft_s, hop_s, win_s


def enhance_batch(model, cfg, noisy_batch, device):
    sys.path.insert(0, './REUSE')
    from models.stfts import mag_phase_stft, mag_phase_istft
    from utils.util import pad_or_trim_to_match

    compress_factor = cfg['model_cfg']['compress_factor']
    n_fft_s, hop_s, win_s = _sfi_stft_params(cfg)

    relu = nn.ReLU()
    wav = noisy_batch.to(device)

    noisy_mag, noisy_pha, _ = mag_phase_stft(
        wav, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
        compress_factor=compress_factor, center=True, addeps=False
    )
    amp_g, pha_g, _ = model(noisy_mag, noisy_pha)
    mag = torch.expm1(relu(amp_g))
    zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
    amp_g[:, :, (zero_portion > 0.5)[0]] = 0
    audio_g = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, compress_factor)
    audio_g = pad_or_trim_to_match(wav.detach(), audio_g, pad_value=1e-8)

    return audio_g


def enhance_stft_domain(model, cfg, noisy_batch, clean_batch, device):
    sys.path.insert(0, './REUSE')
    from models.stfts import mag_phase_stft, mag_phase_istft
    from utils.util import pad_or_trim_to_match

    compress_factor = cfg['model_cfg']['compress_factor']
    n_fft_s, hop_s, win_s = _sfi_stft_params(cfg)

    noisy = noisy_batch.to(device)
    clean = clean_batch.to(device)

    noisy_mag, noisy_pha, _ = mag_phase_stft(
        noisy, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
        compress_factor=compress_factor, center=True, addeps=False
    )
    mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
    audio_g = mag_phase_istft(mag_g, pha_g, n_fft_s, hop_s, win_s, compress_factor)
    audio_g = pad_or_trim_to_match(noisy.detach(), audio_g, pad_value=1e-8)

    clean_mag, clean_pha, clean_com = mag_phase_stft(
        clean, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
        compress_factor=compress_factor, center=True, addeps=False
    )

    _, _, rec_com = mag_phase_stft(
        audio_g, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s,
        compress_factor=compress_factor, center=True, addeps=True
    )

    tf = min(mag_g.shape[-1], clean_mag.shape[-1], com_g.shape[-2], rec_com.shape[-2])
    mag_g, pha_g, com_g = mag_g[..., :tf], pha_g[..., :tf], com_g[..., :tf, :]
    clean_mag, clean_pha = clean_mag[..., :tf], clean_pha[..., :tf]
    clean_com = clean_com[..., :tf, :]
    rec_com = rec_com[..., :tf, :]
    L = min(audio_g.shape[-1], clean.shape[-1])

    return {
        "mag_g": mag_g, "pha_g": pha_g, "com_g": com_g,
        "audio_g": audio_g[..., :L],
        "clean_mag": clean_mag, "clean_pha": clean_pha, "clean_com": clean_com,
        "clean_audio": clean[..., :L],
        "rec_com": rec_com,
        "n_fft_s": n_fft_s,
    }


def si_sdr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return 10.0 * torch.log10((target.pow(2).sum() + eps) / (noise.pow(2).sum() + eps))


@torch.no_grad()
def run_validation(generator, reuse_cfg, vq_metric, test_loader, device):
    generator.eval()
    vq_list, sisdr_list = [], []
    for noisy, clean, _ in tqdm(test_loader, desc="Validation", leave=False):
        noisy = noisy.to(device)
        clean = clean.to(device)
        enhanced = enhance_batch(generator, reuse_cfg, noisy, device)
        dists = vq_metric.compute_batch_distance(enhanced, clean)
        vq_list.extend(dists.tolist())
        for i in range(enhanced.shape[0]):
            L = min(enhanced.shape[-1], clean.shape[-1])
            sisdr_list.append(
                si_sdr(enhanced[i, :L].detach().cpu().float(),
                       clean[i, :L].detach().cpu().float()).item()
            )
    generator.train()
    mean_vq = sum(vq_list) / max(1, len(vq_list))
    mean_sisdr = sum(sisdr_list) / max(1, len(sisdr_list))
    return mean_vq, mean_sisdr


def train(config_path, arm_override=None, resume_path=None,
          lambda_vq_override=None, tag=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    import random as _random
    seed = int(config.get('seed', 42))
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tcfg = config['training']
    arm = arm_override or config.get('experiment', {}).get('arm', 'A1')
    if arm not in ARMS:
        raise ValueError(f"Unknown arm '{arm}'. Choose from {list(ARMS)}")
    spec = ARMS[arm]
    loss_family = spec['loss_family']
    use_pesq_metric = spec['use_pesq_metric']
    use_vq = spec['use_vq']
    direct_vq = spec.get('direct_vq')
    print(f"=== ARM {arm}: loss_family={loss_family}, "
          f"PESQ-Metric={use_pesq_metric}, VQ-Metric={use_vq}, "
          f"direct_vq={direct_vq} ===")

    run_name = arm + (f"_{tag}" if tag else "")
    ckpt_dir = os.path.join(tcfg['checkpoint_dir'], run_name)
    log_dir = os.path.join(tcfg['log_dir'], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print("Loading RE-USE model...")
    generator, reuse_cfg = load_reuse_model(config['generator']['pretrained'], device)

    print("Loading voice quality metric...")
    vq_metric = VoiceQualityMetric(
        vox_profile_path="./vox-profile-release",
        model_id=config['vq_metric']['model_id'],
        device=str(device),
    )

    disc_modules = {}
    if loss_family == "stft" and use_vq:

        disc_modules['vq_dist'] = VQDistanceDiscriminator(
            n_fft=config['discriminator']['n_fft'],
            hop_length=config['discriminator']['hop_length'],
        ).to(device)
    if loss_family == "semamba":

        if use_pesq_metric:
            disc_modules['pesq'] = SEMambaMetricDiscriminator().to(device)
        if use_vq:
            disc_modules['vq_qual'] = SEMambaMetricDiscriminator().to(device)
    for d in disc_modules.values():
        d.train()

    print("Loading data...")
    train_loader, test_loader = create_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    stft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=config['stft_loss']['fft_sizes'],
        hop_sizes=config['stft_loss']['hop_sizes'],
        win_sizes=config['stft_loss']['win_sizes'],
    )
    l1_loss_fn = nn.L1Loss()

    sw = config.get('semamba_loss', {})
    w_mag = sw.get('magnitude', 0.9)
    w_pha = sw.get('phase', 0.3)
    w_com = sw.get('complex', 0.1)
    w_time = sw.get('time', 0.2)
    w_con = sw.get('consistency', 0.1)
    w_pesq_metric = sw.get('metric', 0.05)
    w_vq_metric = (lambda_vq_override if lambda_vq_override is not None
                   else config['vq_metric'].get('lambda_vq_metric', 0.05))

    vq_range_min = float(config['vq_metric'].get('range_min', 0.0))
    vq_range_max = float(config['vq_metric'].get('range_max', 1.0))
    vq_range_span = max(vq_range_max - vq_range_min, 1e-6)
    if use_vq:
        print(f"[VQ] lambda_vq_metric={w_vq_metric}, "
              f"range=[{vq_range_min}, {vq_range_max}] (min-max norm), run={run_name}")

    vq_eot = bool(config['vq_metric'].get('eot', True))
    vq_layers = config['vq_metric'].get('perceptual_layers', None)
    if direct_vq is not None:

        try:
            _probe = next(iter(test_loader))[1][0]
            _md, _ok = vq_metric.validate_diff_mel(_probe)
            print(f"[direct-VQ] mode={direct_vq}, lambda={w_vq_metric}, eot={vq_eot}, "
                  f"layers={vq_layers}; diff-mel max|Δ| vs feature_extractor={_md:.2e} "
                  f"({'OK' if _ok else 'MISMATCH -- DO NOT TRUST'})")
            if not _ok:
                raise RuntimeError(
                    f"Differentiable log-mel does not match the Whisper feature "
                    f"extractor (max|Δ|={_md:.3e} >= 1e-3). Aborting to avoid a "
                    f"corrupted VQ signal. Fix _logmel_diff before training.")
        except StopIteration:
            raise RuntimeError("Empty test_loader; cannot validate diff log-mel.")
    num_workers_pesq = config['training'].get('num_workers', 8)

    opt_g = optim.AdamW(
        generator.parameters(),
        lr=config['generator']['learning_rate'],
        weight_decay=config['generator']['weight_decay'],
    )
    disc_params = [p for d in disc_modules.values() for p in d.parameters()]
    dcfg = config['discriminator']
    opt_d = optim.AdamW(
        disc_params,
        lr=dcfg['learning_rate'],
        betas=(dcfg.get('adam_b1', 0.8), dcfg.get('adam_b2', 0.99)),
        weight_decay=dcfg['weight_decay'],
    ) if disc_params else None

    lambda_spec = tcfg['lambda_spectral']
    lambda_gan_target = tcfg['lambda_metricgan']
    gan_warmup = tcfg.get('lambda_metricgan_warmup_steps', 0)
    warmup_steps = tcfg.get('warmup_steps', 0)
    grad_clip = tcfg['grad_clip']
    d_update_interval = max(1, tcfg.get('d_update_interval', 1))
    validate_every = tcfg.get('validate_every_n_steps', len(train_loader))
    patience = tcfg.get('early_stop_patience', 2)
    sisdr_margin = tcfg.get('sisdr_floor_margin_db', 0.5)

    total_steps = tcfg['epochs'] * len(train_loader)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler_g = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    scheduler_d = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda) if opt_d else None

    resume_state = None
    if resume_path:
        rp = os.path.join(ckpt_dir, 'resume_state.pt') if resume_path == 'auto' else resume_path
        if os.path.exists(rp):
            print(f"Resuming from {rp}")
            resume_state = torch.load(rp, map_location=device)
            if resume_state.get('arm') != arm:
                raise ValueError(
                    f"Resume checkpoint is for arm '{resume_state.get('arm')}', "
                    f"but you requested arm '{arm}'.")
            generator.load_state_dict(resume_state['generator_state_dict'])
            for name, d in disc_modules.items():
                key = f'disc_{name}_state_dict'
                if key in resume_state:
                    d.load_state_dict(resume_state[key])
            opt_g.load_state_dict(resume_state['opt_g_state_dict'])
            if opt_d is not None and resume_state.get('opt_d_state_dict') is not None:
                opt_d.load_state_dict(resume_state['opt_d_state_dict'])
            scheduler_g.load_state_dict(resume_state['scheduler_g_state_dict'])
            if scheduler_d is not None and resume_state.get('scheduler_d_state_dict') is not None:
                scheduler_d.load_state_dict(resume_state['scheduler_d_state_dict'])
        else:
            print(f"--resume given but no checkpoint found at {rp}; starting fresh.")

    def metric_ramp(step):

        if gan_warmup <= 0:
            return 1.0
        return min(1.0, step / gan_warmup)

    if resume_state is not None:
        base_vq = resume_state['baseline_vq_dist']
        base_sisdr = resume_state['baseline_si_sdr']
        sisdr_floor = base_sisdr - sisdr_margin
        print(f"  Restored baseline: VQ dist={base_vq:.4f}, SI-SDR={base_sisdr:.2f} dB "
              f"(VQ arms must keep val SI-SDR >= {sisdr_floor:.2f} dB)")
    else:
        print("Measuring baseline (frozen RE-USE) on validation...")
        base_vq, base_sisdr = run_validation(generator, reuse_cfg, vq_metric, test_loader, device)
        sisdr_floor = base_sisdr - sisdr_margin
        print(f"  Baseline: VQ dist={base_vq:.4f}, SI-SDR={base_sisdr:.2f} dB "
              f"(VQ arms must keep val SI-SDR >= {sisdr_floor:.2f} dB)")
        writer.add_scalar('Val/baseline_vq_distance', base_vq, 0)
        writer.add_scalar('Val/baseline_si_sdr', base_sisdr, 0)

    minimize = use_vq or (direct_vq is not None)
    best_metric = float('inf') if minimize else float('-inf')
    stale_validations = 0
    global_step = 0
    start_epoch = 0
    skip_in_epoch = 0
    stop = False
    if resume_state is not None:
        best_metric = resume_state.get('best_metric', best_metric)
        stale_validations = resume_state.get('stale_validations', 0)
        global_step = resume_state.get('global_step', 0)
        start_epoch = resume_state.get('epoch_in_progress', 0)
        skip_in_epoch = resume_state.get('steps_in_epoch', 0)
        print(f"  Resumed at epoch {start_epoch + 1}, global_step {global_step}, "
              f"skipping {skip_in_epoch} batches in this epoch.")

    def maybe_checkpoint_and_earlystop(epoch, val_vq, val_sisdr):
        nonlocal best_metric, stale_validations, stop
        if minimize:
            eligible = val_sisdr >= sisdr_floor
            improved = eligible and (val_vq < best_metric)
            cur = val_vq
            tag = "below SI-SDR floor -- not eligible" if not eligible else ""
        else:
            improved = val_sisdr > best_metric
            cur = val_sisdr
            tag = ""
        print(f"\n[step {global_step}] Val VQ={val_vq:.4f}, SI-SDR={val_sisdr:.2f} dB"
              f"{('  (' + tag + ')') if tag else ''}")
        if improved:
            best_metric = cur
            stale_validations = 0
            ckpt = {
                'arm': arm, 'epoch': epoch + 1, 'global_step': global_step,
                'generator_state_dict': generator.state_dict(),
                'val_vq_dist': val_vq, 'val_si_sdr': val_sisdr,
                'baseline_vq_dist': base_vq, 'baseline_si_sdr': base_sisdr,
            }
            for name, d in disc_modules.items():
                ckpt[f'disc_{name}_state_dict'] = d.state_dict()
            torch.save(ckpt, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  Saved best model ({'VQ' if minimize else 'SI-SDR'}="
                  f"{cur:.4f}, VQ={val_vq:.4f}, SI-SDR={val_sisdr:.2f} dB)")
        else:
            stale_validations += 1
            if stale_validations >= patience:
                print(f"  Early stopping: no improvement in {patience} validations.")
                stop = True

    one = torch.ones

    def save_resume_state(epoch_in_progress, steps_in_epoch):

        state = {
            'arm': arm,
            'epoch_in_progress': epoch_in_progress,
            'steps_in_epoch': steps_in_epoch,
            'global_step': global_step,
            'best_metric': best_metric,
            'stale_validations': stale_validations,
            'baseline_vq_dist': base_vq,
            'baseline_si_sdr': base_sisdr,
            'generator_state_dict': generator.state_dict(),
            'opt_g_state_dict': opt_g.state_dict(),
            'opt_d_state_dict': opt_d.state_dict() if opt_d is not None else None,
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict() if scheduler_d is not None else None,
        }
        for name, d in disc_modules.items():
            state[f'disc_{name}_state_dict'] = d.state_dict()
        tmp = os.path.join(ckpt_dir, 'resume_state.pt.tmp')
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(ckpt_dir, 'resume_state.pt'))

    for epoch in range(start_epoch, tcfg['epochs']):
        if stop:
            break
        generator.train()
        for d in disc_modules.values():
            d.train()

        steps_in_epoch = 0
        pbar = tqdm(train_loader, desc=f"[{arm}] Epoch {epoch+1}/{tcfg['epochs']}")
        for noisy, clean, fnames in pbar:

            if epoch == start_epoch and steps_in_epoch < skip_in_epoch:
                steps_in_epoch += 1
                continue
            noisy = noisy.to(device)
            clean = clean.to(device)
            logd = {}
            do_metric = (len(disc_modules) > 0) and (global_step % d_update_interval == 0)

            if loss_family == "stft":
                enhanced = enhance_batch(generator, reuse_cfg, noisy, device)
                loss_fid = l1_loss_fn(enhanced, clean) + stft_loss_fn(enhanced, clean)
                loss_g = lambda_spec * loss_fid

                if use_vq and do_metric:
                    with torch.no_grad():
                        vq_dist = vq_metric.compute_batch_distance(
                            enhanced.detach(), clean).to(device)
                    D = disc_modules['vq_dist']
                    opt_d.zero_grad()
                    pred = D(enhanced.detach(), clean).squeeze(-1)
                    loss_d = F.mse_loss(pred, vq_dist)
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
                    opt_d.step()

                    pred_g = D(enhanced, clean).squeeze(-1)
                    loss_metric = F.mse_loss(pred_g, torch.zeros_like(pred_g))
                    loss_g = loss_g + metric_ramp(global_step) * lambda_gan_target * loss_metric
                    logd['Loss/discriminator'] = loss_d.item()
                    logd['Loss/metricgan_vq'] = loss_metric.item()
                    logd['Metric/vq_distance'] = vq_dist.mean().item()

                logd['Loss/fidelity'] = loss_fid.item()

            else:
                out = enhance_stft_domain(generator, reuse_cfg, noisy, clean, device)
                terms = semamba_generator_loss_terms(
                    out['clean_mag'], out['clean_pha'], out['clean_com'], out['clean_audio'],
                    out['mag_g'], out['pha_g'], out['com_g'], out['audio_g'],
                    out['rec_com'], out['n_fft_s'],
                )
                loss_fid = (w_mag * terms['loss_mag'] + w_pha * terms['loss_pha']
                            + w_com * terms['loss_com'] + w_time * terms['loss_time']
                            + w_con * terms['loss_con'])
                loss_g = loss_fid

                if do_metric:
                    B = out['mag_g'].shape[0]
                    one_labels = one(B, device=device)
                    clean_mag = out['clean_mag']
                    mag_g = out['mag_g']
                    opt_d.zero_grad()
                    loss_d_total = 0.0

                    if use_pesq_metric:
                        Dp = disc_modules['pesq']
                        metric_r = Dp(clean_mag, clean_mag)
                        metric_g = Dp(clean_mag, mag_g.detach())
                        pesq_norm = batch_pesq(
                            list(out['clean_audio'].detach().cpu().numpy()),
                            list(out['audio_g'].detach().cpu().numpy()),
                            num_workers=num_workers_pesq,
                        )
                        loss_dp = F.mse_loss(one_labels, metric_r.flatten())
                        if pesq_norm is not None:
                            loss_dp = loss_dp + F.mse_loss(pesq_norm.to(device), metric_g.flatten())
                        loss_d_total = loss_d_total + loss_dp
                        logd['Loss/disc_pesq'] = loss_dp.item()

                    if use_vq:
                        Dv = disc_modules['vq_qual']
                        with torch.no_grad():
                            vq_dist = vq_metric.compute_batch_distance(
                                out['audio_g'].detach(), out['clean_audio']).to(device)

                        vq_quality = ((vq_range_max - vq_dist) / vq_range_span).clamp(0.0, 1.0)
                        metric_r = Dv(clean_mag, clean_mag)
                        metric_g = Dv(clean_mag, mag_g.detach())
                        loss_dv = (F.mse_loss(one_labels, metric_r.flatten())
                                   + F.mse_loss(vq_quality, metric_g.flatten()))
                        loss_d_total = loss_d_total + loss_dv
                        logd['Loss/disc_vq'] = loss_dv.item()
                        logd['Metric/vq_distance'] = vq_dist.mean().item()

                    loss_d_total.backward()
                    torch.nn.utils.clip_grad_norm_(disc_params, grad_clip)
                    opt_d.step()

                    ramp = metric_ramp(global_step)
                    if use_pesq_metric:
                        mg = disc_modules['pesq'](clean_mag, mag_g).flatten()
                        loss_metric_pesq = F.mse_loss(mg, one_labels)
                        loss_g = loss_g + ramp * w_pesq_metric * loss_metric_pesq
                        logd['Loss/metric_pesq'] = loss_metric_pesq.item()
                    if use_vq:
                        mg = disc_modules['vq_qual'](clean_mag, mag_g).flatten()
                        loss_metric_vq = F.mse_loss(mg, one_labels)
                        loss_g = loss_g + ramp * w_vq_metric * loss_metric_vq
                        logd['Loss/metric_vq'] = loss_metric_vq.item()

                if direct_vq is not None:
                    perc = vq_metric.perceptual_loss(
                        out['audio_g'], out['clean_audio'],
                        mode=direct_vq, layers=vq_layers, eot=vq_eot)
                    loss_g = loss_g + w_vq_metric * perc
                    logd['Loss/vq_perceptual'] = perc.item()

                logd['Loss/fidelity'] = loss_fid.item()
                logd['Loss/mag'] = terms['loss_mag'].item()
                logd['Loss/phase'] = terms['loss_pha'].item()
                logd['Loss/complex'] = terms['loss_com'].item()
                logd['Loss/time'] = terms['loss_time'].item()
                logd['Loss/consistency'] = terms['loss_con'].item()

            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_g.step()
            scheduler_g.step()
            if scheduler_d:
                scheduler_d.step()

            logd['Loss/generator_total'] = loss_g.item()
            logd['LR/generator'] = scheduler_g.get_last_lr()[0]
            for k, v in logd.items():
                writer.add_scalar(k, v, global_step)
            pbar.set_postfix({'L_g': f'{loss_g.item():.4f}'})

            global_step += 1
            steps_in_epoch += 1

            if global_step % validate_every == 0:
                val_vq, val_sisdr = run_validation(
                    generator, reuse_cfg, vq_metric, test_loader, device)
                writer.add_scalar('Val/vq_distance', val_vq, global_step)
                writer.add_scalar('Val/si_sdr', val_sisdr, global_step)
                maybe_checkpoint_and_earlystop(epoch, val_vq, val_sisdr)
                save_resume_state(epoch, steps_in_epoch)
                if stop:
                    break

        if (epoch + 1) % tcfg.get('save_every_epoch', 1) == 0:
            ckpt = {'arm': arm, 'epoch': epoch + 1, 'global_step': global_step,
                    'generator_state_dict': generator.state_dict()}
            for name, d in disc_modules.items():
                ckpt[f'disc_{name}_state_dict'] = d.state_dict()
            torch.save(ckpt, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt'))

        if not stop:
            save_resume_state(epoch + 1, 0)

    writer.close()
    print(f"\n[{arm}] Training complete. Best {'VQ dist' if minimize else 'SI-SDR'}="
          f"{best_metric:.4f} (baseline VQ {base_vq:.4f}, SI-SDR {base_sisdr:.2f} dB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--arm', type=str, default=None, choices=list(ARMS),
                        help='Override experiment.arm from the config.')
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume training: path to a resume_state.pt, or 'auto' "
                             "to use <checkpoint_dir>/<run_name>/resume_state.pt.")
    parser.add_argument('--lambda-vq', type=float, default=None, dest='lambda_vq',
                        help='Override vq_metric.lambda_vq_metric (VQ-weight sweep).')
    parser.add_argument('--tag', type=str, default=None,
                        help='Suffix for checkpoint/log dirs so sweep runs do not '
                             'collide (e.g. vqW0.2).')
    args = parser.parse_args()
    train(args.config, arm_override=args.arm, resume_path=args.resume,
          lambda_vq_override=args.lambda_vq, tag=args.tag)
