import argparse
import os
import yaml
import torch
import torchaudio
from tqdm import tqdm

from models.udit.udit import UDiT
from models.t_predicter import TPredicter
from core.speaker_gate import SpeakerVerificationGate
from utils.transforms import stft_torch, istft_torch
from flow_matching.solver.ode_solver import ODESolver


def parse_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_flowtse_model(ckpt_path, model_config, device):
    model = UDiT(**model_config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {
        k.replace('model.', ''): v
        for k, v in ckpt['state_dict'].items()
        if k.startswith('model.')
    }
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run AD-FlowTSE target speaker extraction on arbitrary audio.'
    )
    parser.add_argument('--config', required=True,
                        help='Path to FlowTSE config YAML (e.g. config/config_FlowTSE_large.yaml)')
    parser.add_argument('--mixture', required=True,
                        help='Path to mixture audio file')
    parser.add_argument('--enrollment', required=True,
                        help='Path to enrollment audio of the target speaker (~3 seconds recommended)')
    parser.add_argument('--output', default='extracted.wav',
                        help='Output path for extracted speech (default: extracted.wav)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Fixed mixing ratio in [0,1]. If set, overrides --t_predicter_ckpt.')
    parser.add_argument('--t_predicter_ckpt', type=str, default=None,
                        help='Path to trained TPredicter checkpoint (.ckpt) for per-chunk alpha estimation.')
    parser.add_argument('--chunk_batch_size', type=int, default=16,
                        help='Number of 3s chunks processed at once. Lower this if you run out of GPU memory.')
    parser.add_argument('--solver_method', type=str, default=None,
                        help='Override solver method from config (e.g. euler, midpoint).')
    parser.add_argument('--test_step_size', type=float, default=None,
                        help='Override solver test step size from config.')
    parser.add_argument('--chunk_seconds', type=float, default=3.0,
                        help='Chunk size in seconds for inference and per-chunk alpha estimation.')
    parser.add_argument('--sv_gate_ckpt', type=str, default=None,
                        help='Path to Speaker Verification Gate checkpoint. '
                             'When set, chunks classified as target-absent are silenced.')
    parser.add_argument('--gate_threshold', type=float, default=0.5,
                        help='Gate threshold: chunks with P(present) below this are silenced (default: 0.5).')
    return parser.parse_args()


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0)


def pad_and_reshape(tensor, multiple):
    n, d, l = tensor.shape
    padding_length = (multiple - (l % multiple)) % multiple
    padded = torch.nn.functional.pad(tensor, (0, padding_length))
    chunks = torch.cat(
        torch.chunk(padded, padded.shape[-1] // multiple, dim=-1), dim=0
    )
    return chunks, l


def reshape_and_remove_padding(tensor, original_length):
    n_k, d, multiple = tensor.shape
    n = original_length // multiple + (1 if original_length % multiple != 0 else 0)
    combined = torch.cat(torch.chunk(tensor, n, dim=0), dim=-1)
    return combined[:, :, :original_length]


def load_t_predicter(ckpt_path, model_config, device):
    t_predicter = TPredicter(**model_config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {
        k.replace('model.', ''): v
        for k, v in ckpt['state_dict'].items()
        if k.startswith('model.')
    }
    t_predicter.load_state_dict(state_dict)
    return t_predicter.eval().to(device)


def load_speaker_gate(ckpt_path, device):
    gate = SpeakerVerificationGate(C=1024)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {
        k.replace('model.', ''): v
        for k, v in ckpt['state_dict'].items()
        if k.startswith('model.')
    }
    gate.load_state_dict(state_dict)
    return gate.eval().to(device)


def estimate_gate_per_chunk(gate, mixture_wav, enrollment_wav, chunk_samples, device, threshold=0.5):
    total = mixture_wav.shape[-1]
    presence = []
    probs = []
    with torch.no_grad():
        for start in range(0, total, chunk_samples):
            chunk = mixture_wav[start:start + chunk_samples]
            if chunk.shape[-1] < chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
            prob = gate.predict(
                chunk.unsqueeze(0).to(device),
                enrollment_wav.unsqueeze(0).to(device),
            ).item()
            probs.append(prob)
            presence.append(prob >= threshold)
    return presence, probs


def estimate_alpha_per_chunk(t_predicter, mixture_wav, enrollment_wav, chunk_samples, device):
    total = mixture_wav.shape[-1]
    alphas = []
    with torch.no_grad():
        for start in range(0, total, chunk_samples):
            chunk = mixture_wav[start:start + chunk_samples]
            if chunk.shape[-1] < chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
            alpha = t_predicter(
                chunk.unsqueeze(0).to(device),
                enrollment_wav.unsqueeze(0).to(device),
                aug=False,
            )
            alphas.append(alpha.item())
    return alphas


def main():
    args = parse_args()
    config = parse_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sr = config['dataset']['sample_rate']
    n_fft = config['dataset']['n_fft']
    hop_length = config['dataset']['hop_length']
    win_length = config['dataset']['win_length']

    ckpt_path = config['eval']['checkpoint']
    print(f"Loading FlowTSE model from {ckpt_path}")
    model = load_flowtse_model(ckpt_path, config['model'], device)

    print(f"Loading mixture: {args.mixture}")
    mixture_wav = load_audio(args.mixture, sr)
    print(f"  Duration: {mixture_wav.shape[-1] / sr:.1f}s")

    print(f"Loading enrollment: {args.enrollment}")
    enrollment_wav = load_audio(args.enrollment, sr)
    max_enroll = sr * 3
    if enrollment_wav.shape[-1] > max_enroll:
        enrollment_wav = enrollment_wav[:max_enroll]
    elif enrollment_wav.shape[-1] < max_enroll:
        enrollment_wav = torch.nn.functional.pad(
            enrollment_wav, (0, max_enroll - enrollment_wav.shape[-1])
        )
    print(f"  Enrollment: {enrollment_wav.shape[-1] / sr:.1f}s")

    print("Computing STFT...")
    mixture_spec = stft_torch(mixture_wav, n_fft, hop_length, win_length).unsqueeze(0)
    enrollment_spec = stft_torch(enrollment_wav, n_fft, hop_length, win_length).unsqueeze(0).to(device)

    frames_per_chunk = int(sr * args.chunk_seconds) // hop_length + 1
    mixture_chunks, orig_spec_len = pad_and_reshape(mixture_spec, frames_per_chunk)
    num_chunks = mixture_chunks.shape[0]
    print(f"Split into {num_chunks} chunks (~{args.chunk_seconds:.2f}s each)")

    per_chunk_alpha = False

    if args.alpha is not None:
        fixed_alpha = args.alpha
        print(f"Using fixed alpha = {fixed_alpha:.4f}")

    elif args.t_predicter_ckpt is not None:
        print(f"Loading TPredicter from {args.t_predicter_ckpt}")
        t_predicter = load_t_predicter(args.t_predicter_ckpt, config['t_predicter'], device)
        print("Estimating per-chunk alpha from audio...")
        chunk_samples = int(sr * args.chunk_seconds)
        chunk_alphas = estimate_alpha_per_chunk(
            t_predicter, mixture_wav, enrollment_wav, chunk_samples, device
        )
        per_chunk_alpha = True
        print(f"  Alpha range: [{min(chunk_alphas):.4f}, {max(chunk_alphas):.4f}], "
              f"mean={sum(chunk_alphas) / len(chunk_alphas):.4f}")
        del t_predicter
        if device == 'cuda':
            torch.cuda.empty_cache()

    else:
        fixed_alpha = 0.5
        print("No --alpha or --t_predicter_ckpt given. Defaulting to alpha = 0.5")

    chunk_presence = None
    if args.sv_gate_ckpt is not None:
        print(f"Loading Speaker Verification Gate from {args.sv_gate_ckpt}")
        gate = load_speaker_gate(args.sv_gate_ckpt, device)
        chunk_samples_gate = int(sr * args.chunk_seconds)
        chunk_presence, chunk_probs = estimate_gate_per_chunk(
            gate, mixture_wav, enrollment_wav, chunk_samples_gate, device,
            threshold=args.gate_threshold,
        )
        n_present = sum(chunk_presence)
        n_absent = len(chunk_presence) - n_present
        print(f"  Gate: {n_present} present, {n_absent} absent chunks "
              f"(threshold={args.gate_threshold:.2f})")
        del gate
        if device == 'cuda':
            torch.cuda.empty_cache()

    print("Extracting target speaker...")
    solver_method = args.solver_method or config['solver']['method']
    solver_step = args.test_step_size if args.test_step_size is not None else config['solver']['test_step_size']
    print(f"Solver settings: method={solver_method}, test_step_size={solver_step}")
    all_outputs = []

    with torch.no_grad():
        if per_chunk_alpha:
            for i in tqdm(range(num_chunks), desc="Chunks"):
                chunk = mixture_chunks[i:i + 1].to(device)

                if chunk_presence is not None and not chunk_presence[i]:
                    all_outputs.append(torch.zeros_like(chunk).cpu())
                    continue

                alpha_val = chunk_alphas[i]

                solver = ODESolver(velocity_model=model)
                alpha_grid = torch.tensor([alpha_val, 1.0], device=device)
                out = solver.sample(
                    time_grid=alpha_grid,
                    x_init=chunk.float(),
                    method=solver_method,
                    step_size=solver_step,
                    enrollment=enrollment_spec,
                )
                all_outputs.append(out.cpu())
        else:
            alpha_grid = torch.tensor([fixed_alpha, 1.0], device=device)
            for i in tqdm(range(0, num_chunks, args.chunk_batch_size), desc="Batches"):
                batch = mixture_chunks[i:i + args.chunk_batch_size].to(device)
                bs = batch.shape[0]

                if chunk_presence is not None:
                    gate_mask = chunk_presence[i:i + bs]
                    if not any(gate_mask):
                        all_outputs.append(torch.zeros_like(batch).cpu())
                        continue

                solver = ODESolver(velocity_model=model)
                out = solver.sample(
                    time_grid=alpha_grid,
                    x_init=batch.float(),
                    method=solver_method,
                    step_size=solver_step,
                    enrollment=enrollment_spec.repeat(bs, 1, 1),
                )
                all_outputs.append(out.cpu())

    source_hat_spec = torch.cat(all_outputs, dim=0)
    source_hat_spec = reshape_and_remove_padding(source_hat_spec, orig_spec_len)

    source_hat = istft_torch(
        source_hat_spec, n_fft, hop_length, win_length,
        length=mixture_wav.shape[-1],
    )

    max_val = source_hat.abs().max()
    if max_val > 1.0:
        source_hat = source_hat / max_val

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(args.output, source_hat, sr)
    print(f"Saved extracted audio to {args.output} ({source_hat.shape[-1] / sr:.1f}s)")


if __name__ == '__main__':
    main()
