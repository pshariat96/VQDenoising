import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="noisy_mixture",
                   help="Directory with noisy wav files")
    p.add_argument("--output_dir", default="reuse_enhanced",
                   help="Directory for enhanced output")
    p.add_argument("--reuse_dir", default="./REUSE",
                   help="Path to downloaded RE-USE model")
    p.add_argument("--target_sr", type=int, default=16000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, args.reuse_dir)

    try:
        from enhance import load_model, enhance_audio
        print("Loaded RE-USE from enhance module")
    except ImportError:
        print("Could not import from RE-USE. Trying alternative loading...")
        print(f"Contents of {args.reuse_dir}:")
        for f in os.listdir(args.reuse_dir):
            print(f"  {f}")
        print("\nPlease check RE-USE installation and adjust import accordingly.")
        print("You may need to run: cd REUSE && sh inference.sh")
        print("Or manually adapt the RE-USE inference code.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.reuse_dir, device)

    wav_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".wav")])
    print(f"Processing {len(wav_files)} files...")

    for fname in tqdm(wav_files):
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)

        audio, sr = sf.read(input_path)
        if sr != args.target_sr:
            import resampy
            audio = resampy.resample(audio, sr, args.target_sr)
            sr = args.target_sr

        enhanced = enhance_audio(model, audio, sr, device)

        sf.write(output_path, enhanced, sr)

    print(f"\nDone! Enhanced files saved to {args.output_dir}/")
    print(f"Download this folder and place at tse_vs_reuse_results/audio/reuse_enhanced/")


if __name__ == "__main__":
    main()
