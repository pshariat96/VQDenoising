#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import glob


def run(cmd, cwd=None):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: command exited with code {result.returncode}")
    return result.returncode


def download_librispeech(cache_dir, subset):
    url = f"https://www.openslr.org/resources/12/{subset}.tar.gz"
    tar_path = os.path.join(cache_dir, f"{subset}.tar.gz")
    extract_dir = os.path.join(cache_dir, "LibriSpeech")

    if os.path.exists(os.path.join(extract_dir, subset.replace("train-clean-", "train-clean-").replace("dev-clean", "dev-clean"))):
        print(f"{subset} already extracted, skipping download")
        return

    if not os.path.exists(tar_path):
        print(f"Downloading {subset}...")
        run(f"curl -L -o {tar_path} {url}")
    else:
        print(f"{tar_path} exists, skipping download")

    print(f"Extracting {subset}...")
    run(f"tar xzf {tar_path} -C {cache_dir}")


def download_wham_noise(cache_dir):
    wham_dir = os.path.join(cache_dir, "wham_noise")
    if os.path.exists(os.path.join(wham_dir, "tr")):
        print("WHAM noise already exists, skipping")
        return wham_dir

    zip_path = os.path.join(cache_dir, "wham_noise.zip")
    if not os.path.exists(zip_path):
        print("Downloading WHAM noise (~5GB)...")
        run(f"curl -L -o {zip_path} https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip")

    print("Extracting WHAM noise...")
    run(f"unzip -q -o {zip_path} -d {cache_dir}")
    return wham_dir


def clone_librimix(cache_dir):
    repo_dir = os.path.join(cache_dir, "LibriMix")
    if os.path.exists(repo_dir):
        print("LibriMix repo already cloned")
        return repo_dir
    run(f"git clone https://github.com/JorisCos/LibriMix.git {repo_dir}")
    return repo_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.expanduser("~/Desktop/LibriMix_Train"))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--keep_cache", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.output_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  LibriMix Training Data Generator")
    print(f"  Output: {args.output_dir}")
    print(f"  Cache:  {cache_dir}")
    print("=" * 60)

    download_librispeech(cache_dir, "train-clean-100")
    download_librispeech(cache_dir, "dev-clean")
    download_wham_noise(cache_dir)
    repo_dir = clone_librimix(cache_dir)

    try:
        import pysndfx
    except ImportError:
        run(f"{sys.executable} -m pip install pysndfx")

    if shutil.which("sox") is None:
        print("\nWARNING: 'sox' not found. Install it:")
        print("  macOS:  brew install sox")
        print("  Linux:  sudo apt-get install sox")
        sys.exit(1)

    wham_dir = os.path.join(cache_dir, "wham_noise")
    augment_script = os.path.join(repo_dir, "scripts", "augment_train_noise.py")
    if os.path.exists(augment_script):
        print("\nAugmenting WHAM noise...")
        run(f"{sys.executable} {augment_script} --wham_dir {wham_dir}")

    create_script = os.path.join(repo_dir, "scripts", "create_librimix_from_metadata.py")
    metadata_dir = os.path.join(repo_dir, "metadata")
    librispeech_dir = os.path.join(cache_dir, "LibriSpeech")

    for subset, csv_name in [
        ("dev", "libri2mix_dev-clean.csv"),
        ("train-100", "libri2mix_train-clean-100.csv"),
    ]:
        csv_path = os.path.join(metadata_dir, "Libri2Mix", csv_name)
        out_dir = os.path.join(args.output_dir, "Libri2Mix", "wav16k", "min", subset)

        if os.path.exists(out_dir) and len(glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)) > 100:
            print(f"\n{subset} already generated ({out_dir}), skipping")
            continue

        print(f"\nGenerating {subset} (16kHz min)...")
        run(
            f"{sys.executable} {create_script} "
            f"--metadata_dir {metadata_dir}/Libri2Mix "
            f"--librispeech_dir {librispeech_dir} "
            f"--wham_dir {wham_dir} "
            f"--metadata_file {csv_name} "
            f"--librimix_outdir {args.output_dir}/Libri2Mix "
            f"--n_src 2 "
            f"--freqs 16k "
            f"--modes min"
        )

    meta_dst = os.path.join(args.output_dir, "metadata")
    if not os.path.exists(meta_dst):
        shutil.copytree(metadata_dir, meta_dst)
    if not args.keep_cache:
        print(f"\nCleaning up cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("  DONE!")
    print(f"  Data at: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
