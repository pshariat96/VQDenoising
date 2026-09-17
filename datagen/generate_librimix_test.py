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
        print(f"ERROR: command exited with code {result.returncode}")
        sys.exit(1)


def check_tool(name):
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' is not installed. Please install it and retry.")
        sys.exit(1)


def download_librispeech(cache_dir):
    tar_path = os.path.join(cache_dir, "test-clean.tar.gz")
    extracted_marker = os.path.join(cache_dir, "LibriSpeech", "test-clean")

    if os.path.isdir(extracted_marker):
        print("[LibriSpeech] Already extracted, skipping.")
        return os.path.join(cache_dir, "LibriSpeech")

    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    if not os.path.isfile(tar_path):
        print("[LibriSpeech] Downloading test-clean (~350 MB)...")
        run(f'curl -L -C - -o "{tar_path}" "{url}"')
    else:
        print("[LibriSpeech] tar.gz already cached, skipping download.")

    print("[LibriSpeech] Extracting...")
    run(f'tar -xzf "{tar_path}" -C "{cache_dir}"')

    return os.path.join(cache_dir, "LibriSpeech")


def download_wham_noise(cache_dir):
    zip_path = os.path.join(cache_dir, "wham_noise.zip")
    extracted_marker = os.path.join(cache_dir, "wham_noise", "tt")

    if os.path.isdir(extracted_marker):
        print("[WHAM] Already extracted, skipping.")
        return os.path.join(cache_dir, "wham_noise")

    url = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
    if not os.path.isfile(zip_path):
        print("[WHAM] Downloading wham_noise.zip (~18 GB) -- this will take a while...")
        run(f'curl -L -C - -o "{zip_path}" "{url}"')
    else:
        print("[WHAM] zip already cached, skipping download.")

    print("[WHAM] Extracting test split (tt/) and metadata...")
    run(f'unzip -o -q "{zip_path}" "wham_noise/tt/*" "wham_noise/metadata/*" -d "{cache_dir}"')

    return os.path.join(cache_dir, "wham_noise")


def clone_librimix_repo(cache_dir):
    repo_dir = os.path.join(cache_dir, "LibriMix")
    if os.path.isdir(os.path.join(repo_dir, "scripts")):
        print("[LibriMix repo] Already cloned, skipping.")
        return repo_dir

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    print("[LibriMix repo] Cloning...")
    run(f'git clone --depth 1 https://github.com/JorisCos/LibriMix.git "{repo_dir}"')

    print("[LibriMix repo] Installing requirements...")
    run(f'{sys.executable} -m pip install -q -r "{os.path.join(repo_dir, "requirements.txt")}"')

    return repo_dir


def generate_dataset(repo_dir, librispeech_dir, wham_dir, raw_output_dir):
    metadata_dir = os.path.join(repo_dir, "metadata", "Libri2Mix")

    test_csv = os.path.join(metadata_dir, "libri2mix_test-clean.csv")
    if not os.path.isfile(test_csv):
        print(f"ERROR: metadata CSV not found at {test_csv}")
        sys.exit(1)

    for f in glob.glob(os.path.join(metadata_dir, "libri2mix_dev-*.csv")):
        os.remove(f)
    for f in glob.glob(os.path.join(metadata_dir, "libri2mix_train-*.csv")):
        os.remove(f)

    script = os.path.join(repo_dir, "scripts", "create_librimix_from_metadata.py")
    print("[Generate] Running create_librimix_from_metadata.py ...")
    run(
        f'{sys.executable} "{script}" '
        f'--librispeech_dir "{librispeech_dir}" '
        f'--wham_dir "{wham_dir}" '
        f'--metadata_dir "{metadata_dir}" '
        f'--librimix_outdir "{raw_output_dir}" '
        f'--n_src 2 --freqs 16k --modes min --types mix_clean mix_both',
        cwd=repo_dir,
    )


def organize_output(raw_output_dir, final_output_dir, metadata_src):
    audio_src = os.path.join(raw_output_dir, "Libri2Mix", "wav16k", "min", "test")

    if not os.path.isdir(audio_src):
        print(f"ERROR: Expected generated audio at {audio_src} but it doesn't exist.")
        sys.exit(1)

    sep_clean_dst = os.path.join(final_output_dir, "sep_clean")
    metadata_dst = os.path.join(final_output_dir, "metadata")

    if os.path.exists(sep_clean_dst):
        shutil.rmtree(sep_clean_dst)
    if os.path.exists(metadata_dst):
        shutil.rmtree(metadata_dst)

    print(f"[Organize] Copying audio to {sep_clean_dst}")
    shutil.copytree(audio_src, sep_clean_dst)

    os.makedirs(metadata_dst, exist_ok=True)
    test_csv_src = os.path.join(metadata_src, "Libri2Mix", "libri2mix_test-clean.csv")
    if os.path.isfile(test_csv_src):
        shutil.copy(test_csv_src, os.path.join(metadata_dst, "test.csv"))
        print("[Organize] Metadata CSV copied.")
    else:
        gen_meta = os.path.join(raw_output_dir, "Libri2Mix", "metadata")
        if os.path.isdir(gen_meta):
            for f in os.listdir(gen_meta):
                shutil.copy(os.path.join(gen_meta, f), metadata_dst)
            print("[Organize] Generated metadata copied.")


def verify(final_output_dir):
    sep_clean = os.path.join(final_output_dir, "sep_clean")
    expected_folders = ["mix_clean", "mix_both", "s1", "s2", "noise"]
    all_good = True

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    for folder in expected_folders:
        path = os.path.join(sep_clean, folder)
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.endswith(".wav")])
            status = "OK" if count == 3000 else "PARTIAL"
            if count != 3000:
                all_good = False
            print(f"  {folder:15s}: {count:5d} files  [{status}]")
        else:
            print(f"  {folder:15s}: MISSING")
            all_good = False

    if all_good:
        print("\nSUCCESS: All 3000 samples generated for all folders.")
    else:
        print("\nWARNING: Some folders are missing or incomplete.")

    print(f"\nDataset location: {final_output_dir}")
    print("=" * 60)
    return all_good


def main():
    parser = argparse.ArgumentParser(description="Generate the Libri2Mix test dataset")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save the final dataset (e.g. /path/to/Libri2Mix_Test)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory for downloads and intermediate files (default: <output_dir>/_cache)",
    )
    parser.add_argument(
        "--keep_cache", action="store_true",
        help="Keep the cache directory after generation (useful for reruns)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else os.path.join(output_dir, "_cache")

    check_tool("curl")
    check_tool("tar")
    check_tool("unzip")
    check_tool("git")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Libri2Mix Test Dataset Generator")
    print(f"  Output:  {output_dir}")
    print(f"  Cache:   {cache_dir}")
    print("=" * 60)

    librispeech_dir = download_librispeech(cache_dir)
    wham_dir = download_wham_noise(cache_dir)
    repo_dir = clone_librimix_repo(cache_dir)

    raw_output_dir = os.path.join(cache_dir, "generated")
    os.makedirs(raw_output_dir, exist_ok=True)

    generate_dataset(repo_dir, librispeech_dir, wham_dir, raw_output_dir)
    organize_output(raw_output_dir, output_dir, os.path.join(repo_dir, "metadata"))
    success = verify(output_dir)

    if not args.keep_cache:
        print("\nCleaning up cache...")
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("Cache removed.")
    else:
        print(f"\nCache kept at: {cache_dir}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
