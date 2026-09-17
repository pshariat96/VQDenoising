#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict


_max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int // 10)

DISTORTION_TOKENS = (
    "clipping",
    "codec",
    "packet_loss",
    "bandwidth_limitation",
    "wind_noise",
)


def distortion_set(aug):
    aug = (aug or "").strip()
    if aug == "" or aug == "none":
        return "clean"
    found = set()
    for comp in aug.split("/"):
        comp = comp.strip()
        for tok in DISTORTION_TOKENS:
            if comp.startswith(tok):
                found.add(tok)
                break
    return "+".join(sorted(found)) if found else "clean"


def snr_bucket(snr):
    try:
        v = float(snr)
    except (TypeError, ValueError):
        return "na"
    if v < 0:
        return "<0"
    if v < 5:
        return "0-5"
    if v < 10:
        return "5-10"
    if v < 15:
        return "10-15"
    return ">=15"


def reverb_flag(rir):
    rir = (rir or "").strip()
    return "dry" if rir == "" or rir == "none" else "reverb"


def stratum_key(row):
    return (
        distortion_set(row.get("augmentation", "")),
        reverb_flag(row.get("rir_uid", "")),
        snr_bucket(row.get("snr_dB", "")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to full meta.tsv")
    ap.add_argument("--dst", required=True, help="path to write the subset manifest")
    ap.add_argument("--n", type=int, default=10000, help="number of rows to sample")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducibility)")
    args = ap.parse_args()

    with open(args.src, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    if total == 0:
        raise SystemExit(f"No rows found in {args.src}")

    if args.n >= total:
        print(f"--n ({args.n}) >= population ({total}); copying all rows.")
        sampled = list(rows)
    else:
        strata = defaultdict(list)
        for r in rows:
            strata[stratum_key(r)].append(r)

        quotas, frac = {}, {}
        for k, items in strata.items():
            exact = args.n * len(items) / total
            quotas[k] = int(math.floor(exact))
            frac[k] = exact - quotas[k]
        deficit = args.n - sum(quotas.values())
        for k in sorted(strata, key=lambda k: frac[k], reverse=True)[:deficit]:
            quotas[k] += 1

        rng = random.Random(args.seed)
        sampled = []
        for k, items in strata.items():
            q = min(quotas[k], len(items))
            if q > 0:
                sampled.extend(rng.sample(items, q))
        rng.shuffle(sampled)

    with open(args.dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        w.writeheader()
        w.writerows(sampled)

    pop = Counter(stratum_key(r) for r in rows)
    smp = Counter(stratum_key(r) for r in sampled)
    print(f"\nWrote {len(sampled)} / {total} rows -> {args.dst}  (seed={args.seed})")
    print(f"strata: {len(pop)}\n")
    print(f"{'distortion | reverb | snr':<50}{'pop%':>8}{'smp%':>8}")
    print("-" * 66)
    for k in sorted(pop, key=lambda kk: pop[kk], reverse=True):
        key_str = " | ".join(k)
        kp = 100.0 * pop[k] / total
        ks = 100.0 * smp[k] / max(1, len(sampled))
        print(f"{key_str:<50}{kp:8.2f}{ks:8.2f}")


if __name__ == "__main__":
    main()
