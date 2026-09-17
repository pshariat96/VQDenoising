import csv, os, numpy as np
from scipy.stats import wilcoxon

CSV = os.environ.get("VQ_ATTRIBUTES_CSV", "ood_vq_attributes.csv")
ALL = ['shrill', 'nasal', 'deep', 'silky', 'husky', 'raspy', 'guttural', 'vocal-fry', 'booming',
       'authoritative', 'loud', 'hushed', 'soft', 'crisp', 'slurred', 'lisp', 'stammering',
       'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated', 'hesitant']


REVISED = {
    'soft': 'reduced intensity', 'deep': 'lower F0', 'hesitant': 'pauses/psychomotor',
    'slurred': 'articulatory incoord.', 'stammering': 'disfluency', 'vocal-fry': 'creak/source instab.',
    'husky': 'breathy/dysphonic', 'raspy': 'jitter/shimmer', 'pitchy': 'pitch variability',
    'flowing': 'fluency/rhythm',
}
REV = list(REVISED)

with open(CSV) as f:
    header = next(csv.reader(f))
conds = [c for c in ["noisy", "frozen", "C2", "C1"] if f"{c}_{ALL[0]}" in header]
vq = {c: {l: {} for l in ALL} for c in conds}
with open(CSV) as f:
    for row in csv.DictReader(f):
        u = row["uid"]
        for c in conds:
            for l in ALL:
                v = row.get(f"{c}_{l}", "")
                if v not in ("", None):
                    try: vq[c][l][u] = float(v)
                    except ValueError: pass
uids = sorted(set(vq["frozen"][ALL[0]]) & set(vq["C2"][ALL[0]]) & set(vq["noisy"][ALL[0]]))
n = len(uids)
S = lambda c, l: np.array([vq[c][l][u] for u in uids])
stars = lambda p: "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."

print(f"n={n}\n--- monotone activity check ---")
for c in conds:
    print(f"  {c:7s} monotone mean = {S(c,'monotone').mean():.4f}")

print("\n=== Literature-anchored subset: preservation |dev vs noisy|, C2 vs frozen ===")
print(f"{'label':<11}{'proxy':<22}{'|d|froz':>8}{'|d|C2':>8}{'p':>11}{'sig':>5}")
for l in REV:
    nz = S("noisy", l); ef = np.abs(S("frozen", l) - nz); ec = np.abs(S("C2", l) - nz)
    p = wilcoxon(ec, ef)[1] if not np.allclose(ec, ef) else 1.0
    print(f"{l:<11}{REVISED[l]:<22}{ef.mean():>8.3f}{ec.mean():>8.3f}{p:>11.1e}{stars(p):>5}")

ef = np.mean([[abs(vq["frozen"][l][u]-vq["noisy"][l][u]) for l in REV] for u in uids], axis=1)
ec = np.mean([[abs(vq["C2"][l][u]-vq["noisy"][l][u]) for l in REV] for u in uids], axis=1)
ec1 = np.mean([[abs(vq["C1"][l][u]-vq["noisy"][l][u]) for l in REV] for u in uids], axis=1)
print(f"\nAGGREGATE (revised {len(REV)} labels): frozen={ef.mean():.4f} C2={ec.mean():.4f} C1={ec1.mean():.4f}")
print(f"  C2 vs frozen p={wilcoxon(ec,ef)[1]:.1e}  ({(1-ec.mean()/ef.mean())*100:.0f}% closer)")
print(f"  C1 vs frozen p={wilcoxon(ec1,ef)[1]:.1e}")
