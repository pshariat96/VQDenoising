import csv, os, numpy as np
from scipy.stats import wilcoxon

CSV = os.environ.get("VQ_ATTRIBUTES_CSV", "ood_vq_attributes.csv")
ALL = ['shrill', 'nasal', 'deep', 'silky', 'husky', 'raspy', 'guttural', 'vocal-fry', 'booming',
       'authoritative', 'loud', 'hushed', 'soft', 'crisp', 'slurred', 'lisp', 'stammering',
       'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated', 'hesitant']
DEP = ['singsong', 'flowing', 'soft', 'hesitant', 'stammering', 'slurred', 'husky', 'vocal-fry', 'raspy', 'deep']

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
print(f"n clips = {n}\n")

def series(cond, lab):
    return np.array([vq[cond][lab][u] for u in uids])

def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


print("=== 25 labels: C2 vs frozen (mean prob; paired Wilcoxon) ===")
print(f"{'label':<14}{'noisy':>8}{'frozen':>8}{'C2':>8}{'C2-froz':>9}{'p':>12}{'sig':>6}")
rows25 = []
for l in ALL:
    nz, fz, c2 = series("noisy", l).mean(), series("frozen", l), series("C2", l)
    d = c2.mean() - fz.mean()
    if np.allclose(c2, fz):
        p = 1.0
    else:
        p = wilcoxon(c2, fz)[1]
    print(f"{l:<14}{nz:>8.3f}{fz.mean():>8.3f}{c2.mean():>8.3f}{d:>9.3f}{p:>12.2e}{stars(p):>6}")
    rows25.append((l, nz, fz.mean(), c2.mean(), d, p))


print("\n=== Depression labels: preservation |dev vs noisy|, C2 vs frozen (lower=better) ===")
print(f"{'label':<12}{'|froz-noisy|':>13}{'|C2-noisy|':>12}{'better':>8}{'p':>12}{'sig':>6}")
rows_dep = []
for l in DEP:
    nz = series("noisy", l)
    ef = np.abs(series("frozen", l) - nz)
    ec = np.abs(series("C2", l) - nz)
    p = wilcoxon(ec, ef)[1] if not np.allclose(ec, ef) else 1.0
    better = "C2" if ec.mean() < ef.mean() else "frozen"
    print(f"{l:<12}{ef.mean():>13.3f}{ec.mean():>12.3f}{better:>8}{p:>12.2e}{stars(p):>6}")
    rows_dep.append((l, ef.mean(), ec.mean(), better, p))


ef_all = np.mean([[abs(vq["frozen"][l][u] - vq["noisy"][l][u]) for l in DEP] for u in uids], axis=1)
ec_all = np.mean([[abs(vq["C2"][l][u] - vq["noisy"][l][u]) for l in DEP] for u in uids], axis=1)
ec1_all = np.mean([[abs(vq["C1"][l][u] - vq["noisy"][l][u]) for l in DEP] for u in uids], axis=1)
print(f"\nAggregate mean |delta vs noisy| on dep-labels: frozen={ef_all.mean():.4f} "
      f"C2={ec_all.mean():.4f} C1={ec1_all.mean():.4f}")
print(f"C2 vs frozen p={wilcoxon(ec_all, ef_all)[1]:.2e} | C1 vs frozen p={wilcoxon(ec1_all, ef_all)[1]:.2e}")
