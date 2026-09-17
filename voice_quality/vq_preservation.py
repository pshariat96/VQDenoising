import csv, os, numpy as np
from scipy.stats import wilcoxon

CSV = os.environ.get("VQ_ATTRIBUTES_CSV", "ood_vq_attributes.csv")
ALL = ['shrill', 'nasal', 'deep', 'silky', 'husky', 'raspy', 'guttural', 'vocal-fry', 'booming',
       'authoritative', 'loud', 'hushed', 'soft', 'crisp', 'slurred', 'lisp', 'stammering',
       'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated', 'hesitant']

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

print(f"n={n}\nPRESERVATION metric: is |C2-noisy| < |frozen-noisy| ?  (C2 closer to raw voice = GREEN)\n")
print(f"{'label':<14}{'noisy':>7}{'froz':>7}{'C2':>7}{'|d|froz':>9}{'|d|C2':>8}{'winner':>8}{'p':>10}{'sig':>5}")
green = wash = red = 0
red_labels = []
for l in ALL:
    nz = S("noisy", l); fz = S("frozen", l); c2 = S("C2", l)
    ef = np.abs(fz - nz); ec = np.abs(c2 - nz)
    if np.allclose(ec, ef):
        p = 1.0
    else:
        p = wilcoxon(ec, ef)[1]
    if ec.mean() < ef.mean() and p < 0.05:
        w = "C2"; green += 1
    elif ec.mean() > ef.mean() and p < 0.05:
        w = "frozen"; red += 1; red_labels.append(l)
    else:
        w = "tie"; wash += 1
    print(f"{l:<14}{nz.mean():>7.3f}{fz.mean():>7.3f}{c2.mean():>7.3f}{ef.mean():>9.3f}{ec.mean():>8.3f}{w:>8}{p:>10.1e}{stars(p):>5}")

print(f"\nGREEN (C2 better preserved): {green}   TIE: {wash}   RED (frozen better preserved): {red}")
print(f"RED labels (C2 further from raw voice than frozen): {red_labels}")
