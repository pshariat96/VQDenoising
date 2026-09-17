import csv, os

CSV = os.environ.get("VQ_ATTRIBUTES_CSV", "ood_vq_attributes.csv")
ALL = ['shrill','nasal','deep','silky','husky','raspy','guttural','vocal-fry','booming',
       'authoritative','loud','hushed','soft','crisp','slurred','lisp','stammering',
       'singsong','pitchy','flowing','monotone','staccato','punctuated','enunciated','hesitant']

rows = list(csv.DictReader(open(CSV)))
def col(cond, lab):
    out = []
    for r in rows:
        v = r.get(f"{cond}_{lab}", "")
        if v not in ("", None):
            try: out.append((r["uid"], float(v)))
            except ValueError: pass
    return dict(out)

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

print(f"n={len(rows)} clips\n")
print("A label was RED in the old chart when C2_mean < frozen_mean (d_old < 0).")
print("The real question is: is frozen or C2 CLOSER to the raw 'noisy' value, per clip?\n")
hdr = (f"{'label':<13}{'raw':>7}{'frozen':>8}{'C2':>7} | {'d_old':>7}{'old':>6} | "
       f"{'|fz-raw|':>9}{'|C2-raw|':>9}{'closer':>7}")
print(hdr); print("-"*len(hdr))
flipped = []
for l in ALL:
    nz, fz, c2 = col("noisy",l), col("frozen",l), col("C2",l)
    u = set(nz) & set(fz) & set(c2)
    if not u: continue
    mfz, mc2, mraw = mean([fz[k] for k in u]), mean([c2[k] for k in u]), mean([nz[k] for k in u])
    d_old = mc2 - mfz
    oldcol = "RED" if d_old < 0 else "grn"
    dfz = mean([abs(fz[k]-nz[k]) for k in u])
    dc2 = mean([abs(c2[k]-nz[k]) for k in u])
    closer = "C2" if dc2 < dfz else ("frozen" if dc2 > dfz else "tie")
    mark = "  <== was RED, actually C2 closer" if (oldcol=="RED" and closer=="C2") else ""
    if mark: flipped.append(l)
    print(f"{l:<13}{mraw:>7.3f}{mfz:>8.3f}{mc2:>7.3f} | {d_old:>+7.3f}{oldcol:>6} | "
          f"{dfz:>9.3f}{dc2:>9.3f}{closer:>7}{mark}")

print(f"\n{len(flipped)} labels were RED in the old chart but are actually C2-closer-to-raw: {flipped}")
