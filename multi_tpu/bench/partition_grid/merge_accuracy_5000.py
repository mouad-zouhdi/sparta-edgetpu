#!/usr/bin/env python3
"""Reporte la precision INT8 relevee sur 5000 images dans accuracy_vs_tpu.csv.

Seules les colonnes top1_pct, top5_pct et n_images changent : tout le reste du
fichier vient de la campagne de performance, que la precision n'affecte pas.
"""
import sys, pandas as pd
from pathlib import Path

W = Path("/home/a131/Desktop/Project/multi_tpu/models/wave_bench")
TARGET = W / "results/H_accuracy_vs_tpu/accuracy_vs_tpu.csv"
NEW = Path(sys.argv[1])          # accuracy_v2_5000.csv rapatrie

d = pd.read_csv(TARGET)
n = pd.read_csv(NEW)
n = n[n.n_images == 5000]
key = ["kind", "model", "pct"]
m = {(r.kind, r.model, int(r.pct)): (r.top1_pct, r.top5_pct) for r in n.itertuples()}

miss, done = [], 0
for i, r in d.iterrows():
    k = (r.corpus, r.model, int(r.pct))
    if r.n_images == 5000:
        continue
    if k in m:
        d.at[i, "top1_pct"], d.at[i, "top5_pct"] = m[k]
        d.at[i, "n_images"] = 5000
        done += 1
    else:
        miss.append(k)
print(f"{done} lignes mises a jour, {len(miss)} sans correspondance")
for k in miss:
    print("   manquant :", k)
if not miss:
    d.to_csv(TARGET, index=False)
    print("ecrit ->", TARGET)
else:
    print("RIEN ECRIT")
