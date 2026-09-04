#!/usr/bin/env python3
"""
make_accloss.py — le prix de la compression.

figs/panel_precision.tex : precision Top-1 absolue en fonction du taux
   d'elagage atteint, un panneau par architecture, sur les trois que le balayage
   a le plus densement mesurees. Trait horizontal a la precision du modele non
   elague. C'est la lecture qui donne le prix de la compression sur ce jeu de
   donnees et avec ces hyperparametres.

Donnees : wave_bench/results/H_accuracy_vs_tpu/accuracy_vs_tpu.csv.
"""
from pathlib import Path
import csv
from collections import defaultdict

ACC = Path("/home/a131/Desktop/Project/multi_tpu/models/wave_bench"
           "/results/H_accuracy_vs_tpu/accuracy_vs_tpu.csv")
OUT = Path("figs"); DATA = OUT / "data"; DATA.mkdir(parents=True, exist_ok=True)

NAME = {"inception_v1_googlenet": "GoogLeNet", "inception_v2_bninception": "BN-Inception",
        "inception_v3": "Inception-V3", "inception_v4": "Inception-V4",
        "inception_resnet_v2": "Inception-ResNet-V2", "resnet50": "ResNet-50",
        "resnet101": "ResNet-101", "resnet152": "ResNet-152"}
DENSE = ["resnet101", "inception_v4", "inception_resnet_v2"]
OTHER = [("resnet50", "modr50", "pentagon*"), ("resnet152", "modr152", "triangle*"),
         ("inception_v2_bninception", "modbni", "square*"),
         ("inception_v3", "modiv3", "diamond*")]
COLORS = "\n".join([
    "\\definecolor{modbni}{HTML}{17BECF}", "\\definecolor{modiv3}{HTML}{BCBD22}",
    "\\definecolor{modr50}{HTML}{1F77B4}", "\\definecolor{modr152}{HTML}{9467BD}",
    "\\definecolor{accblue}{HTML}{2A78D6}"])

rows = defaultdict(list); base = {}
for r in csv.DictReader(open(ACC)):
    m, pct = r["model"], int(float(r["pct"]))
    b = r["budget"]
    b = None if b in ("", None) or b.lower() == "nan" else int(float(b))
    if pct == 0:
        base[m] = float(r["top1_pct"])
    rows[m].append((pct, float(r["top1_pct"]), b))

# ── 1. precision absolue, trois architectures ────────────────────────────────
cells = []
for k, m in enumerate(DENSE):
    pts = sorted(p for p in rows[m] if p[0] > 0)
    with open(DATA / f"acccomp_{m}.csv", "w") as f:
        f.write("pct,top1\n")
        for pct, a, _ in pts:
            f.write(f"{pct},{a:.2f}\n")
    ys = [a for _, a, _ in pts] + [base[m]]
    # une ligne vide dans les options d'un axe pgfplots casse le parsing : on
    # filtre les entrees vides plutot que d'en laisser passer une
    cells.append("\n".join(x for x in [
        "\\begin{tikzpicture}",
        "\\begin{axis}[width=4.0cm, height=2.9cm, scale only axis,",
        f"  title={{{NAME[m]}}}, title style={{font=\\scriptsize}},",
        "  xlabel={Taux d'élagage (\\%)},",
        ("  ylabel={Précision Top-1 INT8 (\\%)}," if k == 0 else ""),
        "  xmin=0, xmax=100,",
        f"  ymin={min(ys)-1.2:.4g}, ymax={max(ys)+1.2:.4g},",
        "  grid=both, grid style={dashed,gray!30},",
        "  tick label style={font=\\tiny}, label style={font=\\tiny}]",
        f"\\addplot[dashed, black, thin] coordinates {{(0,{base[m]:.2f}) (100,{base[m]:.2f})}};",
        "\\addplot[color=accblue, mark=*, mark size=1.5pt, line width=0.9pt] "
        f"table[x=pct, y=top1, col sep=comma] {{figs/data/acccomp_{m}.csv}};",
        "\\end{axis}", "\\end{tikzpicture}"] if x))

ROW1 = "\n".join([
    "\\setlength{\\tabcolsep}{0pt}",
    "\\noindent\\makebox[\\linewidth][c]{%",
    "\\begin{tabular}{@{}ccc@{}}",
    "  " + "\n  &\n  ".join(cells),
    "\\end{tabular}}"])

(OUT / "panel_precision.tex").write_text("\n".join([
    "% genere par make_accloss.py, ne pas editer a la main",
    COLORS, "", ROW1]) + "\n")
print(f"figs/panel_precision.tex : {len(cells)} panneaux de précision absolue")
