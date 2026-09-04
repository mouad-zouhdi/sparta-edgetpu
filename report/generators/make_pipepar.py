#!/usr/bin/env python3
"""
make_pipepar.py — panneau pgfplots « pipeline contre parallelisme ».

Quatre cas ordonnes par besoin de logement croissant, du modele qui tient sur un
seul accelerateur a celui qui ne tient nulle part. Deux metriques par cas, debit
cumule puis latence par echantillon, en fonction du nombre d'accelerateurs
mobilises. Echelles lineaires, legende sous la grille.

Donnees : final_plots/pipeline_vs_parallel/data.csv, campagne a nombre
d'inferences fixe. Les autres cas mesures partent en annexe, voir
make_pipepar_annexe.py.
"""
from pathlib import Path
import csv
from collections import defaultdict

SRC = Path("/home/a131/Desktop/Project/final_plots/pipeline_vs_parallel/data.csv")
OUT = Path("figs/panel_pipe_par.tex")
DATA = Path("figs/data"); DATA.mkdir(parents=True, exist_ok=True)

NAME = {"inception_v1_googlenet": "GoogLeNet", "inception_v2_bninception": "BN-Inception",
        "inception_v3": "Inception-V3", "inception_v4": "Inception-V4",
        "inception_resnet_v2": "Inception-ResNet-V2", "resnet50": "ResNet-50",
        "resnet101": "ResNet-101", "resnet152": "ResNet-152"}

# du modele qui loge sur un seul accelerateur a celui qui ne loge nulle part :
# le levier gagnant bascule du parallelisme vers le pipeline le long de la ligne
CASES = [("inception_v3", "wave", 84),
         ("inception_v4", "sweepN", 70),
         ("resnet101", "sweepN", 43),
         ("resnet152", "baseline", 0)]

PANEL = "width=3.5cm, height=2.9cm, scale only axis"
COLORS = """\\definecolor{regpipeline}{HTML}{2A78D6}
\\definecolor{regparallel}{HTML}{EB6834}"""


def load():
    rows = defaultdict(dict)
    for r in csv.DictReader(open(SRC)):
        k = (r["model"], r["corpus"], int(float(r["pruning_rate_pct"])))
        n = int(float(r["accelerators"]))
        rows[k].setdefault(n, {})[r["regime"]] = (
            float(r["throughput_fps"]), float(r["latency_ms"]))
        rows[k]["budget"] = r["budget_tpu"]
    return rows


def emit(key, d):
    """Un CSV par cas. A un accelerateur les deux regimes sont la meme
    configuration : le point est partage plutot que duplique."""
    name = f"pipepar_{key[0]}_{key[1]}_{key[2]}.csv"
    with open(DATA / name, "w") as f:
        f.write("n,par_fps,par_lat,pipe_fps,pipe_lat\n")
        for n in range(1, 9):
            e = d.get(n, {})
            par = e.get("parallel"); pipe = e.get("pipeline") or (par if n == 1 else None)
            g = lambda t, i: f"{t[i]:.4f}" if t else "nan"
            f.write(f"{n},{g(par,0)},{g(par,1)},{g(pipe,0)},{g(pipe,1)}\n")
    return name


def axis(csv_name, ycol_par, ycol_pipe, title, ylabel, xlabel):
    o = ["\\begin{tikzpicture}", f"\\begin{{axis}}[{PANEL},"]
    if title:
        o.append(f"  title={{\\shortstack{{{title}}}}}, "
                 "title style={font=\\tiny, align=center},")
    if ylabel:
        o.append(f"  ylabel={{{ylabel}}},")
    if xlabel:
        o.append(f"  xlabel={{{xlabel}}},")
    o += ["  ymin=0,",
          "  xmin=0.5, xmax=8.5, xtick={2,4,6,8},",
          "  scaled y ticks=false, y tick label style={/pgf/number format/1000 sep={}},",
          "  max space between ticks=26,",
          "  grid=both, grid style={dashed,gray!30},",
          "  tick label style={font=\\tiny}, label style={font=\\tiny}]"]
    for col, colour, mark in ((ycol_par, "regparallel", "*"),
                              (ycol_pipe, "regpipeline", "square*")):
        o.append(f"\\addplot[color={colour}, mark={mark}, mark size=1.2pt, "
                 f"line width=0.7pt, unbounded coords=discard] "
                 f"table[x=n, y={col}, col sep=comma] {{figs/data/{csv_name}}};")
    o += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(o)


def legend():
    return "\n".join([
        "\\begin{tikzpicture}",
        "\\begin{axis}[hide axis, scale only axis, width=1pt, height=1pt,",
        "  xmin=0, xmax=1, ymin=0, ymax=1,",
        "  legend columns=2, legend to name=leg:pipepar,",
        "  legend style={draw=gray!50, font=\\scriptsize, column sep=8pt}]",
        "\\addplot[draw=none, forget plot] coordinates {(0,0)};",
        "\\addlegendimage{color=regparallel, mark=*, mark size=1.4pt, line width=0.8pt}"
        "\\addlegendentry{Parallélisme : $N$ instances d'un étage}",
        "\\addlegendimage{color=regpipeline, mark=square*, mark size=1.4pt, line width=0.8pt}"
        "\\addlegendentry{Pipeline : une instance de $N$ étages}",
        "\\end{axis}", "\\end{tikzpicture}"])


def besoin(b):
    if b in ("", None) or b != b or str(b).lower() == "nan":
        return "ne loge pas sur huit"
    n = int(float(b))
    return f"loge sur {n} accélérateur{'s' if n > 1 else ''}"


data = load()
top, bot = [], []
for i, key in enumerate(CASES):
    d = data[key]
    name = emit(key, d)
    pct = key[2]
    head = (f"{NAME[key[0]]}, {pct}~\\%\\\\{besoin(d['budget'])}"
            if pct else f"{NAME[key[0]]} non élagué\\\\{besoin(d['budget'])}")
    top.append(axis(name, "par_fps", "pipe_fps", head,
                    "Débit cumulé (im/s)" if i == 0 else None, None))
    bot.append(axis(name, "par_lat", "pipe_lat", None,
                    "Latence (ms)" if i == 0 else None, "Accélérateurs"))

OUT.write_text("\n".join([
    "% genere par make_pipepar.py, ne pas editer a la main",
    COLORS, "", legend(), "",
    "\\setlength{\\tabcolsep}{0pt}",
    "\\noindent\\makebox[\\linewidth][c]{%",
    "\\begin{tabular}{@{}cccc@{}}",
    "  " + "\n  &\n  ".join(top) + " \\\\[3pt]",
    "  " + "\n  &\n  ".join(bot),
    "\\end{tabular}}",
    "\\vspace{4pt}",
    "\\noindent\\makebox[\\linewidth][c]{\\ref{leg:pipepar}}"]) + "\n")
print(f"{OUT} genere, {len(CASES)} cas")
