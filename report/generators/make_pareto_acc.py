#!/usr/bin/env python3
"""
make_pareto_acc.py — fronts de Pareto precision contre performance.

Une rangee par architecture, deux panneaux par rangee : la precision contre la
latence a gauche, contre le debit cumule a droite. Chaque panneau porte les 22
repartitions de huit accelerateurs pour chaque taux d'elagage mesure, soit de
220 a 242 configurations, et met en evidence celles que rien ne domine.

Une configuration est retenue quand aucune autre n'est au moins aussi bonne sur
les deux criteres et strictement meilleure sur l'un. Les dominees restent
tracees en gris : sans elles, le nombre de configurations que le front ecarte
serait invisible. La couleur code le taux d'elagage, l'etiquette la repartition,
et le trait horizontal marque la precision du modele non elague.

Donnees : wave_bench/results_raw/partitions_fixedn_par_config.csv (campagne a
mille inferences) et results/H_accuracy_vs_tpu/accuracy_vs_tpu.csv.

Sortie : figs/panel_pareto_acc.tex + figs/data/paretoacc_*.csv
"""
from pathlib import Path
import csv
import math

W = Path("/home/a131/Desktop/Project/multi_tpu/models/wave_bench")
CFG = W / "results_raw/partitions_fixedn_par_config.csv"
ACC = W / "results/H_accuracy_vs_tpu/accuracy_vs_tpu.csv"
OUT = Path("figs"); DATA = OUT / "data"; DATA.mkdir(parents=True, exist_ok=True)

MODELS = [("resnet101", "ResNet-101"),
          ("inception_v4", "Inception-V4"),
          ("inception_resnet_v2", "Inception-ResNet-V2")]
PANEL = "width=6.6cm, height=4.0cm, scale only axis"
# rampe bleue ordinale, du taux le plus faible au plus eleve ; la teinte la plus
# claire reste lisible sur fond blanc
RAMP = ["86B6EF", "6DA7EC", "5598E7", "3987E5", "2A78D6",
        "256ABF", "1C5CAB", "184F95", "0D366B", "081F3F"]
COLORS = "\n".join(["\\definecolor{pdom}{HTML}{C9C8C3}"]
                   + [f"\\definecolor{{prate{i}}}{{HTML}}{{{c}}}"
                      for i, c in enumerate(RAMP)])


def compact(shape):
    """4+4 -> 4x2 : les parts egales sont regroupees."""
    p = shape.split("+"); out = []; i = 0
    while i < len(p):
        j = i
        while j < len(p) and p[j] == p[i]:
            j += 1
        out.append(p[i] if j - i == 1 else f"{p[i]}$\\times${j - i}")
        i = j
    return "+".join(out)


def pareto(pts, xsense):
    """pts : (label, x, y, rate). y toujours a maximiser (precision)."""
    ge = (lambda a, b: a <= b) if xsense == "min" else (lambda a, b: a >= b)
    gt = (lambda a, b: a < b) if xsense == "min" else (lambda a, b: a > b)
    keep = [p for i, p in enumerate(pts)
            if not any(ge(q[1], p[1]) and q[2] >= p[2] and (gt(q[1], p[1]) or q[2] > p[2])
                       for j, q in enumerate(pts) if j != i)]
    return sorted(keep, key=lambda t: t[1])


# Geometrie du panneau, en centimetres, pour convertir une taille de texte en
# coordonnees normalisees : sans cela impossible de savoir si deux etiquettes
# se recouvrent.
PANEL_W_CM, PANEL_H_CM = 6.6, 4.0
CHAR_W_CM, LINE_H_CM = 0.105, 0.21          # \tiny dans un document 11 pt
# huit directions, essayees dans cet ordre ; le rayon est en fraction de panneau
# seize directions et trois rayons : au coude du front les points sont si
# serres qu'un seul rayon ne suffit pas a degager toutes les etiquettes
DIRS = [(math.cos(a), math.sin(a)) for a in
        (k * math.pi / 8 for k in range(16))]
RADII = [0.10, 0.18, 0.26, 0.34, 0.42]


def visible_len(lab):
    """Longueur affichee : $\\times$ ne compte que pour un caractere."""
    return len(lab.replace("$\\times$", "x"))


def place_labels(front, xlim, ylim):
    """Position de chaque etiquette en coordonnees de donnees.

    Les etiquettes du front se recouvrent des que deux configurations sont
    proches, ce qui arrive systematiquement au coude. On les decale donc dans la
    premiere des huit directions qui evite a la fois les autres etiquettes et
    les points deja places, et on relie chacune a son point par un trait.
    """
    (x0, x1), (y0, y1) = xlim, ylim
    to_n = lambda x, y: ((x - x0) / (x1 - x0), (y - y0) / (y1 - y0))
    to_d = lambda u, v: (x0 + u * (x1 - x0), y0 + v * (y1 - y0))
    pts_n = [to_n(t[1], t[2]) for t in front]
    boxes, out = [], []
    for i, tt in enumerate(front):
        w = visible_len(tt[0]) * CHAR_W_CM / PANEL_W_CM
        h = LINE_H_CM / PANEL_H_CM
        px, py = pts_n[i]
        best, best_pen = None, None
        for r in RADII:
            for dx, dy in DIRS:
                cx = px + dx * r
                cy = py + dy * r * PANEL_W_CM / PANEL_H_CM
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                pen = r * 0.22                      # on prefere rester pres du point
                if box[0] < 0.01 or box[2] > 0.99 or box[1] < 0.01 or box[3] > 0.99:
                    pen += 50.0
                for b in boxes:
                    ox = min(box[2], b[2]) - max(box[0], b[0]) + 0.006
                    oy = min(box[3], b[3]) - max(box[1], b[1]) + 0.006
                    if ox > 0 and oy > 0:
                        pen += 400 * ox * oy
                for qx, qy in pts_n:
                    if (box[0] - 0.012 < qx < box[2] + 0.012
                            and box[1] - 0.012 < qy < box[3] + 0.012):
                        pen += 3.0
                if best_pen is None or pen < best_pen:
                    best, best_pen = (cx, cy, box), pen
            if best_pen is not None and best_pen < r * 0.22 + 1e-9:
                break
        boxes.append(best[2])
        out.append(to_d(best[0], best[1]))
    return out


acc, budget, base = {}, {}, {}
for r in csv.DictReader(open(ACC)):
    k = (r["corpus"], r["model"], int(float(r["pct"])))
    acc[k] = float(r["top1_pct"])
    budget[k] = r["budget"]
    if int(float(r["pct"])) == 0:
        base[r["model"]] = float(r["top1_pct"])

cfg = list(csv.DictReader(open(CFG)))


def besoin(b):
    if b in ("", None) or str(b).lower() == "nan":
        return "streame sur huit"
    n = int(float(b))
    return f"{n} accél."


def panel(model, label, xcol, xsense, tag, rates, colour, first_col):
    pts = []
    for r in cfg:
        if r["model"] != model:
            continue
        k = (r["corpus"], r["model"], int(float(r["pct"])))
        if k not in acc:
            continue
        pts.append((compact(r["shape"]), float(r[xcol]), acc[k], int(float(r["pct"]))))
    front = pareto(pts, xsense)
    keys = {(t[0], round(t[1], 4)) for t in front}
    dom = [t for t in pts if (t[0], round(t[1], 4)) not in keys]
    with open(DATA / f"paretoacc_{model}_{tag}_dom.csv", "w") as f:
        f.write("x,y\n")
        for t in dom:
            f.write(f"{t[1]:.4f},{t[2]:.3f}\n")
    # un fichier par taux present sur le front, pour colorer par taux
    byrate = {}
    for t in front:
        byrate.setdefault(t[3], []).append(t)
    with open(DATA / f"paretoacc_{model}_{tag}_line.csv", "w") as f:
        f.write("x,y\n")
        for t in front:
            f.write(f"{t[1]:.4f},{t[2]:.3f}\n")
    xs = [t[1] for t in pts]; ys = [t[2] for t in pts] + [base[model]]
    lo, hi = min(xs), max(xs)
    o = ["\\begin{tikzpicture}", f"\\begin{{axis}}[{PANEL},",
         f"  title={{{label}}}, title style={{font=\\scriptsize}},",
         "  scaled x ticks=false, x tick label style={/pgf/number format/1000 sep={}},",
         "  xmin=0, " + f"xmax={hi * 1.22:.4g},",
         f"  ymin={min(ys) - 1.4:.3g}, ymax={max(ys) + 1.4:.3g},",
         "  grid=both, grid style={dashed,gray!30},",
         "  tick label style={font=\\tiny}, label style={font=\\tiny},"]
    if first_col:
        o.append("  ylabel={Précision Top-1 INT8 (\\%)},")
    o.append("  xlabel={" + ("Latence de l'instance la plus lente (ms)"
                             if tag == "lat" else "Débit cumulé (im/s)") + "}]")
    # asymptote : precision du modele non elague
    o.append(f"\\addplot[dashed, black, thin, forget plot] coordinates "
             f"{{(0,{base[model]:.3f}) ({hi*1.22:.6g},{base[model]:.3f})}};")
    o.append("\\addplot[only marks, mark=*, mark size=0.6pt, color=pdom] "
             f"table[x=x, y=y, col sep=comma] {{figs/data/paretoacc_{model}_{tag}_dom.csv}};")
    o.append("\\addplot[color=black!45, line width=0.7pt, mark=none] "
             f"table[x=x, y=y, col sep=comma] {{figs/data/paretoacc_{model}_{tag}_line.csv}};")
    for rate, ts in sorted(byrate.items()):
        nm = f"paretoacc_{model}_{tag}_r{rate}.csv"
        with open(DATA / nm, "w") as f:
            f.write("x,y\n")
            for t in ts:
                f.write(f"{t[1]:.4f},{t[2]:.3f}\n")
        o.append(f"\\addplot[only marks, mark=*, mark size=1.7pt, color={colour[rate]}] "
                 f"table[x=x, y=y, col sep=comma] {{figs/data/{nm}}};")
    labs = place_labels(front, (0.0, hi * 1.22), (min(ys) - 1.4, max(ys) + 1.4))
    for (lab, px, py, _), (lx, ly) in zip(front, labs):
        o.append(f"\\draw[gray, opacity=0.5, line width=0.25pt] "
                 f"(axis cs:{px:.4f},{py:.3f}) -- (axis cs:{lx:.4f},{ly:.3f});")
        o.append(f"\\node[font=\\tiny, inner sep=0.5pt, fill=white, fill opacity=0.75, "
                 f"text opacity=1] at (axis cs:{lx:.4f},{ly:.3f}) {{{lab}}};")
    o += ["\\end{axis}", "\\end{tikzpicture}"]
    print(f"  {model:22} {tag}  {len(pts):3d} configurations, front de {len(front)}")
    return "\n".join(o)


rows = []
for model, label in MODELS:
    rates = sorted({int(float(r["pct"])) for r in cfg if r["model"] == model})
    colour = {v: f"prate{min(len(RAMP)-1, round(i*(len(RAMP)-1)/max(1,len(rates)-1)))}"
              for i, v in enumerate(rates)}
    legname = f"leg:par{model.replace('_','')}"
    entries = []
    for v in rates:
        k = next(kk for kk in acc if kk[1] == model and kk[2] == v)
        tag = "non élagué" if v == 0 else f"{v}~\\%"
        entries.append(f"\\addlegendimage{{only marks, mark=*, mark size=1.5pt, "
                       f"color={colour[v]}}}\\addlegendentry{{{tag}, {besoin(budget[k])}}}")
    entries.append("\\addlegendimage{dashed, black, thin}"
                   "\\addlegendentry{précision non élaguée}")
    leg = "\n".join([
        "\\begin{tikzpicture}",
        "\\begin{axis}[hide axis, scale only axis, width=1pt, height=1pt,",
        "  xmin=0, xmax=1, ymin=0, ymax=1,",
        f"  legend columns=4, legend to name={legname},",
        "  legend style={draw=gray!50, font=\\tiny, column sep=5pt}]",
        "\\addplot[draw=none, forget plot] coordinates {(0,0)};"] + entries
        + ["\\end{axis}", "\\end{tikzpicture}"])
    a = panel(model, label, "lat_worst_ms_mean", "min", "lat", rates, colour, True)
    b = panel(model, label, "fps_total", "max", "thr", rates, colour, False)
    rows.append(leg + "\n\n\\noindent\\makebox[\\linewidth][c]{%\n"
                "\\begin{tabular}{@{}cc@{}}\n  " + a + "\n  &\n  " + b
                + "\n\\end{tabular}}\n\\vspace{2pt}\n"
                f"\\noindent\\makebox[\\linewidth][c]{{\\ref{{{legname}}}}}")

(OUT / "panel_pareto_acc.tex").write_text("\n".join([
    "% genere par make_pareto_acc.py, ne pas editer a la main",
    COLORS, "",
    "\\setlength{\\tabcolsep}{0pt}",
    "\n\n\\vspace{3pt}\n\n".join(rows)]) + "\n")
print("figs/panel_pareto_acc.tex genere")
