#!/usr/bin/env python3
"""
make_ft_curves.py — trajectoires de recuperation, pour les deux axes.

Produit des fragments pgfplots (et non des images), au meme format que les
autres figures du rapport : grille de panneaux, une courbe par serie, une
seule legende partagee, libelles en francais.

Axe 1 (CIFAR-100) : une figure par architecture, neuf panneaux (les neuf
  cibles d'elagage), sept courbes (les sept criteres d'importance). Les traits
  verticaux marquent les deux decroissances du taux d'apprentissage.

Axe 2 (ImageNet-1k) : deux figures.
  - le balayage N = 1..8, trois panneaux (les trois architectures), une courbe
    par taux d'elagage atteint. C'est la campagne la plus dense et celle dont
    le budget est cale sur le taux reel : elle va dans le corps du rapport.
  - la wave initiale, un panneau par architecture, une courbe par cible.

Le second axe horizontal en temps de calcul cumule, present sur la version
matplotlib d'origine, n'est pas repris : il demanderait un axe superposé par
panneau, pour une information que le texte donne deja globalement.

Sortie : figs/ft_*.tex + figs/data/ft_*.csv
"""
from __future__ import annotations
from pathlib import Path
import json, glob, re

ROOT = Path("/home/a131/Desktop/Project")
OUT = Path("figs")
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ── Axe 1 ────────────────────────────────────────────────────────────────
A1_LOGDIRS = [ROOT / "data/models/pruning_logs",
              Path("/home/a131/Desktop/second/pruning_logs")]
A1_MODELS = [
    ("resnet18",      "ResNet-18"),
    ("resnet50",      "ResNet-50"),
    ("vgg19",         "VGG-19"),
    ("wrn_28_10",     "WRN-28-10"),
    ("mobilenetv2",   "MobileNetV2"),
    ("googlenet",     "GoogLeNet"),
    ("squeezenet1_1", "SqueezeNet 1.1"),
]
# (cle de log, couleur pgf, libelle) — memes couleurs que les autres panneaux
CRITERIA = [
    ("magnitude_l1", "impmagnitudel1", "Magnitude L1"),
    ("magnitude_l2", "impmagnitudel2", "Magnitude L2"),
    ("bn_scale",     "impbnscale",     "BN-Scale"),
    ("fpgm",         "impfpgm",        "FPGM"),
    ("taylor",       "imptaylor",      "Taylor"),
    ("obdc",         "impobdc",        "OBD-C"),
    ("random",       "imprandom",      "Aléatoire"),
]
COLORDEFS = """\\definecolor{impmagnitudel1}{HTML}{1F77B4}
\\definecolor{impmagnitudel2}{HTML}{AEC7E8}
\\definecolor{impbnscale}{HTML}{FF7F0E}
\\definecolor{impfpgm}{HTML}{2CA02C}
\\definecolor{imptaylor}{HTML}{D62728}
\\definecolor{impobdc}{HTML}{9467BD}
\\definecolor{imprandom}{HTML}{7F7F7F}"""
A1_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# Le corps du rapport ne montre qu'une architecture a trois cibles : celle dont
# les criteres se separent le plus a 90 %, hors SqueezeNet dont l'effondrement
# rend la comparaison ininterpretable. Etendue des precisions finales a 90 % :
# MobileNetV2 17,9 pts, VGG-19 10,5, ResNet-18 10,3, WRN 10,0, GoogLeNet 6,6,
# ResNet-50 6,0. La version complete des neuf cibles reste disponible.
A1_CORPS_MODEL = "mobilenetv2"
A1_CORPS_PCTS = [30, 60, 90]
A1_LR_DROPS = [60, 80]          # MultiStep de la recette PruningBench

# ── Axe 2 ────────────────────────────────────────────────────────────────
A2_LOGDIR = ROOT / "multi_tpu/models/pruning_logs_imagenet"
A2_SWEEP_MODELS = [
    ("resnet101",           "ResNet-101"),
    ("inception_v4",        "Inception-V4"),
    ("inception_resnet_v2", "Inception-ResNet-V2"),
]
# Rampe ordinale : du plus clair (elagage leger) au plus fonce (severe).
# Bornee en clair a un bleu encore visible sur fond blanc : les deux premieres
# teintes d'une rampe Blues classique disparaissent a l'impression.
RAMP = ["a8d3ea", "7fbadd", "5aa0d0", "3d86c0",
        "2a6cae", "1a5296", "0d3b7a", "04245c"]

PANEL3 = "width=5.0cm, height=4.2cm"      # grille 3 colonnes
PANEL4 = "width=4.4cm, height=3.9cm"      # grille 4 colonnes


# ── lecture des logs ─────────────────────────────────────────────────────
A1_REF: dict[str, float] = {}


def a1_load():
    """{(model, pct): {criterion: [val_acc par epoque]}}"""
    crit_keys = sorted((c[0] for c in CRITERIA), key=len, reverse=True)
    out: dict[tuple[str, int], dict[str, list[float]]] = {}
    for d in A1_LOGDIRS:
        for p in glob.glob(str(d / "*.json")):
            m = re.match(r"(.+)_(\d+)pct\.json$", Path(p).name)
            if not m:
                continue
            stem, pct = m.group(1), int(m.group(2))
            crit = next((c for c in crit_keys if stem.endswith("_" + c)), None)
            if crit is None:
                continue
            model = stem[: -len(crit) - 1]
            j = json.loads(Path(p).read_text())
            hist = j.get("ft_history") or []
            if not hist:
                continue
            out.setdefault((model, pct), {})[crit] = [h["val_acc"] for h in hist]
            # La reference se deduit de chaque run : final_acc - acc_delta. Elle
            # varie legerement d'un run a l'autre ; le maximum retrouve
            # exactement la precision de validation du modele non elague.
            if "final_acc" in j and "acc_delta" in j:
                r = j["final_acc"] - j["acc_delta"]
                A1_REF[model] = max(A1_REF.get(model, r), r)
    return out


A2_REF: dict[str, float] = {}


def a2_load(pattern: str, key, label=None):
    """{model: [(cle_de_tri, libelle, [val_top1 par epoque]), ...]}.

    `key` sert au tri, `label` au texte de legende ; par defaut le libelle est
    la cle. Les deux sont dissocies parce que le balayage se trie par nombre
    d'accelerateurs mais se lit avec le taux d'elagage en regard.
    """
    out: dict[str, list] = {}
    for p in sorted(glob.glob(str(A2_LOGDIR / pattern))):
        j = json.loads(Path(p).read_text())
        hist = j.get("ft_history") or []
        if not hist:
            continue
        k = key(j, p)
        out.setdefault(j["model"], []).append(
            (k, label(j, p) if label else k, [h["val_top1_pct"] for h in hist]))
        A2_REF[j["model"]] = j.get("ref_top1_pct", A2_REF.get(j["model"]))
    for m in out:
        out[m].sort(key=lambda t: t[0])
    return out


# ── ecriture des CSV ─────────────────────────────────────────────────────
def write_csv(name: str, cols: list[str], series: list[list[float]]) -> int:
    n = max(len(s) for s in series)
    lines = ["epoch," + ",".join(cols)]
    for i in range(n):
        cells = [f"{s[i]:.4f}" if i < len(s) else "nan" for s in series]
        lines.append(f"{i+1}," + ",".join(cells))
    (DATA / f"{name}.csv").write_text("\n".join(lines) + "\n")
    return n


# ── fragments pgfplots ───────────────────────────────────────────────────
def ylims(all_series) -> tuple[int, int]:
    """Bornes communes a une figure, arrondies a la dizaine.

    Calculees explicitement plutot que laissees a l'autoscale, pour deux
    raisons : les traits verticaux marquant les paliers de taux d'apprentissage
    doivent aller d'un bord a l'autre sans etendre l'axe, et des bornes
    partagees rendent les panneaux comparables entre eux.
    """
    vals = [v for s in all_series for v in s]
    lo = int(min(vals) // 10) * 10
    hi = int(-(-max(vals) // 10)) * 10
    return lo, min(hi, 100)


def axis(csv: str, cols_colors_labels, title, ylabel, xlabel, panel,
         xmax, vlines=(), ylim=None, ref=None):
    o = ["\\begin{tikzpicture}", f"\\begin{{axis}}[{panel},"]
    if title:
        o.append(f"  title={{{title}}}, title style={{font=\\scriptsize}},")
    if ylabel:
        o.append(f"  ylabel={{{ylabel}}},")
    if xlabel:
        o.append(f"  xlabel={{{xlabel}}},")
    o.append(f"  xmin=0, xmax={xmax},")
    if ylim is not None:
        o.append(f"  ymin={ylim[0]}, ymax={ylim[1]},")
    o += ["  grid=both, grid style={dashed,gray!30},",
          "  tick label style={font=\\tiny}, label style={font=\\tiny}]"]
    if ref is not None:
        o.append(f"\\addplot[black, dashed, line width=0.7pt, forget plot, "
                 f"domain=0:{xmax}, samples=2] {{{ref:.3f}}};")
    for x in vlines:
        lo, hi = ylim if ylim else (0, 100)
        o.append(f"\\addplot[black, dotted, line width=0.5pt, forget plot] "
                 f"coordinates {{({x},{lo}) ({x},{hi})}};")
    for col, color, _ in cols_colors_labels:
        o.append(f"\\addplot[color={color}, mark=none, line width=0.5pt, "
                 f"unbounded coords=discard] table[x=epoch, y={col}, "
                 f"col sep=comma] {{figs/data/{csv}.csv}};")
    o += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(o)


def legend(name: str, entries, columns=4, compact=False):
    """`compact` resserre la legende sans toucher aux panneaux : c'est elle qui
    deborde de la ligne quand un modele a huit entrees."""
    style = ("font=\\fontsize{5}{6}\\selectfont, inner sep=1.5pt, "
             "/tikz/every even column/.append style={column sep=2pt}, "
             "legend image post style={scale=0.7}"
             if compact else "font=\\tiny")
    o = ["\\begin{tikzpicture}",
         "\\begin{axis}[hide axis, scale only axis, width=1pt, height=1pt,",
         "  xmin=0, xmax=1, ymin=0, ymax=1,",
         f"  legend columns={columns}, legend to name={name},",
         f"  legend style={{draw=gray!50, {style}}}]",
         "\\addplot[draw=none, forget plot] coordinates {(0,0)};"]
    for color, label in entries:
        o.append(f"\\addlegendimage{{color={color}, mark=none, line width=0.9pt}}"
                 f"\\addlegendentry{{{label}}}")
    o += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(o)


def stacked_legends(names, refs):
    """Une legende par ligne, precedee du nom du modele.

    Les taux d'elagage atteints different d'un modele a l'autre : une legende
    unique est impossible, et les mettre cote a cote deborde de la ligne.
    """
    rows = [f"  \\tiny {n} & {r} \\\\" for n, r in zip(names, refs)]
    return "\n".join(["\\noindent\\makebox[\\linewidth][c]{%",
                      "\\begin{tabular}{@{}r@{~~}l@{}}", *rows,
                      "\\end{tabular}}"])


def grid(cells, ncol):
    rows = []
    for i in range(0, len(cells), ncol):
        chunk = cells[i:i + ncol] + [""] * (ncol - len(cells[i:i + ncol]))
        rows.append("  " + "\n  &\n  ".join(chunk))
    return "\n".join([
        "\\setlength{\\tabcolsep}{0pt}",
        "\\noindent\\makebox[\\linewidth][c]{%",
        "\\begin{tabular}{@{}" + "c" * ncol + "@{}}",
        " \\\\[3pt]\n".join(rows),
        "\\end{tabular}}"])


# ══ AXE 1 ════════════════════════════════════════════════════════════════
a1 = a1_load()
made1 = []
for model, nice in A1_MODELS:
    # 1re passe : bornes communes a toute la figure
    lim = ylims([s for pct in A1_PCTS for s in a1.get((model, pct), {}).values()]
                + [[A1_REF[model]]])
    cells = []
    present = set()
    for k, pct in enumerate(A1_PCTS):
        series = a1.get((model, pct), {})
        if not series:
            continue
        cols = [(c, col, lab) for c, col, lab in CRITERIA if c in series]
        present.update(c for c, _, _ in cols)
        name = f"ft_a1_{model}_{pct}"
        n = write_csv(name, [c[0] for c in cols], [series[c[0]] for c in cols])
        cells.append(axis(
            name, cols, f"Cible {pct} \\%",
            "Top-1 de validation (\\%)" if k % 3 == 0 else None,
            "Époque" if k >= 6 else None,
            PANEL3, n, vlines=A1_LR_DROPS, ylim=lim, ref=A1_REF.get(model)))
    if not cells:
        continue
    legname = f"leg:fta1{model.replace('_','')}"
    entries = [(col, lab) for c, col, lab in CRITERIA if c in present]
    entries.append(("black", "Modèle de référence"))
    (OUT / f"ft_a1_{model}.tex").write_text("\n".join([
        "% genere par make_ft_curves.py, ne pas editer a la main",
        COLORDEFS, "", legend(legname, entries), "",
        grid(cells, 3), "\\vspace{3pt}",
        f"\\noindent\\makebox[\\linewidth]{{\\ref{{{legname}}}}}"]) + "\n")
    made1.append((model, nice, len(cells), len(entries)))

    # variante courte pour le corps du rapport : trois cibles au lieu de neuf,
    # la version complete restant en annexe
    if model == A1_CORPS_MODEL:
        short = []
        for k, pct in enumerate(A1_CORPS_PCTS):
            series = a1.get((model, pct), {})
            cols = [(c, col, lab) for c, col, lab in CRITERIA if c in series]
            nep = max(len(series[c]) for c, _, _ in cols)
            short.append(axis(
                f"ft_a1_{model}_{pct}", cols, f"Cible {pct} \\%",
                "Top-1 de validation (\\%)" if k == 0 else None, "Époque",
                "width=4.6cm, height=3.4cm, scale only axis", nep,
                vlines=A1_LR_DROPS, ylim=lim,
                ref=A1_REF.get(model)))
        (OUT / f"ft_a1_{model}_corps.tex").write_text("\n".join([
            "% genere par make_ft_curves.py, ne pas editer a la main",
            COLORDEFS, "", legend(legname + "c", entries), "",
            grid(short, 3), "\\vspace{3pt}",
            f"\\noindent\\makebox[\\linewidth]{{\\ref{{{legname}c}}}}"]) + "\n")

# ══ AXE 2 : balayage N = 1..8 (corps du rapport) ═════════════════════════
A2_SWEEP_MODELS_NICE = [(nice, m) for m, nice in A2_SWEEP_MODELS]
def _n_of(path):
    return int(re.search(r"_N(\d+)\.json$", str(path)).group(1))

# Le balayage produit un modele par nombre d'accelerateurs : c'est cette
# grandeur qui ordonne les courbes, le taux d'elagage n'en etant que la
# consequence (il est fixe par la contrainte de tenue en memoire interne).
def _hours(j):
    """Temps GPU de la recuperation, en heures."""
    return j["duration_s"]["finetune"] / 3600.0


# Le temps de calcul figure dans la legende plutot que sur un axe parallele :
# le temps par epoque varie d'un facteur 2 a 4 entre executions d'une meme
# architecture, selon le GPU sur lequel la tache a atterri, si bien qu'aucune
# echelle commune ne serait exacte pour toutes les courbes d'un panneau.
sweep = a2_load("*_taylor_N*.json",
                key=lambda j, p: _n_of(p),
                label=lambda j, p: (f"{_n_of(p)} : {round(j['actual_pct'])}~\\%, "
                                    f"{_hours(j):.0f}~h"))
# bornes communes aux trois panneaux : sans cela ResNet-101 s'affiche en 20..80
# et les deux Inception en 0..80, ce qui fausse la comparaison visuelle
SWEEP_LIM = ylims([h for m, _ in A2_SWEEP_MODELS for _, _, h in sweep.get(m, [])]
                  + [[A2_REF[m]] for m, _ in A2_SWEEP_MODELS if m in A2_REF])
cells, legend_pcts = [], []
for k, (model, nice) in enumerate(A2_SWEEP_MODELS):
    runs = sweep.get(model, [])
    if not runs:
        continue
    # teinte foncee = peu d'accelerateurs, donc elagage severe
    cols = []
    for i, (nseg, lab, _) in enumerate(runs):
        cols.append((f"n{nseg}", RAMP[max(0, len(RAMP) - 1 - i)], lab))
    name = f"ft_a2_sweep_{model}"
    n = write_csv(name, [c[0] for c in cols], [h for _, _, h in runs])
    cells.append(axis(name, cols, nice,
                      "Top-1 de validation (\\%)" if k == 0 else None,
                      "Époque", "width=4.6cm, height=3.4cm, scale only axis", n, ylim=SWEEP_LIM,
                      ref=A2_REF.get(model)))
    legend_pcts.append((model, [c[2] for c in cols]))

# une legende par panneau : les taux different d'un modele a l'autre
sweep_legs = []
for (model, _), (_, labels) in zip(A2_SWEEP_MODELS, legend_pcts):
    nm = f"leg:fta2{model.replace('_','')}"
    sweep_legs.append((nm, labels))
COLORRAMP = "\n".join(f"\\definecolor{{ramp{i}}}{{HTML}}{{{c.upper()}}}"
                      for i, c in enumerate(RAMP))
# les couleurs de rampe sont referencees par nom dans les CSV : on les redefinit
cells = [c.replace(f"color=#", "color=") for c in cells]
for i, c in enumerate(RAMP):
    cells = [x.replace(f"color={c}", f"color=ramp{i}") for x in cells]
legs_tex, legs_ref = [], []
for i, (nm, labels) in enumerate(sweep_legs):
    legs_tex.append(legend(nm, [(f"ramp{j}", l) for j, l in enumerate(labels)],
                           columns=len(labels), compact=True))
    legs_ref.append(f"\\ref{{{nm}}}")

(OUT / "ft_a2_sweep.tex").write_text("\n".join([
    "% genere par make_ft_curves.py, ne pas editer a la main",
    COLORRAMP, "", "\n\n".join(legs_tex), "",
    grid(cells, 3), "\\vspace{4pt}",
    stacked_legends([n for n, _ in A2_SWEEP_MODELS_NICE], legs_ref)]) + "\n")

# ══ AXE 2 : wave initiale (annexe) ═══════════════════════════════════════
wave = a2_load("*_taylor_target*mb.json",
               key=lambda j, p: round(j.get("actual_pct", 0)),
               label=lambda j, p: f"{round(j.get('actual_pct', 0))} \\%")
NICE2 = dict(A2_SWEEP_MODELS)
NICE2.update({"inception_v2_bninception": "BN-Inception", "inception_v3": "Inception-V3",
              "resnet50": "ResNet-50", "resnet152": "ResNet-152",
              "inception_v1_googlenet": "GoogLeNet"})
wcells, wlegs_tex, wlegs_ref = [], [], []
for k, model in enumerate(sorted(wave)):
    runs = wave[model]
    cols = [(f"p{pct}", f"ramp{min(2*i+2, len(RAMP)-1)}", lab)
            for i, (pct, lab, _) in enumerate(runs)]
    name = f"ft_a2_wave_{model}"
    n = write_csv(name, [c[0] for c in cols], [h for _, _, h in runs])
    wcells.append(axis(name, cols, NICE2.get(model, model),
                       "Top-1 de validation (\\%)" if k % 4 == 0 else None,
                       "Époque" if k >= 4 else None, PANEL4, n,
                       ylim=ylims([h for _, _, h in runs]
                                  + ([[A2_REF[model]]] if model in A2_REF else [])),
                       ref=A2_REF.get(model)))
    nm = f"leg:ftw{model.replace('_','')}"
    wlegs_tex.append(legend(nm, [(c[1], c[2]) for c in cols], columns=len(cols)))
    wlegs_ref.append(f"\\ref{{{nm}}}")

(OUT / "ft_a2_wave.tex").write_text("\n".join([
    "% genere par make_ft_curves.py, ne pas editer a la main",
    COLORRAMP, "", "\n\n".join(wlegs_tex), "",
    grid(wcells, 4), "\\vspace{4pt}",
    stacked_legends([NICE2.get(m, m) for m in sorted(wave)], wlegs_ref)]) + "\n")

# ── recapitulatif ────────────────────────────────────────────────────────
print("AXE 1")
for model, nice, npan, ncrit in made1:
    print(f"  ft_a1_{model:<14} {npan} panneaux, {ncrit} criteres")
print("AXE 2")
print(f"  ft_a2_sweep     {len(cells)} panneaux "
      f"({sum(len(l) for _, l in legend_pcts)} trajectoires)")
print(f"  ft_a2_wave      {len(wcells)} panneaux "
      f"({sum(len(wave[m]) for m in wave)} trajectoires)")
