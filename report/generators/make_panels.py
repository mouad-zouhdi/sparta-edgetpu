#!/usr/bin/env python3
"""
make_panels.py — assemble les fragments pgfplots de
work/final_tables_and_plots/ en panneaux multi-modeles pour le rapport.

Regles de mise en forme demandees :
  - 4 graphes par ligne au maximum ;
  - une seule legende par ensemble de graphes presentant des donnees similaires,
    placee sous le panneau (les legendes individuelles sont retirees) ;
  - les tables volumineuses partent en annexe.

Sortie : rapport_build/figs/*.tex
"""
from pathlib import Path
import re

SRC = Path("/home/a131/Desktop/Project/work/final_tables_and_plots")
OUT = Path("/home/a131/Desktop/Project/rapport_build/figs")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["resnet18", "resnet50", "vgg19", "wrn_28_10",
          "mobilenetv2", "googlenet", "squeezenet1_1"]

# taille d'un panneau en grille 4 colonnes sur une page A4 a marges 2,5 cm
PANEL = "width=3.9cm, height=3.0cm, scale only axis"
PANEL_W, PANEL_H = "3.9cm", "3.0cm"

BENCH = Path("/home/a131/Desktop/mesures_brut/benchmark_results.csv")


def base_sizes() -> dict:
    """Taille du fichier TFLite INT8 du modele non elague, par architecture.

    La taille suit la reduction de parametres a 0,2 Mio pres sur tout le panel,
    ce qui autorise a lire l'axe superieur comme base * (1 - x/100)."""
    import csv as _csv
    out = {}
    for r in _csv.DictReader(open(BENCH)):
        if r["tag"] != "finetuned":
            continue
        out[r["model"].replace("_finetuned", "")] = float(r["size_int8_mib"])
    return out


SIZES = base_sizes()


def size_ticks(base: float):
    """Positions en pourcentage de reduction et libelles en Mio.

    On choisit un pas rond donnant 3 a 5 graduations sous la taille de base,
    puis on convertit chaque taille en l'abscisse qui lui correspond."""
    for step in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20):
        vals = [step * k for k in range(1, 40) if step * k < base]
        if 2 <= len(vals) <= 5:
            break
    vals = sorted(vals, reverse=True)
    pos = [100.0 * (1 - v / base) for v in vals]
    fmt = (lambda v: f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ","))
    return pos, [fmt(v) for v in vals]


def top_axis(model: str, keep_label: bool) -> str:
    """Axe parallele superieur donnant la taille du modele INT8 en Mio.

    Superpose au panneau principal par `at=(main.south west)` : les deux axes
    partagent xmin, xmax et les dimensions exactes, ce que `scale only axis`
    garantit."""
    base = SIZES.get(model)
    if base is None:
        return ""
    pos, lab = size_ticks(base)
    o = [f"\\begin{{axis}}[width={PANEL_W}, height={PANEL_H}, scale only axis,",
         "  at=(main.south west), anchor=south west,",
         "  axis x line*=top, axis y line=none, hide y axis,",
         "  xmin=0, xmax=100, ymin=0, ymax=1,",
         "  xtick={" + ",".join(f"{x:.2f}" for x in pos) + "},",
         # la virgule decimale est le separateur de liste de pgfplots : sans
         # accolades, "0,8" produit deux graduations "0" et "8"
         "  xticklabels={" + ",".join("{" + s + "}" for s in lab) + "},",
         "  tick label style={font=\\tiny},",
         # pas d'etiquette d'axe : elle entrerait en collision avec le titre du
         # panneau, l'unite est rappelee dans la legende de figure
         "  ]"]
    o.append("\\end{axis}")
    return "\n".join(o)


LABELS = [
    ("xlabel={Parameter reduction (\\%)}", "xlabel={Réduction (\\%)}"),
    ("ylabel={Edge TPU speedup ($\\times$)}", "ylabel={Accélération ($\\times$)}"),
    ("ylabel={INT8 Top-1 acc.\\ (\\%)}", "ylabel={Top-1 INT8 (\\%)}"),
    ("xlabel={Pruning target (\\%)}", "xlabel={Cible d'élagage (\\%)}"),
    ("ylabel={Top-1 val acc right after pruning (\\%)}", "ylabel={Top-1 après élagage (\\%)}"),
    ("xlabel={Edge TPU latency (ms)}", "xlabel={Latence Edge TPU (ms)}"),
]

CRITERIA = [
    ("impmagnitudel1", "o",         "Magnitude L1"),
    ("impmagnitudel2", "square",    "Magnitude L2"),
    ("impbnscale",     "diamond",   "BN-Scale"),
    ("impfpgm",        "triangle",  "FPGM"),
    ("imptaylor",      "triangle*", "Taylor"),
    ("impobdc",        "pentagon",  "OBD-C"),
    ("imprandom",      "x",         "Random"),
]
COLORDEFS = """\\definecolor{impmagnitudel1}{HTML}{1F77B4}
\\definecolor{impmagnitudel2}{HTML}{AEC7E8}
\\definecolor{impbnscale}{HTML}{FF7F0E}
\\definecolor{impfpgm}{HTML}{2CA02C}
\\definecolor{imptaylor}{HTML}{D62728}
\\definecolor{impobdc}{HTML}{9467BD}
\\definecolor{imprandom}{HTML}{7F7F7F}"""


def _drop_braced(t: str, key: str, opener: str = "={") -> str:
    """Retire `key<opener>...}` en respectant les accolades imbriquees."""
    while True:
        i = t.find(key + opener)
        if i < 0:
            return t
        j = i + len(key) + len(opener)
        depth = 1
        while j < len(t) and depth:
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
            j += 1
        while j < len(t) and t[j] in ", ":      # virgule et espace qui suivent
            j += 1
        t = t[:i] + t[j:]


def strip_fragment(path: Path, keep_ylabel: bool, keep_xlabel: bool = True,
                   size_axis: str | None = None, keep_toplabel: bool = False) -> str:
    """Retire la legende du fragment, le redimensionne, allege les polices."""
    t = path.read_text()
    t = re.sub(r"^%.*\n", "", t, flags=re.M)                 # commentaires d'entete
    t = re.sub(r"\\definecolor\{[^}]*\}\{HTML\}\{[^}]*\}\n", "", t)  # couleurs mutualisees
    t = _drop_braced(t, "\\addlegendentry", "{")        # legendes individuelles
    t = _drop_braced(t, "legend style")
    t = re.sub(r"legend columns=-?\d+, ", "", t)
    # graduations explicites trop denses pour un panneau etroit
    t = t.replace("xtick={10,20,30,40,50,60,70,80,90}", "xtick={20,40,60,80}")
    if size_axis:
        # l'axe parallele exige des bornes fixes et un nom d'ancrage : en
        # autoscale les deux axes ne se superposeraient pas
        t = t.replace("width=8cm, height=6cm",
                      PANEL + ", name=main, xmin=0, xmax=100, max space between ticks=34")
        # le titre est ancre au bord haut de l'axe principal, ou l'axe parallele
        # place ses graduations : sans decalage les deux se chevauchent
        t = t.replace("title style={font=\\small}",
                      "title style={font=\\scriptsize, yshift=7pt}")
    else:
        t = t.replace("width=8cm, height=6cm", PANEL + ", max space between ticks=34")
    t = t.replace("tick label style={font=\\scriptsize}", "tick label style={font=\\tiny}")
    t = t.replace("label style={font=\\scriptsize}", "label style={font=\\tiny}")
    t = t.replace("title style={font=\\small}", "title style={font=\\scriptsize}")
    if not keep_ylabel:                                       # une seule etiquette Y par ligne
        t = re.sub(r"ylabel=\{[^}]*\}, ", "", t)
    if not keep_xlabel:                                       # etiquette X sur la derniere ligne seule
        t = re.sub(r"xlabel=\{[^}]*\}, ", "", t)
    # 2. chaque panneau ne porte que le nom du modele ; le detail va en legende
    t = re.sub(r"(title=\{[^}\u2014]*)\s*\u2014[^}]*\}", r"\1}", t)
    # autoscale : on retire les bornes fixes
    t = re.sub(r"xmin=-?[\d.]+, xmax=-?[\d.]+, ymin=[^,]*, ymax=[^,]*,", "", t)
    # la verticale "tient en SRAM" devient un trait plein-cadre, sans effet
    # sur les limites (sinon son ymax code en dur casserait l'autoscale)
    t = re.sub(
        r"\\addplot\[mark=none, color=red!70!black, dash dot, thick\] coordinates \{\(([0-9.]+),0\) \([0-9.]+,[0-9.]+\)\};",
        r"\\draw[red!70!black, dash dot, thick] ({axis cs:\1,0}|-{rel axis cs:0,0}) -- ({axis cs:\1,0}|-{rel axis cs:0,1});",
        t)
    # asymptote a la precision du modele non elague : on la lit sur l'etoile du
    # fragment Pareto, seul endroit ou cette valeur figure
    m = re.search(r"\\addplot\[[^\n]*mark=star,[^\n]*coordinates \{\(([0-9.]+),([0-9.]+)\)\};", t)
    if m:
        y = m.group(2)
        # \draw plein-cadre plutot qu'un \addplot : en autoscale les bornes ne
        # sont pas lisibles, et un addplot etendrait l'axe
        t = t.replace(m.group(0),
                      "\\draw[dashed, black, thin] ({rel axis cs:0,0}|-{axis cs:0," + y + "}) "
                      "-- ({rel axis cs:1,0}|-{axis cs:0," + y + "});\n" + m.group(0))
    for a, b in LABELS:
        t = t.replace(a, b)
    # les CSV sont references depuis la racine de compilation
    t = t.replace("{curves/data/", "{plots/curves/data/")
    if size_axis:
        top = top_axis(size_axis, keep_toplabel)
        if top:
            t = t.replace("\\end{tikzpicture}", top + "\n\\end{tikzpicture}")
    return t.strip()


def shared_legend(name: str, extra, columns: int = 5) -> str:
    """Legende unique, construite a la main pour rester generique."""
    lines = [f"\\begin{{tikzpicture}}",
             "\\begin{axis}[hide axis, scale only axis, width=1pt, height=1pt,",
             "  xmin=0, xmax=1, ymin=0, ymax=1,",
             f"  legend columns={columns}, legend to name={name},",
             "  legend style={draw=gray!50, font=\\tiny,",
             "                /tikz/every even column/.append style={column sep=9pt}}]",
             "\\addplot[draw=none, forget plot] coordinates {(0,0)};"]
    for col, mark, label in CRITERIA:
        lines.append(f"\\addlegendimage{{color={col}, mark={mark}, mark size=1.5pt, line width=0.8pt}}"
                     f"\\addlegendentry{{{label}}}")
    for spec, label in extra:
        lines.append(f"\\addlegendimage{{{spec}}}\\addlegendentry{{{label}}}")
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines)


def panel(kind: str, legname: str, extra, models=None, size_axis: bool = False) -> str:
    """Grille 4 colonnes alignees ; la legende occupe la case libre en (2,4)."""
    models = models or MODELS
    cells = []
    for i, m in enumerate(models):
        frag = SRC / "curves" / kind / f"{m}.tex"
        cells.append(strip_fragment(frag, keep_ylabel=(i % 4 == 0), keep_xlabel=(i >= 4),
                                    size_axis=(m if size_axis else None),
                                    keep_toplabel=(size_axis and i % 4 == 0)))
    while len(cells) % 4:
        cells.append("")
    cells[-1] = ("\\raisebox{7.3mm}[0pt][0pt]{\\hspace*{3.3mm}\\ref{" + legname + "}}")   # cale le centre de la legende sur celui de WRN en X et de SqueezeNet en Y
    rows = ["  " + "\n  &\n  ".join(cells[r:r + 4]) for r in range(0, len(cells), 4)]
    return "\n".join([
        "% genere par make_panels.py, ne pas editer a la main",
        COLORDEFS, "",
        shared_legend(legname, extra, columns=1), "",
        "\\setlength{\\tabcolsep}{0pt}",
        "\\noindent\\makebox[\\linewidth][c]{%",
        "\\begin{tabular}{@{}cccc@{}}",
        (" \\\\[4pt]\n").join(rows),
        "\\end{tabular}}"])


def single(kind: str, model: str, legname: str, extra) -> str:
    """Graphe seul, legende partagee dessous, format plus large."""
    t = SRC / "curves" / kind / f"{model}.tex"
    frag = strip_fragment(t, keep_ylabel=True).replace(PANEL, "width=8.5cm, height=6cm")
    frag = frag.replace("tick label style={font=\\tiny}", "tick label style={font=\\scriptsize}")
    frag = frag.replace("label style={font=\\tiny}", "label style={font=\\scriptsize}")
    return "\n".join([COLORDEFS, "", shared_legend(legname, extra), "",
                      "\\noindent\\makebox[\\linewidth]{" + frag + "}",
                      "\\vspace{2pt}",
                      f"\\noindent\\makebox[\\linewidth]{{\\ref{{{legname}}}}}"])


REF_NOSPEED = ("mark=none, dashed, black, thin", "Référence non élaguée")
PAR_STAR  = ("only marks, mark=star, mark size=4pt, color=black, "
             "mark options={fill=yellow!90!orange}", "Référence non élaguée")
PAR_FRONT = ("mark=none, color=black, dotted, line width=0.9pt", "Front de Pareto")
PAR_OPAC  = ("draw=none, mark=none", "Opacité $\\propto$ compression")
REF_SRAM    = ("mark=none, color=red!70!black, dash dot, thick", "Passage sous 8\\,Mio")

(OUT / "panel_speedup.tex").write_text(
    panel("speedup_vs_compression", "leg:speedup", [REF_NOSPEED, REF_SRAM],
          size_axis=True) + "\n")
(OUT / "panel_accuracy.tex").write_text(
    panel("accuracy_vs_compression", "leg:accuracy", [REF_NOSPEED, REF_SRAM],
          size_axis=True) + "\n")
(OUT / "panel_post_prune.tex").write_text(
    panel("post_prune_acc", "leg:postprune", [REF_NOSPEED], size_axis=True) + "\n")
(OUT / "panel_pareto.tex").write_text(
    panel("pareto_accuracy_vs_latency", "leg:pareto", [PAR_STAR, PAR_FRONT, PAR_OPAC]) + "\n")

print("panneaux generes :", ", ".join(sorted(p.name for p in OUT.glob("panel_*.tex"))))
