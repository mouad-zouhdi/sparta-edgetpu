#!/usr/bin/env python3
"""
make_annexe_multitpu.py — annexe de l'axe multi-TPU.

L'axe mono-TPU dispose de tables exhaustives en annexe ; l'axe multi-TPU n'en avait
aucune, alors que sa campagne compte 946 configurations. Ce script produit
figs/annexe_multitpu.tex, qui rassemble trois tables :

  B.1  synthese par checkpoint : besoin, precision, meilleure repartition en
       latence et en debit, avec les valeurs associees ;
  B.2  journal des entrainements : budget d'epoques, temps GPU, et precision
       flottante sur les 50 000 images avant elagage, juste apres, et apres
       recuperation ;
  B.3  les 22 manieres de repartir 8 accelerateurs, qui donnent sa lecture a la
       notation employee tout au long du chapitre 5.

Donnees : wave_bench/results/H_accuracy_vs_tpu/accuracy_vs_tpu.csv,
multi_tpu/models/pruning_logs_imagenet/*.json et les fichiers TFLite INT8.
"""
from pathlib import Path
import csv, glob, json, os, re

ROOT = Path("/home/a131/Desktop/Project")
MODELS = ROOT / "multi_tpu/models"
ACC = MODELS / "wave_bench/results/H_accuracy_vs_tpu/accuracy_vs_tpu.csv"
LOGS = MODELS / "pruning_logs_imagenet"
OUT = Path("figs/annexe_multitpu.tex")

NAME = {"inception_v1_googlenet": "GoogLeNet", "inception_v2_bninception": "BN-Inception",
        "inception_v3": "Inception-V3", "inception_v4": "Inception-V4",
        "inception_resnet_v2": "Inception-ResNet-V2", "resnet50": "ResNet-50",
        "resnet101": "ResNet-101", "resnet152": "ResNet-152"}
ORDER = ["inception_v1_googlenet", "inception_v2_bninception", "inception_v3",
         "inception_v4", "inception_resnet_v2", "resnet50", "resnet101", "resnet152"]


def fr(x, n=2):
    """Nombre au format francais, virgule decimale.

    Le signe negatif passe en mode mathematique : en mode texte LaTeX compose un
    trait d'union, visiblement plus court qu'un signe moins."""
    s = f"{x:.{n}f}".replace(".", ",")
    return s if not s.startswith("-") else "$-$" + s[1:]


def tflite_sizes() -> dict:
    """Taille du fichier TFLite INT8 soumis au compilateur, par checkpoint."""
    out = {}
    for f in glob.glob(str(MODELS / "tflite_int8*/*.tflite")):
        b = os.path.basename(f)
        mb = os.path.getsize(f) / 2**20
        m = re.match(r"(.+?)_pruned(\d+)pct_taylor_int8\.tflite", b)
        if m:
            out[(m.group(1), int(m.group(2)))] = mb
            continue
        m = re.match(r"(.+?)_(?:pretrained_)?int8\.tflite", b)
        if m:
            out.setdefault((m.group(1), 0), mb)
    return out


def compact(shape: str) -> str:
    """4+4 -> 4$\\times$2 : les parts egales sont regroupees."""
    p = shape.split("+"); out = []; i = 0
    while i < len(p):
        j = i
        while j < len(p) and p[j] == p[i]:
            j += 1
        out.append(p[i] if j - i == 1 else f"{p[i]}$\\times${j - i}")
        i = j
    return "{+}".join(out).replace("{+}", "$+$")


def partitions(n: int, cap: int = None):
    """Toutes les facons d'ecrire n comme somme d'entiers, parts decroissantes."""
    cap = cap or n
    if n == 0:
        yield []
        return
    for k in range(min(n, cap), 0, -1):
        for rest in partitions(n - k, k):
            yield [k] + rest


# ── B.1 synthese par checkpoint ──────────────────────────────────────────────
sizes = tflite_sizes()
rows = list(csv.DictReader(open(ACC)))
rows.sort(key=lambda r: (ORDER.index(r["model"]), int(float(r["pct"]))))

t1 = ["\\setlength{\\tabcolsep}{4pt}",
      "\\begin{longtable}{@{}llrrrrlrlr@{}}",
      "\\caption[Synthèse de l'axe multi-TPU]{Synthèse par checkpoint de l'axe multi-TPU : "
      "besoin de logement, précision INT8 relevée sur 5000 images de validation, "
      "et répartitions mesurées comme les meilleures en latence puis en débit, "
      "avec les valeurs qu'elles atteignent.}\\label{tab:ann_multitpu}\\\\",
      "\\toprule",
      "Modèle & Élagage & Taille & Besoin & Top-1 & Top-5 & \\multicolumn{2}{c}{Meilleure latence}"
      " & \\multicolumn{2}{c}{Meilleur débit} \\\\",
      "\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
      " & (\\%) & (Mio) & & (\\%) & (\\%) & Répartition & (ms) & Répartition & (im/s) \\\\",
      "\\midrule", "\\endfirsthead",
      "\\toprule",
      "Modèle & Élagage & Taille & Besoin & Top-1 & Top-5 & \\multicolumn{2}{c}{Meilleure latence}"
      " & \\multicolumn{2}{c}{Meilleur débit} \\\\",
      "\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
      " & (\\%) & (Mio) & & (\\%) & (\\%) & Répartition & (ms) & Répartition & (im/s) \\\\",
      "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
prev = None
for r in rows:
    m, pct = r["model"], int(float(r["pct"]))
    b = r["budget"]
    bes = "$>8$" if b in ("", None) or b.lower() == "nan" else str(int(float(b)))
    if prev is not None and m != prev:
        t1.append("\\addlinespace[2pt]")
    prev = m
    t1.append(f"{NAME[m]} & {pct if pct else '--'} & {fr(sizes[(m, pct)])} & {bes} & "
              f"{fr(float(r['top1_pct']))} & {fr(float(r['top5_pct']))} & "
              f"{compact(r['shape_latence'])} & {fr(float(r['lat_best']), 1)} & "
              f"{compact(r['shape_debit'])} & {fr(float(r['fps_best']), 0)} \\\\")
t1.append("\\end{longtable}")

# ── B.2 journal des entrainements ────────────────────────────────────────────
logs = []
for f in sorted(glob.glob(str(LOGS / "*taylor*.json"))):
    if "pipeline_summary" in f:
        continue
    d = json.loads(Path(f).read_text())
    logs.append((d["model"], d["actual_pct"], d["ft_epochs"],
                 d["duration_s"]["finetune"] / 3600.0,
                 d["ref_top1_pct"], d["post_prune_top1_pct"], d["final_top1_pct"]))
logs.sort(key=lambda r: (ORDER.index(r[0]), r[1]))

t2 = ["\\setlength{\\tabcolsep}{4pt}",
      "\\begin{longtable}{@{}lrrrrrrr@{}}",
      "\\caption[Journal des entraînements de l'axe multi-TPU]{Récupération des 36 checkpoints "
      "élagués de l'axe multi-TPU. Les précisions sont en flottant sur les 50 000 images de "
      "validation, ce qui les rend comparables entre elles ; la colonne « après coupe » "
      "est relevée avant toute récupération.}\\label{tab:ann_ft}\\\\",
      "\\toprule",
      "Modèle & Élagage & Époques & Calcul & \\multicolumn{3}{c}{Top-1 flottant (\\%)} & Écart \\\\",
      "\\cmidrule(lr){5-7}",
      " & (\\%) & & (h GPU) & Référence & Après coupe & Récupéré & (pts) \\\\",
      "\\midrule", "\\endfirsthead",
      "\\toprule",
      "Modèle & Élagage & Époques & Calcul & \\multicolumn{3}{c}{Top-1 flottant (\\%)} & Écart \\\\",
      "\\cmidrule(lr){5-7}",
      " & (\\%) & & (h GPU) & Référence & Après coupe & Récupéré & (pts) \\\\",
      "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
prev = None
for m, pct, ep, h, ref, post, fin in logs:
    if prev is not None and m != prev:
        t2.append("\\addlinespace[2pt]")
    prev = m
    t2.append(f"{NAME[m]} & {fr(pct, 1)} & {ep} & {fr(h, 1)} & {fr(ref)} & "
              f"{fr(post)} & {fr(fin)} & {fr(fin - ref, 2)} \\\\")
t2.append("\\end{longtable}")

# ── B.3 les 22 repartitions ──────────────────────────────────────────────────
parts = list(partitions(8))
parts.sort(key=lambda p: (len(p), [-x for x in p]))
t3 = ["\\begin{table}[H]", "\\centering\\small",
      "\\caption[Les 22 répartitions de 8 accélérateurs]{Les 22 manières de répartir "
      "8 accélérateurs entre instances, soit les 22 façons d'écrire 8 comme une somme "
      "d'entiers. La notation compacte regroupe les parts égales et est celle employée "
      "au chapitre~\\ref{chap:resultats}.}",
      "\\label{tab:ann_shapes}",
      "\\begin{tabular}{@{}rll@{}}", "\\toprule",
      "Instances & Profondeurs & Notation compacte \\\\", "\\midrule"]
prevn = None
for p in parts:
    if prevn is not None and len(p) != prevn:
        t3.append("\\addlinespace[2pt]")
    prevn = len(p)
    t3.append(f"{len(p)} & {' + '.join(str(x) for x in p)} & {compact('+'.join(str(x) for x in p))} \\\\")
t3 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

body = "\n".join([
    "% genere par make_annexe_multitpu.py, ne pas editer a la main",
    "",
    "\\chapter{Mesures détaillées de l'axe multi-TPU}",
    "\\label{ann:multitpu}",
    "",
    "Cette annexe rassemble les relevés par configuration de l'axe multi-TPU, dont le "
    "chapitre~\\ref{chap:resultats} ne présente que des synthèses et quelques cas. Le "
    "tableau~\\ref{tab:ann_multitpu} donne, pour chacun des 43 checkpoints mesurés, son "
    "besoin de logement, sa précision et les deux répartitions que la mesure retient ; le "
    "tableau~\\ref{tab:ann_ft} rend compte du coût de la récupération et de l'effondrement "
    "de précision que l'élagage provoque avant elle ; le tableau~\\ref{tab:ann_shapes} "
    "énumère les 22 répartitions possibles.",
    "", "\\section{Synthèse par checkpoint}", "", "{\\footnotesize", *t1, "}",
    "", "\\clearpage", "\\section{Récupération après élagage}", "", "{\\footnotesize", *t2, "}",
    "", "\\section{Répartitions des 8 accélérateurs}", "", *t3, ""])
OUT.write_text(body + "\n")
print(f"{OUT} : {len(rows)} checkpoints, {len(logs)} entraînements, {len(parts)} répartitions")
