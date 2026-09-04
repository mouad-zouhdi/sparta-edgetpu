#!/usr/bin/env python3
"""
sweepN_configs.py — catalogue des 19 configurations du balayage N = 1..8.

Distinct de wave_configs_v2 pour trois raisons, toutes de correction :

1. **Collision de noms.** Les checkpoints post-recuperation s'appellent
   `<model>_pruned<pct>pct_taylor.pt`, sans identifiant de campagne. La wave de
   juillet et le sweep produisent tous deux un `resnet101_pruned88pct_taylor`,
   qui sont deux entrainements differents (juillet : 75 epoques calees sur le
   taux predit ; sweep : 90 epoques calees sur le taux atteint). Indexer les
   deux sous la meme cle `(kind, model, pct)` en ferait disparaitre un
   silencieusement. D'ou `kind="sweepN"`.

2. **Racine distincte.** Meme collision au niveau des dossiers de segments :
   `resnet101_pruned88pct_taylor_N1_segments` existe des deux cotes. Les
   binaires du sweep vivent donc dans leur propre racine.

3. **Indexation par N et non par cible memoire.** La wave declarait ses paliers
   par `target_mb` dans {8, 16, 32}. Le sweep produit un modele par nombre
   d'accelerateurs, y compris impair, ce que la grille en puissances de deux ne
   sait pas representer.

Les binaires de juillet ne sont ni deplaces ni ecrases : les deux campagnes
coexistent et restent distinguables dans les CSV de sortie.
"""
from __future__ import annotations
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from wave_configs_v2 import BenchPoint, PREPROC  # noqa: F401  (PREPROC reexporte)

# (model, pct atteint arrondi, N vise par le sweep)
# Le pct est celui qui apparait dans le nom du checkpoint post-FT.
SWEEPN_CONFIGS = [
    ("resnet101",            88, 1),
    ("resnet101",            73, 2),
    ("resnet101",            58, 3),
    ("resnet101",            43, 4),
    ("resnet101",            28, 5),
    ("resnet101",             7, 6),
    ("resnet101",            19, 7),
    ("inception_v4",         90, 1),
    ("inception_v4",         80, 2),
    ("inception_v4",         70, 3),
    ("inception_v4",         59, 4),
    ("inception_v4",         48, 5),
    ("inception_v4",         38, 6),
    ("inception_v4",         10, 8),
    ("inception_resnet_v2",  95, 1),
    ("inception_resnet_v2",  87, 2),
    ("inception_resnet_v2",  78, 3),
    ("inception_resnet_v2",  70, 4),
    ("inception_resnet_v2",  60, 5),
]

# Configurations recuperees par finish_sweepN.slurm : celles que la boucle
# guidee avait abandonnees sur son critere d'arret trop strict (off_chip == 0
# exactement) et son plafond de dix iterations. Elles reprennent le meilleur
# PREFT de la boucle, d'ou des taux d'elagage qui ne coincident avec aucun de
# ceux ci-dessus : aucune collision de nom de fichier n'est donc possible.
RECOVERED_CONFIGS = [
    ("inception_v4",         26, 7),
    ("inception_resnet_v2",  31, 8),
    ("inception_resnet_v2",  39, 7),
    ("inception_resnet_v2",  45, 6),
]

# Ces quatre configurations n'ont PAS toutes le meme statut que les autres.
# Trois d'entre elles ne tiennent pas strictement en memoire interne au nombre
# d'accelerateurs vise : la boucle guidee n'y avait jamais atteint un volume
# externe nul, et la recuperation ne change pas l'architecture. Le residu vaut
# 22 Kio pour inception_v4 a 26 %, negligeable, mais 0,24 a 0,30 Mio pour les
# trois Inception-ResNet-V2, soit 0,6 a 0,8 ms par inference sur PCIe. A
# signaler dans toute figure : ces points streament la ou les autres logent.
STREAMING_AT_TARGET = {
    ("inception_resnet_v2", 31): 0.238,
    ("inception_resnet_v2", 39): 0.244,
    ("inception_resnet_v2", 45): 0.302,
}

ALL_N = [1, 2, 3, 4, 5, 6, 7, 8]


def basename(model: str, pct: int) -> str:
    return f"{model}_pruned{pct}pct_taylor"


def build_sweepN_points(ns: list[int] | None = None) -> list[BenchPoint]:
    """Un BenchPoint par (configuration, N compile).

    Nommage produit par reconvert_sweepN.slurm :
        <basename>_N<N>_segments/<basename>_int8_edgetpu.tflite              (N=1)
        <basename>_N<N>_segments/<basename>_int8_segment_<i>_of_<N>_edgetpu.tflite
    """
    ns = ns or ALL_N
    pts = []
    for model, pct, n_target in SWEEPN_CONFIGS + RECOVERED_CONFIGS:
        base = basename(model, pct)
        for N in ns:
            seg_file = (f"{base}_int8_edgetpu.tflite" if N == 1
                        else f"{base}_int8_segment_*_of_{N}_edgetpu.tflite")
            pts.append(BenchPoint(
                kind="sweepN", model=model, pct=pct,
                # target_mb sert de colonne libre dans les CSV : on y range le N
                # vise par le sweep, seule facon de retrouver la configuration
                # d'origine une fois les mesures a plat.
                target_mb=n_target, N=N,
                tag=f"{model}_sweepN_{pct}pct_target{n_target}tpu_N{N}",
                segment_glob=f"{base}_N{N}_segments/{seg_file}",
            ))
    return pts


def unique_targets() -> list[tuple[str, str, int, int]]:
    """(kind, model, pct, n_target) : une entree par configuration.

    Utilise par la mesure de precision, qui n'a besoin que d'un point par
    modele et le mesure sur le binaire N=1.
    """
    return [("sweepN", m, p, n) for m, p, n in SWEEPN_CONFIGS + RECOVERED_CONFIGS]
