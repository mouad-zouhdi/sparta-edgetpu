#!/usr/bin/env python3
"""
multi_tpu_corpus.py — catalogue unifie des checkpoints compilables a N = 1..8.

Les trois campagnes multi-accelerateur ont chacune leur racine de binaires, leur
convention de nommage de dossier de segments et leur facon d'identifier une
configuration. Ce module les reunit derriere une seule cle et une seule
resolution de chemins, pour que le banc n'ait pas a connaitre ces differences.

    balayage   23 checkpoints   3 architectures, un palier par nombre d'etages
    vague      12 checkpoints   7 architectures, une ou deux cibles de taille
    baselines   8 checkpoints   8 architectures, poids pre-entraines non elagues

Total 43 entrees, 42 noms distincts : `resnet101_pruned88pct_taylor` existe des
deux cotes et designe deux entrainements differents (vague 75 epoques calees sur
le taux predit, balayage 90 epoques calees sur le taux atteint). C'est pour
cette raison que la cle porte le corpus, et que les deux racines restent
separees. Ne pas les fusionner.

La cle publique est `(corpus, model, pct)`. Pour les baselines `pct = 0`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from wave_configs_v2 import (BenchPoint, PREPROC, WAVE_PALIERS, BASELINE_MODELS,
                             build_all_bench_points, discover_segments)
from sweepN_configs import build_sweepN_points, STREAMING_AT_TARGET

ALL_N = [1, 2, 3, 4, 5, 6, 7, 8]

# Sous-dossiers de la racine de deploiement, un par corpus.
CORPUS_DIR = {
    "sweepN":   "tflite_sweepn",
    "wave":     "tflite_pruned_v2",
    "baseline": "tflite_baseline_v2",
}


@dataclass
class Checkpoint:
    """Un modele mesurable, avec ses binaires pour chaque nombre d'etages."""
    corpus: str                  # sweepN | wave | baseline
    model: str
    pct: int                     # taux d'elagage atteint ; 0 pour une baseline
    fit_tpu: int | None          # nb d'accelerateurs vise par la campagne d'origine
    seg: dict[int, list[Path]] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.corpus, self.model, self.pct)

    @property
    def tag(self) -> str:
        return (f"{self.model}_baseline" if self.corpus == "baseline"
                else f"{self.model}_{self.corpus}_{self.pct}pct")

    @property
    def streaming_residue_mb(self) -> float:
        """Volume externe residuel a la cible, pour les points qui ne logent pas."""
        return STREAMING_AT_TARGET.get((self.model, self.pct), 0.0)

    def complete(self) -> bool:
        return all(n in self.seg for n in ALL_N)


def build_corpus(root: Path, corpora=("sweepN", "wave", "baseline")) -> list[Checkpoint]:
    """Resout les segments des trois campagnes sous `root`.

    `root` contient un sous-dossier par corpus (cf. CORPUS_DIR). Les checkpoints
    auxquels il manque un nombre d'etages sont renvoyes quand meme, avec
    `complete()` faux : c'est a l'appelant de decider s'il les ecarte ou s'il
    reduit sa grille de repartitions.
    """
    out: dict[tuple, Checkpoint] = {}

    def add(pts, corpus, fit_of):
        sub = root / CORPUS_DIR[corpus]
        for p in pts:
            if not discover_segments(p, sub):
                continue
            k = (corpus, p.model, p.pct)
            c = out.setdefault(k, Checkpoint(corpus, p.model, p.pct, fit_of(p)))
            c.seg[p.N] = list(p.segments)

    if "sweepN" in corpora:
        add(build_sweepN_points(ALL_N), "sweepN", lambda p: p.target_mb)
    wave_baseline = build_all_bench_points()
    if "wave" in corpora:
        # la vague declarait ses paliers par cible memoire, 8 / 16 / 32 Mio, ce
        # qui correspond a 1 / 2 / 4 accelerateurs. On convertit pour que la
        # colonne ait la meme unite que celle du balayage.
        add([p for p in wave_baseline if p.kind == "wave"], "wave",
            lambda p: p.target_mb // 8)
    if "baseline" in corpora:
        add([p for p in wave_baseline if p.kind == "baseline"], "baseline",
            lambda p: None)

    return sorted(out.values(), key=lambda c: (c.corpus, c.model, c.pct))


def partitions(n: int = 8) -> list[list[int]]:
    """Partitions de n, parts decroissantes. p(8) = 22."""
    def rec(rest, mx):
        if rest == 0:
            yield []
            return
        for k in range(min(rest, mx), 0, -1):
            for tail in rec(rest - k, k):
                yield [k] + tail
    return list(rec(n, n))


if __name__ == "__main__":
    r = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cps = build_corpus(r)
    full = [c for c in cps if c.complete()]
    print(f"racine {r}")
    for corpus in ("sweepN", "wave", "baseline"):
        sub = [c for c in cps if c.corpus == corpus]
        print(f"  {corpus:9s} {len(sub):3d} checkpoints, "
              f"{sum(c.complete() for c in sub):3d} complets N=1..8")
    print(f"  {'TOTAL':9s} {len(cps):3d} checkpoints, {len(full):3d} complets")
    print(f"  grille : {len(full)} x {len(partitions())} = "
          f"{len(full)*len(partitions())} points")
    for c in cps:
        if not c.complete():
            print("  INCOMPLET", c.tag, "manque N =",
                  sorted(set(ALL_N) - set(c.seg)))
