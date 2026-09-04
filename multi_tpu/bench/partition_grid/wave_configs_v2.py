"""
wave_configs_v2.py — Catalogue étendu (paliers × N).

Structure alignée sur le format des résultats demandé :
- **BenchPoint** : 1 point de mesure = (palier ou baseline, N segments compilé)
- **TableRow** : 1 ligne du tableau D par archi × 1 des 4 configs multi-TPU
  Pour chaque ligne, on résout le meilleur palier via `closest(target_mb)`.

Paliers wave (target_mb) :
    8 MB  → cible 1 TPU (fully-on-chip sur 1 TPU)
    16 MB → cible 2 TPUs
    32 MB → cible 4 TPUs
    64 MB → cible 8 TPUs (n'existe pas dans notre wave)

Baselines pretrained : compilés à N ∈ {1, 2, 4, 8}.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


PREPROC = {
    "inception_v1_googlenet": dict(
        input_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        bgr=False, input_range_255=False),
    "inception_v2_bninception": dict(
        input_size=224, mean=(104.0, 117.0, 128.0), std=(1.0, 1.0, 1.0),
        bgr=True, input_range_255=True),
    "inception_v3": dict(
        input_size=299, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        bgr=False, input_range_255=False),
    "inception_v4": dict(
        input_size=299, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        bgr=False, input_range_255=False),
    "inception_resnet_v2": dict(
        input_size=299, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        bgr=False, input_range_255=False),
    "resnet50": dict(
        input_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        bgr=False, input_range_255=False),
    "resnet101": dict(
        input_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        bgr=False, input_range_255=False),
    "resnet152": dict(
        input_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        bgr=False, input_range_255=False),
}


# ──────────────────────────────────────────────
# Wave paliers post-FT : (model, pct_pruning) → target_mb
# ──────────────────────────────────────────────
WAVE_PALIERS = {
    ("inception_v2_bninception", 52): 8,
    ("inception_v3",              84): 8,
    ("inception_v3",              65): 16,
    ("inception_v4",              91): 8,
    ("inception_v4",              60): 32,
    ("inception_resnet_v2",       86): 16,
    ("inception_resnet_v2",       68): 32,
    ("resnet50",                  47): 16,
    ("resnet101",                 88): 8,
    ("resnet101",                 45): 32,
    ("resnet152",                 81): 16,
    ("resnet152",                 58): 32,
}


BASELINE_MODELS = [
    "inception_v1_googlenet", "inception_v2_bninception", "inception_v3",
    "inception_v4", "inception_resnet_v2", "resnet50", "resnet101", "resnet152",
]


# ──────────────────────────────────────────────
# Résolution segments : où chaque (kind, model, pct, N) est stocké
# ──────────────────────────────────────────────
@dataclass
class BenchPoint:
    kind: str            # "wave" or "baseline"
    model: str
    pct: int             # 0 for baseline, pruning_pct for wave
    target_mb: int | None    # None for baseline
    N: int
    tag: str             # unique id for CSV
    segment_glob: str    # pattern relative to root

    segments: list[Path] = field(default_factory=list)


def wave_basename(model: str, pct: int) -> str:
    return f"{model}_pruned{pct}pct_taylor"


def baseline_basename(model: str) -> str:
    return f"{model}_pretrained"


def build_all_bench_points() -> list[BenchPoint]:
    """Enumerate all (palier, N) pairs to potentially bench.

    Retourne une liste de BenchPoint prête à être filtrée selon les besoins.
    Le path des segments est un glob relatif (à résoudre via discover_segments()).
    """
    pts = []
    # Wave points
    for (model, pct), target_mb in WAVE_PALIERS.items():
        basename = wave_basename(model, pct)
        target_N = target_mb // 8
        for N in [1, 2, 3, 4, 5, 6, 7, 8]:
            if N == target_N:
                # segments dir sans suffixe _N (target du reconvert_wave)
                seg_dirname = f"{basename}_segments"
                seg_file_glob = (
                    f"{basename}_int8_edgetpu.tflite" if N == 1
                    else f"{basename}_int8_segment_*_of_{N}_edgetpu.tflite"
                )
            else:
                # segments dir avec suffixe _N (compile_extra)
                seg_dirname = f"{basename}_int8_N{N}_segments"
                seg_file_glob = (
                    f"{basename}_int8_edgetpu.tflite" if N == 1
                    else f"{basename}_int8_segment_*_of_{N}_edgetpu.tflite"
                )
            tag = f"{model}_wave_{pct}pct_target{target_mb}mb_N{N}"
            pts.append(BenchPoint(
                kind="wave", model=model, pct=pct, target_mb=target_mb, N=N,
                tag=tag, segment_glob=f"{seg_dirname}/{seg_file_glob}",
            ))
    # Baseline points
    # Note : conventions de nommage historique variables selon le job qui a compilé :
    #   - reconvert_baseline (N=1,2,4) : <basename>_N{N}_segments/
    #   - compile_extra_N (N=8) et compile local (N=3,5,6,7) : <basename>_int8_N{N}_segments/
    for model in BASELINE_MODELS:
        basename = baseline_basename(model)
        for N in [1, 2, 3, 4, 5, 6, 7, 8]:
            # On stocke seulement le pattern de fichier, discover_segments essaie les 2 dirs
            seg_file_glob = (
                f"{basename}_int8_edgetpu.tflite" if N == 1
                else f"{basename}_int8_segment_*_of_{N}_edgetpu.tflite"
            )
            tag = f"{model}_baseline_N{N}"
            pts.append(BenchPoint(
                kind="baseline", model=model, pct=0, target_mb=None, N=N,
                tag=tag, segment_glob=f"BASELINE_MULTI_DIR:{basename}:N{N}:{seg_file_glob}",
            ))
    return pts


def discover_segments(pt: BenchPoint, root: Path) -> bool:
    """Résout les paths absolus des segments pour un BenchPoint."""
    # Cas spécial : baselines peuvent être dans 2 conventions de dossier
    if pt.segment_glob.startswith("BASELINE_MULTI_DIR:"):
        _, basename, ncore, file_glob = pt.segment_glob.split(":")
        for seg_dirname in [f"{basename}_{ncore}_segments",
                            f"{basename}_int8_{ncore}_segments"]:
            trial_glob = f"{seg_dirname}/{file_glob}"
            matches = sorted(root.glob(trial_glob))
            if pt.N == 1:
                if len(matches) == 1:
                    pt.segments = matches
                    return True
            elif len(matches) == pt.N:
                matches.sort(key=lambda p: int(p.stem.split("_segment_")[1].split("_of_")[0]))
                pt.segments = matches
                return True
        return False

    matches = sorted(root.glob(pt.segment_glob))
    if pt.N == 1:
        if len(matches) == 1:
            pt.segments = matches
            return True
        return False
    if len(matches) != pt.N:
        return False
    matches.sort(key=lambda p: int(p.stem.split("_segment_")[1].split("_of_")[0]))
    pt.segments = matches
    return True


# ──────────────────────────────────────────────
# Table D : mapping (archi, config_row) → wave_palier utilisé
# ──────────────────────────────────────────────

# Configurations de la table D
TABLE_D_CONFIGS = [
    # (config_name, ideal_target_mb, n_pipelines, N_per_pipe, regime)
    ("Baseline 1 TPU",       0,  1, 1, "baseline"),   # baseline pretrained, N=1
    ("Parallel 8×1 TPU",     8,  8, 1, "parallel"),   # k=8 copies, chaque à N=1
    ("4× pipe N=2",         16,  4, 2, "pipeline"),   # 4 groupes, chaque = 1 pipeline N=2
    ("2× pipe N=4",         32,  2, 4, "pipeline"),   # 2 groupes, chaque = 1 pipeline N=4
    ("1× pipe N=8",         64,  1, 8, "pipeline"),   # 1 pipeline complet
]


def closest_palier(available_paliers: list[int], target: int) -> int | None:
    """Retourne le target_mb le plus proche parmi ceux disponibles."""
    if not available_paliers:
        return None
    return min(available_paliers, key=lambda p: (abs(p - target), p))


def paliers_for_archi(model: str) -> list[int]:
    """Liste des target_mb disponibles pour cette archi (via WAVE_PALIERS)."""
    return sorted(set(t for (m, _p), t in WAVE_PALIERS.items() if m == model))


def wave_pct_for_target(model: str, target_mb: int) -> int | None:
    """% pruning correspondant à ce palier pour cette archi. None si pas trouvé."""
    for (m, pct), t in WAVE_PALIERS.items():
        if m == model and t == target_mb:
            return pct
    return None


def resolve_table_row(archi: str, config_idx: int) -> dict:
    """Résout quel (palier, N) utiliser pour l'archi et le config_idx.

    Retourne dict avec :
      - config_name, regime, n_pipelines, N_per_pipe
      - kind ("wave" | "baseline")
      - target_mb, pct (None pour baseline)
      - is_fallback (True si palier choisi ≠ palier idéal)
      - point_tag : identifiant du BenchPoint à mesurer
    """
    cname, ideal_mb, n_pipe, N_per_pipe, regime = TABLE_D_CONFIGS[config_idx]

    if regime == "baseline":
        return dict(
            config_name=cname, regime="baseline", n_pipelines=1, N_per_pipe=1,
            kind="baseline", target_mb=None, pct=0, is_fallback=False,
            point_tag=f"{archi}_baseline_N1",
        )

    available = paliers_for_archi(archi)
    if not available:
        # googlenet — no wave. Use baseline at the desired N.
        return dict(
            config_name=cname, regime=regime, n_pipelines=n_pipe, N_per_pipe=N_per_pipe,
            kind="baseline", target_mb=None, pct=0, is_fallback=True,
            point_tag=f"{archi}_baseline_N{N_per_pipe}",
        )

    palier = closest_palier(available, ideal_mb)
    is_fallback = (palier != ideal_mb)
    pct = wave_pct_for_target(archi, palier)
    return dict(
        config_name=cname, regime=regime, n_pipelines=n_pipe, N_per_pipe=N_per_pipe,
        kind="wave", target_mb=palier, pct=pct, is_fallback=is_fallback,
        point_tag=f"{archi}_wave_{pct}pct_target{palier}mb_N{N_per_pipe}",
    )


def all_table_rows() -> list[tuple[str, dict]]:
    """Retourne [(archi, row_info), ...] pour tous les tableaux du groupe D."""
    out = []
    for archi in BASELINE_MODELS:
        for i in range(len(TABLE_D_CONFIGS)):
            out.append((archi, resolve_table_row(archi, i)))
    return out


if __name__ == "__main__":
    print("=== Wave paliers ===")
    for (m, p), t in WAVE_PALIERS.items():
        print(f"  {m:<25} pct={p:>2}%  target_mb={t}")

    print("\n=== Table D resolution par archi ===")
    for archi in BASELINE_MODELS:
        print(f"\n{archi}:")
        for i in range(len(TABLE_D_CONFIGS)):
            r = resolve_table_row(archi, i)
            fb = " [fallback]" if r["is_fallback"] else ""
            tag = r["point_tag"]
            print(f"  {r['config_name']:<22} → {tag}{fb}")

    print("\n=== BenchPoints (total: enum(kind, palier or None, N)) ===")
    pts = build_all_bench_points()
    print(f"  {len(pts)} points au total")
    from collections import Counter
    per_kind = Counter(p.kind for p in pts)
    print(f"  per kind: {dict(per_kind)}")
