#!/usr/bin/env python3
"""
aggregate_partitions_fixedn.py — brut vers les deux agregats.

    partitions_fixedn.csv                brut, une ligne par (point, instance)
      -> partitions_fixedn_par_config.csv    une ligne par (checkpoint, repartition)
      -> partitions_fixedn_par_instance.csv  une ligne par instance, enrichie

ATTENTION A LA LECTURE DES COLONNES DE DISPERSION. La campagne compte une seule
passe. Les colonnes `*_std` et les percentiles decrivent la dispersion **entre
les mille inferences d'une meme mesure**, pas l'incertitude sur la moyenne. Les
deux different d'un facteur onze : l'erreur type de la moyenne sur mille
inferences vaut 0,018 %, alors que la dispersion reellement observee de cette
moyenne entre sessions vaut 0,197 % sur la campagne a dix passes. Utiliser un
`*_std` divise par racine de mille comme barre d'erreur annoncerait donc une
precision onze fois meilleure que la realite. Ces colonnes servent aux queues de
distribution, ce pour quoi mille inferences est le bon compte ; elles ne servent
pas a comparer deux configurations.

Latence d'une repartition = celle de l'instance la plus lente, jamais la moyenne
des instances : c'est le temps que voit celui qui attend sa reponse.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ["corpus", "model", "pct", "shape"]
CK = ["corpus", "model", "pct"]


def lst(s):
    return [float(x) for x in str(s).split("|")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, type=Path)
    ap.add_argument("--meta-csv", type=Path, default=None)
    ap.add_argument("--out-prefix", required=True, type=Path)
    a = ap.parse_args()

    d = pd.read_csv(a.in_csv)

    # ------------------------------------------------------------ instances --
    ins = d.copy()
    st = ins.thr_stage_median.map(lst)
    ins["stage_critical_idx"] = st.map(lambda v: int(np.argmax(v)))
    ins["stage_critical_ms"] = st.map(max)
    # desequilibre du decoupage : rapport de l'etage le plus lent a la moyenne.
    # C'est lui qui plafonne le debit, le pipeline valant l'inverse de son etage
    # le plus lent.
    ins["stage_imbalance"] = st.map(lambda v: max(v) / np.mean(v) if len(v) else np.nan)
    ins["stage_ratio_max_min"] = st.map(lambda v: max(v) / min(v) if min(v) > 0 else np.nan)
    if a.meta_csv and a.meta_csv.exists():
        m = pd.read_csv(a.meta_csv)
        m = m[m.seg_idx == -1][CK + ["N", "on_chip_mb", "off_chip_mb", "size_mb",
                                     "ops_cpu"]]
        ins = ins.merge(m, left_on=CK + ["n_stages"], right_on=CK + ["N"],
                        how="left").drop(columns=["N"])
    ins.to_csv(f"{a.out_prefix}_par_instance.csv", index=False)
    print(f"{len(ins)} lignes -> {a.out_prefix}_par_instance.csv")

    # -------------------------------------------------------------- configs --
    g = d.groupby(KEY, sort=False)
    worst = d.loc[g.lat_ms_mean.idxmax()].set_index(KEY)
    cold = d.loc[g.cold_ms_mean.idxmax()].set_index(KEY)
    cfg = pd.DataFrame({
        "fit_tpu": g.fit_tpu.first(),
        "n_instances": g.n_instances.first(),
        "tpu_used": g.tpu_used.first(),
        "fps_total": g.thr_fps_total.first(),
        "fps_min_instance": g.thr_fps_instance.min(),
        "fps_max_instance": g.thr_fps_instance.max(),
        # la latence d'une repartition est celle de son instance la plus lente
        "lat_worst_ms_mean": worst.lat_ms_mean,
        "lat_worst_ms_median": worst.lat_ms_median,
        "lat_worst_ms_std": worst.lat_ms_std,
        "lat_worst_ms_p95": worst.lat_ms_p95,
        "lat_worst_ms_p99": worst.lat_ms_p99,
        "lat_best_ms_mean": g.lat_ms_mean.min(),
        "cold_worst_ms_mean": cold.cold_ms_mean,
        "cold_worst_ms_median": cold.cold_ms_median,
        "cold_worst_ms_std": cold.cold_ms_std,
        "cold_worst_ms_p95": cold.cold_ms_p95,
        "gap_ms_median_worst": worst.gap_ms_median,
        "blk_fps_std_worst": worst.blk_fps_std,
        "setup_ms_total": g.setup_ms_mean.sum(),
        "stage_imbalance_max": ins.groupby(KEY).stage_imbalance.max(),
        "lat_ms_per_instance": g.lat_ms_mean.apply(lambda s: "|".join(f"{v:.4f}" for v in s)),
        "fps_per_instance": g.thr_fps_instance.apply(lambda s: "|".join(f"{v:.3f}" for v in s)),
        "n_lat": g.n_lat.first(), "n_thr": g.n_thr.first(), "n_cold": g.n_cold.first(),
    }).reset_index()
    cfg.to_csv(f"{a.out_prefix}_par_config.csv", index=False)
    print(f"{len(cfg)} lignes -> {a.out_prefix}_par_config.csv")


if __name__ == "__main__":
    main()
