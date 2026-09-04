#!/usr/bin/env python3
"""
aggregate_partitions.py — agrege partitions_sweepN.csv sur les passes.

Le CSV brut a une ligne par (passe, checkpoint, repartition, instance). Deux
niveaux d'agregat en sortie :

  <out>_par_config.csv    une ligne par (checkpoint, repartition)
      debit total, latence de la pire instance, premiere inference, avec
      moyenne, ecart-type et demi-intervalle de confiance a 95 % CALCULES
      ENTRE PASSES. C'est le seul niveau qui donne une barre d'erreur
      honnete : la dispersion dominante sur cette carte est entre mesures
      (sigma ~ 1,6 % du debit, experience C' du 2026-08-16), pas a
      l'interieur d'une mesure.

  <out>_par_instance.csv  une ligne par (checkpoint, repartition, instance)
      idem plus les temps par etage, l'etage critique et le desequilibre
      max/moyenne des etages, qui est ce qui plafonne le debit d'un pipeline.

Jointure avec segment_metadata_sweepN.csv quand il est fourni : volume interne
et externe de l'instance, soit les termes I et E du modele de latence.
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def ci95(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v[0]) if len(v) else float("nan"), 0.0, 0.0, len(v))
    sd = float(np.std(v, ddof=1))
    return float(np.mean(v)), sd, 1.96 * sd / np.sqrt(len(v)), len(v)


def parse_list(s):
    return [float(x) for x in s.split("|")] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, type=Path)
    ap.add_argument("--meta-csv", type=Path)
    ap.add_argument("--out-prefix", required=True, type=Path)
    a = ap.parse_args()

    rows = list(csv.DictReader(a.in_csv.open()))
    print(f"[lu] {len(rows)} lignes")

    meta = {}
    if a.meta_csv and a.meta_csv.exists():
        for r in csv.DictReader(a.meta_csv.open()):
            if r["seg_idx"] == "-1":
                meta[(r["model"], int(r["pct"]), int(r["N"]))] = r

    # ---- niveau instance ----
    byi = defaultdict(list)
    for r in rows:
        byi[(r["model"], int(r["pct"]), r["shape"], int(r["inst_idx"]))].append(r)

    inst_cols = ["model", "pct", "fit_tpu", "shape", "n_instances", "inst_idx",
                 "n_stages", "n_passes",
                 "lat_ms_mean", "lat_ms_sd", "lat_ms_ci95", "lat_ms_p95",
                 "cold_ms_mean", "cold_ms_sd", "cold_ms_ci95",
                 "fps_mean", "fps_sd", "fps_ci95", "fps_cv_pct",
                 "stage_ms_median", "stage_ms_thr_median",
                 "stage_critical_idx", "stage_imbalance",
                 "setup_ms_mean", "n_inferences_lat", "n_inferences_thr",
                 "on_chip_mb", "off_chip_mb", "size_mb"]
    inst_out = []
    for (model, pct, shape, j), rs in sorted(byi.items()):
        lat = [float(r["lat_e2e_median"]) for r in rs]
        cold = [float(r["cold_e2e_ms"]) for r in rs]
        fps = [float(r["thr_fps_instance"]) for r in rs]
        st_l = np.array([parse_list(r["lat_stage_median"]) for r in rs], dtype=float)
        st_t = np.array([parse_list(r["thr_stage_median"]) for r in rs], dtype=float)
        sl = np.nanmedian(st_l, axis=0) if st_l.size else np.array([])
        st = np.nanmedian(st_t, axis=0) if st_t.size else np.array([])
        base = st if st.size and np.isfinite(st).all() else sl
        m_l, sd_l, ci_l, n = ci95(lat)
        m_c, sd_c, ci_c, _ = ci95(cold)
        m_f, sd_f, ci_f, _ = ci95(fps)
        md = meta.get((model, pct, int(rs[0]["n_stages"])), {})
        inst_out.append(dict(
            model=model, pct=pct, fit_tpu=rs[0]["fit_tpu"], shape=shape,
            n_instances=rs[0]["n_instances"], inst_idx=j,
            n_stages=rs[0]["n_stages"], n_passes=n,
            lat_ms_mean=round(m_l, 4), lat_ms_sd=round(sd_l, 4), lat_ms_ci95=round(ci_l, 4),
            lat_ms_p95=round(float(np.mean([float(r["lat_e2e_p95"]) for r in rs])), 4),
            cold_ms_mean=round(m_c, 3), cold_ms_sd=round(sd_c, 3), cold_ms_ci95=round(ci_c, 3),
            fps_mean=round(m_f, 3), fps_sd=round(sd_f, 3), fps_ci95=round(ci_f, 3),
            fps_cv_pct=round(100 * sd_f / m_f, 3) if m_f else float("nan"),
            stage_ms_median="|".join(f"{x:.4f}" for x in sl),
            stage_ms_thr_median="|".join(f"{x:.4f}" for x in st),
            stage_critical_idx=int(np.argmax(base)) if base.size else -1,
            stage_imbalance=round(float(np.max(base) / np.mean(base)), 4) if base.size else float("nan"),
            setup_ms_mean=round(float(np.mean([float(r["setup_ms"]) for r in rs])), 2),
            n_inferences_lat=int(sum(int(r["lat_n"]) for r in rs)),
            n_inferences_thr=int(sum(int(r["thr_n"]) for r in rs)),
            on_chip_mb=md.get("on_chip_mb", ""), off_chip_mb=md.get("off_chip_mb", ""),
            size_mb=md.get("size_mb", ""),
        ))

    # ---- niveau configuration ----
    byc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        byc[(r["model"], int(r["pct"]), r["shape"])][int(r["pass"])].append(r)
    cfg_cols = ["model", "pct", "fit_tpu", "shape", "n_instances", "n_passes",
                "fps_total_mean", "fps_total_sd", "fps_total_ci95", "fps_total_cv_pct",
                "lat_worst_ms_mean", "lat_worst_ms_sd", "lat_worst_ms_ci95",
                "lat_best_ms_mean", "cold_worst_ms_mean", "cold_worst_ms_sd",
                "cold_worst_ms_ci95", "lat_ms_per_instance", "fps_per_instance",
                "stage_imbalance_max", "n_inferences_total"]
    cfg_out = []
    for (model, pct, shape), passes in sorted(byc.items()):
        tot, lw, lb, cw, ninf = [], [], [], [], 0
        per_lat, per_fps, imb = defaultdict(list), defaultdict(list), []
        for p, rs in passes.items():
            tot.append(float(rs[0]["thr_fps_total"]))
            lw.append(max(float(r["lat_e2e_median"]) for r in rs))
            lb.append(min(float(r["lat_e2e_median"]) for r in rs))
            cw.append(max(float(r["cold_e2e_ms"]) for r in rs))
            ninf += sum(int(r["lat_n"]) + int(r["thr_n"]) for r in rs)
            for r in rs:
                per_lat[int(r["inst_idx"])].append(float(r["lat_e2e_median"]))
                per_fps[int(r["inst_idx"])].append(float(r["thr_fps_instance"]))
                s = parse_list(r["thr_stage_median"]) or parse_list(r["lat_stage_median"])
                if s: imb.append(max(s) / (sum(s) / len(s)))
        m_t, sd_t, ci_t, n = ci95(tot)
        m_w, sd_w, ci_w, _ = ci95(lw)
        m_cw, sd_cw, ci_cw, _ = ci95(cw)
        cfg_out.append(dict(
            model=model, pct=pct, fit_tpu=passes[list(passes)[0]][0]["fit_tpu"],
            shape=shape, n_instances=passes[list(passes)[0]][0]["n_instances"],
            n_passes=n,
            fps_total_mean=round(m_t, 3), fps_total_sd=round(sd_t, 3),
            fps_total_ci95=round(ci_t, 3),
            fps_total_cv_pct=round(100 * sd_t / m_t, 3) if m_t else float("nan"),
            lat_worst_ms_mean=round(m_w, 4), lat_worst_ms_sd=round(sd_w, 4),
            lat_worst_ms_ci95=round(ci_w, 4),
            lat_best_ms_mean=round(float(np.mean(lb)), 4),
            cold_worst_ms_mean=round(m_cw, 3), cold_worst_ms_sd=round(sd_cw, 3),
            cold_worst_ms_ci95=round(ci_cw, 3),
            lat_ms_per_instance="|".join(f"{np.mean(per_lat[k]):.2f}" for k in sorted(per_lat)),
            fps_per_instance="|".join(f"{np.mean(per_fps[k]):.1f}" for k in sorted(per_fps)),
            stage_imbalance_max=round(max(imb), 4) if imb else float("nan"),
            n_inferences_total=ninf,
        ))

    for name, cols, data in [("par_instance", inst_cols, inst_out),
                             ("par_config", cfg_cols, cfg_out)]:
        p = a.out_prefix.with_name(a.out_prefix.name + f"_{name}.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in data: w.writerow(r)
        print(f"-> {p} : {len(data)} lignes")

    if cfg_out:
        cv = [r["fps_total_cv_pct"] for r in cfg_out if r["fps_total_cv_pct"] == r["fps_total_cv_pct"]]
        npass = [r["n_passes"] for r in cfg_out]
        print(f"[repetabilite] CV du debit entre passes : median {np.median(cv):.2f} %, "
              f"p90 {np.percentile(cv,90):.2f} %")
        print(f"[passes] min {min(npass)}, median {np.median(npass):.0f}, max {max(npass)}")
        print(f"[inferences] total {sum(r['n_inferences_total'] for r in cfg_out):,}")


if __name__ == "__main__":
    main()
