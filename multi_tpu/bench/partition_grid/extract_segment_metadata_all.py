#!/usr/bin/env python3
"""
extract_segment_metadata_all.py — tailles et regime memoire des trois corpus.

Meme role que extract_segment_metadata.py, mais sur les 43 checkpoints des trois
campagnes multi-accelerateur et non sur le seul balayage. La cle porte le corpus
pour la meme raison que dans multi_tpu_corpus : les deux
`resnet101_pruned88pct_taylor` sont deux entrainements differents.

Une ligne par (corpus, checkpoint, N, segment), plus une ligne d'agregat par
(corpus, checkpoint, N) sous seg_idx = -1.

Colonnes memoire, telles que le compilateur les rapporte :
  on_chip_mb   volume loge en SRAM, charge une fois puis conserve
  off_chip_mb  volume externe, retransmis a chaque inference
Ce sont les termes I et E du modele de latence.
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from multi_tpu_corpus import build_corpus, CORPUS_DIR, ALL_N

COLS = ["corpus", "model", "pct", "fit_tpu", "N", "seg_idx", "n_segments",
        "file", "size_mb", "on_chip_mb", "off_chip_mb", "ops_total",
        "ops_edgetpu", "ops_cpu", "source"]


def find_report(seg_dir: Path):
    """Le rapport du compilateur voisin du dossier de segments, s'il existe.

    Les trois campagnes ne suffixent pas leurs noms de la meme facon : le
    balayage ecrit `<base>_N<N>_compile_report.json`, la vague et les baselines
    intercalent `_int8`, et la cible d'origine de la vague n'a pas de suffixe de
    N du tout. On essaie les trois formes plutot que de coder la convention.
    """
    stem = seg_dir.name[:-len("_segments")]
    for cand in (f"{stem}_compile_report.json",
                 f"{stem}_int8_compile_report.json",
                 re.sub(r"_N\d+$", "", stem) + "_compile_report.json"):
        p = seg_dir.parent / cand
        if p.exists():
            return p
    alt = seg_dir.parent / "regenerated_reports" / f"{stem}_compile_report.json"
    return alt if alt.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    a = ap.parse_args()

    rows, missing = [], []
    for c in build_corpus(a.root):
        for N in ALL_N:
            segs = c.seg.get(N)
            if not segs:
                continue
            rep = find_report(segs[0].parent)
            j = json.load(open(rep)) if rep else None
            src = "rapport_compilateur" if j else "taille_binaire"
            if not j:
                missing.append((c.tag, N))
            tot = dict(size=0.0, on=0.0, off=0.0, ops=0, tpu=0, cpu=0)
            for i, s in enumerate(segs):
                mb = s.stat().st_size / 1e6
                d = (j["segments"][i] if j and i < len(j.get("segments", []))
                     else {})
                on = d.get("on_chip_used_mb", float("nan"))
                off = d.get("off_chip_used_mb", float("nan"))
                oe, oc = d.get("ops_edgetpu", 0), d.get("ops_cpu", 0)
                rows.append(dict(corpus=c.corpus, model=c.model, pct=c.pct,
                                 fit_tpu=c.fit_tpu if c.fit_tpu is not None else "",
                                 N=N, seg_idx=i, n_segments=N, file=s.name,
                                 size_mb=round(mb, 3), on_chip_mb=on,
                                 off_chip_mb=off, ops_total=oe + oc,
                                 ops_edgetpu=oe, ops_cpu=oc, source=src))
                tot["size"] += mb
                if j:
                    tot["on"] += on; tot["off"] += off
                tot["ops"] += oe + oc; tot["tpu"] += oe; tot["cpu"] += oc
            t = j["totals"] if j else {}
            rows.append(dict(corpus=c.corpus, model=c.model, pct=c.pct,
                             fit_tpu=c.fit_tpu if c.fit_tpu is not None else "",
                             N=N, seg_idx=-1, n_segments=N, file="",
                             size_mb=round(tot["size"], 3),
                             on_chip_mb=t.get("on_chip_used_mb", float("nan")),
                             off_chip_mb=t.get("off_chip_used_mb", float("nan")),
                             ops_total=tot["ops"], ops_edgetpu=tot["tpu"],
                             ops_cpu=tot["cpu"], source=src))

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(rows)} lignes -> {a.out_csv}")
    if missing:
        print(f"sans rapport du compilateur ({len(missing)}) : "
              f"{sorted(set(m[0] for m in missing))}")


if __name__ == "__main__":
    main()
