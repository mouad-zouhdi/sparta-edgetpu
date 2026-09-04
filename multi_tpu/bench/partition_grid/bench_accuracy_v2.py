#!/usr/bin/env python3
"""
bench_accuracy_v2.py — Top-1/Top-5 int8 sur ImageNet val, un point par
(kind, model, pct) unique (20 mesures : 8 baselines + 12 wave paliers).

Utilise wave_configs_v2 pour l'énumération, discover_segments pour les paths.
Chaque mesure utilise N=1 (single TPU, simple + rapide).

Usage :
  python bench_accuracy_v2.py --imagenet-val ... --pruned-root ... \\
      --baseline-root ... --out-csv outputs/accuracy_v2.csv \\
      --n-images 2000
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from wave_configs_v2 import (
    build_all_bench_points, discover_segments, PREPROC,
    WAVE_PALIERS, BASELINE_MODELS,
)


def list_val_images(val_root: Path, n_images: int, seed: int = 42):
    class_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    all_items = []
    for d in class_dirs:
        for f in sorted(d.iterdir()):
            all_items.append((f, class_to_idx[d.name]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_items))
    return [all_items[i] for i in perm[:n_images]]


def preprocess_one(path: Path, pp: dict) -> np.ndarray:
    input_size = pp["input_size"]
    resize_size = int(input_size * 256 / 224)
    mean = np.asarray(pp["mean"], dtype=np.float32)
    std = np.asarray(pp["std"], dtype=np.float32)
    range_255 = pp["input_range_255"]
    bgr = pp["bgr"]

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w = resize_size; new_h = int(round(h * resize_size / w))
    else:
        new_h = resize_size; new_w = int(round(w * resize_size / h))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - input_size) // 2; top = (new_h - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))
    arr = np.asarray(img, dtype=np.float32)
    if bgr:
        arr = arr[..., ::-1].copy()
    if not range_255:
        arr /= 255.0
    return (arr - mean) / std


def quantize_input(fp: np.ndarray, scale: float, zp: int, dtype) -> np.ndarray:
    q = np.round(fp / scale + zp)
    if dtype == np.int8:
        return np.clip(q, -128, 127).astype(np.int8)
    return np.clip(q, 0, 255).astype(np.uint8)


def infer_single_tpu(seg: Path, model: str, items, n_images: int, tpu_id: int = 0) -> dict:
    import tflite_runtime.interpreter as tflite
    d = tflite.load_delegate("libedgetpu.so.1", options={"device": f":{tpu_id}"})
    interp = tflite.Interpreter(model_path=str(seg), experimental_delegates=[d])
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    inp_scale, inp_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    pp = PREPROC[model]

    n_top1 = n_top5 = 0
    t0 = time.perf_counter()
    for i, (path, label) in enumerate(items[:n_images]):
        fp = preprocess_one(path, pp)
        q = quantize_input(fp, inp_scale, inp_zp, inp["dtype"])
        interp.set_tensor(inp["index"], q[np.newaxis, ...])
        interp.invoke()
        logits_q = interp.get_tensor(out["index"])[0]
        logits = (logits_q.astype(np.float32) - out_zp) * out_scale
        top5_idx = np.argpartition(logits, -5)[-5:]
        n_top1 += int(np.argmax(logits) == label)
        n_top5 += int(label in top5_idx)
        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {i+1}/{n_images}  top1={100*n_top1/(i+1):.2f}%  "
                  f"top5={100*n_top5/(i+1):.2f}%  ({(i+1)/elapsed:.1f} img/s)",
                  flush=True)
    return dict(n=n_images, top1=n_top1, top5=n_top5,
                sec=time.perf_counter() - t0)


CSV_COLS = ["timestamp", "kind", "model", "pct", "target_mb",
            "n_images", "top1_pct", "top5_pct", "n_top1", "n_top5",
            "duration_s", "fps"]


def csv_append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def csv_read_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    seen = set()
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            seen.add((row["kind"], row["model"], int(row["pct"])))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-val", required=True, type=Path)
    ap.add_argument("--pruned-root", required=True, type=Path)
    ap.add_argument("--baseline-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Liste des cibles uniques : 12 wave paliers + 8 baselines
    targets = []
    for (model, pct), target_mb in WAVE_PALIERS.items():
        targets.append(("wave", model, pct, target_mb))
    for model in BASELINE_MODELS:
        targets.append(("baseline", model, 0, None))
    print(f"[targets] {len(targets)} paliers à mesurer.")

    # Résoudre segment N=1 pour chaque cible
    pts = build_all_bench_points()
    by_tag = {p.tag: p for p in pts}

    seen = csv_read_keys(args.out_csv) if args.resume else set()

    print(f"[dataset] Chargement liste {args.imagenet_val} ...")
    items = list_val_images(args.imagenet_val, args.n_images)
    print(f"[dataset] {len(items)} images.")

    for i, (kind, model, pct, target_mb) in enumerate(targets, 1):
        key = (kind, model, pct)
        if key in seen:
            print(f"\n[{i}/{len(targets)}] SKIP {kind} {model} pct={pct} (déjà mesuré)")
            continue
        if kind == "wave":
            tag = f"{model}_wave_{pct}pct_target{target_mb}mb_N1"
        else:
            tag = f"{model}_baseline_N1"
        if tag not in by_tag:
            print(f"\n[{i}/{len(targets)}] {kind} {model} pct={pct} → SKIP: tag {tag} absent")
            continue
        pt = by_tag[tag]
        root = args.pruned_root if kind == "wave" else args.baseline_root
        if not discover_segments(pt, root):
            print(f"\n[{i}/{len(targets)}] {kind} {model} pct={pct} → SKIP: segments non trouvés")
            continue

        print(f"\n[{i}/{len(targets)}] {kind} {model} pct={pct} target={target_mb}MB")
        try:
            t0 = time.perf_counter()
            r = infer_single_tpu(pt.segments[0], model, items, args.n_images)
            elapsed = time.perf_counter() - t0
            row = dict(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                kind=kind, model=model, pct=pct, target_mb=target_mb,
                n_images=r["n"],
                top1_pct=round(100 * r["top1"] / r["n"], 3),
                top5_pct=round(100 * r["top5"] / r["n"], 3),
                n_top1=r["top1"], n_top5=r["top5"],
                duration_s=round(elapsed, 1), fps=round(r["n"] / elapsed, 2),
            )
            csv_append(args.out_csv, row)
            print(f"  ✓ top1={row['top1_pct']}%  top5={row['top5_pct']}%  "
                  f"fps={row['fps']}  ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
