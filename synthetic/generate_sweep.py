"""
Orchestrator for regeneration of all 400 synth models on dio, v2.

Improvements over v1 (see build_one_v2.py for the per-config pipeline):

* Pre-filter configs by params count (loaded from params_table.json). Configs
  above --max_params are marked "skipped_too_big" and NOT run — they OOM
  reliably on dio's 15 GB RAM. Rerun on a131 or cluster instead.

* Startup cleanup of /tmp/regen_* orphaned tmp dirs (they accumulate ~6 GB
  each when the parent process gets SIGKILL by the OOM killer, since
  the finally block doesn't run under SIGKILL).

* Disk-space guard before each config: if less than --min_disk_gb GB free,
  the sweep pauses (5s cycle) until space frees. Prevents disk-full cascades
  which triggered the previous stall.

* Retry semantics via --skip-existing:
    - aeq_status == success              → skip (already done)
    - metadata has "too_big_skipped: true" → skip (deliberately skipped)
    - anything else                       → retry

Usage:
  cd /home/mzouhdi/Bureau/generate_models
  ./miniforge/envs/gen/bin/python -u regen_sweep_v2.py \\
      --workers 2 --num_calib 100 --timeout 1800 \\
      --max_params 80000000 --min_disk_gb 30 --skip-existing
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Root of the generator. Defaults to this script's own directory; override with
# SYNTH_ROOT when the scripts are deployed outside the repository.
SYNTH_ROOT = Path(os.environ.get("SYNTH_ROOT", Path(__file__).resolve().parent))

# Interpreter used for the per-config subprocesses. Must be the synth-env one
# (Python 3.12 with TensorFlow and onnx2tf); defaults to the interpreter running
# this sweep, which is the right answer when launched from that env.
PYENV = os.environ.get("SYNTH_PY", sys.executable)

BUILD_ONE = SYNTH_ROOT / "build_one.py"
PARAMS_TABLE = SYNTH_ROOT / "params_table.json"

FAMILIES = ["sequential", "residual", "dense", "branched_2way", "branched_4way"]
DEPTHS = [4, 8, 16, 32]
WIDTHS = [16, 32, 64, 128, 256]
RESOLUTIONS = [96, 160, 224, 384]


def all_configs():
    """Return the 400 configurations of the factorial design."""
    for f in FAMILIES:
        for d in DEPTHS:
            for w in WIDTHS:
                for r in RESOLUTIONS:
                    yield (f, d, w, r)


def cleanup_tmp():
    """Remove orphaned /tmp/regen_* directories left by killed builds.

    build_one.py cleans up its scratch directory in a finally block, which does NOT
    run when the OOM killer SIGKILLs the process. Each orphan is around 6 GB, and on
    one host 27 of them filled the disk to 97 % within hours. Cleaning at start-up
    makes a resumed sweep self-healing.
    """
    orphans = glob.glob("/tmp/regen_*")
    freed_gb = 0.0
    for d in orphans:
        try:
            for root, dirs, files in os.walk(d):
                for f in files:
                    try: freed_gb += os.path.getsize(os.path.join(root, f)) / 1e9
                    except OSError: pass
            shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            print(f"  cleanup {d}: {e}", flush=True)
    return len(orphans), freed_gb


def free_disk_gb(path="/"):
    """Free space in GB on the filesystem holding a path."""
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / 1e9


def wait_for_disk(min_gb):
    """Block until at least min_gb of free space is available.

    A disk that fills mid-sweep produces truncated outputs and a cascade of failures
    that all look like build errors. Pausing is recoverable; the cascade is not.
    """
    while free_disk_gb() < min_gb:
        print(f"  ⚠ disk full ({free_disk_gb():.1f} GB free < {min_gb} GB) — pausing 30s", flush=True)
        time.sleep(30)


def load_params_table():
    """Load the precomputed parameter count of every configuration."""
    if not PARAMS_TABLE.exists():
        print(f"!! {PARAMS_TABLE} missing — no pre-filtering", flush=True)
        return {}
    return json.loads(PARAMS_TABLE.read_text())


def mark_too_big(meta_dir: Path, tag: str, params: int, threshold: int):
    """Write a metadata file marking this config as skipped for being too big."""
    p = meta_dir / f"{tag}.json"
    # Preserve any existing keys if metadata exists (e.g., previously logged errors)
    existing = {}
    if p.exists():
        try: existing = json.loads(p.read_text())
        except Exception: pass
    existing.update({
        "tag": tag,
        "too_big_skipped": True,
        "num_params": params,
        "skip_threshold": threshold,
        "note": f"skipped on dio: {params/1e6:.1f}M params > {threshold/1e6:.0f}M threshold (OOMs reliably at 15 GB RAM)",
    })
    p.write_text(json.dumps(existing, indent=2))


def run_one(config, out_dir: str, num_calib: int, timeout: int, min_disk_gb: float):
    """Build one configuration in a subprocess, with a timeout.

    Subprocess isolation is what makes the sweep survivable: a TensorFlow OOM kills
    the child only, and the parent records build_status="crashed" and continues. In
    process, the first oversized configuration would end the run.
    """
    family, depth, w, r = config
    tag = f"{family}_d{depth}_w{w}_r{r}"

    # Wait for disk to have room before starting
    while free_disk_gb() < min_disk_gb:
        time.sleep(10)

    cmd = [PYENV, str(BUILD_ONE),
           "--family", family, "--depth", str(depth),
           "--base_width", str(w), "--resolution", str(r),
           "--out_dir", out_dir,
           "--num_calib", str(num_calib)]
    t0 = time.time()
    try:
        r_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc = r_proc.returncode
        stderr_tail = r_proc.stderr[-300:] if r_proc.stderr else ""
    except subprocess.TimeoutExpired:
        rc = -1; stderr_tail = "TIMEOUT"
    return (tag, rc, round(time.time() - t0, 1), stderr_tail, free_disk_gb())


def should_skip_existing(tag: str, meta_dir: Path) -> str | None:
    """Decide whether a configuration can be skipped on a resumed sweep.

    Skips only genuine successes and deliberate too-big skips; anything else is
    retried. An earlier version skipped every tag whose metadata file existed, which
    meant failures were never retried, and a resumed sweep silently made no progress
    on exactly the configurations that needed it.
    """
    p = meta_dir / f"{tag}.json"
    if not p.exists(): return None
    try:
        m = json.loads(p.read_text())
    except Exception:
        return None
    if m.get("aeq_status") == "success": return "already_done"
    if m.get("too_big_skipped"): return "too_big_previously"
    return None


def main():
    """Run the sweep over all configurations, with a worker pool and resume support."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(SYNTH_ROOT / "outputs"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num_calib", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only-family", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--max_params", type=int, default=80_000_000,
                    help="Skip configs above this param count (default 80M)")
    ap.add_argument("--min_disk_gb", type=float, default=30.0,
                    help="Refuse to start a config if free disk < this")
    args = ap.parse_args()

    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    META = OUT / "metadata"; META.mkdir(exist_ok=True)
    (OUT / "tflite").mkdir(exist_ok=True)
    LOG = OUT / "sweep.log"

    print(f"=== regen_sweep_v2 startup {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    # 1) Cleanup orphan tmp dirs
    n_orphans, freed = cleanup_tmp()
    print(f"[cleanup] removed {n_orphans} orphan /tmp/regen_* dirs ({freed:.1f} GB freed)", flush=True)
    print(f"[cleanup] disk free after cleanup: {free_disk_gb():.1f} GB", flush=True)

    # 2) Load params table for pre-filtering
    params_table = load_params_table()
    print(f"[params] loaded {len(params_table)} entries from {PARAMS_TABLE}", flush=True)

    # 3) Build config list, apply filters
    configs = list(all_configs())
    if args.only_family:
        configs = [c for c in configs if c[0] == args.only_family]

    to_run = []
    n_skip_existing = n_skip_too_big = 0
    for c in configs:
        tag = f"{c[0]}_d{c[1]}_w{c[2]}_r{c[3]}"

        if args.skip_existing:
            reason = should_skip_existing(tag, META)
            if reason:
                n_skip_existing += 1
                continue

        # Params pre-filter — reliable OOM predictor
        params = params_table.get(tag, 0)
        if params > args.max_params:
            mark_too_big(META, tag, params, args.max_params)
            n_skip_too_big += 1
            continue

        to_run.append(c)

    print(f"[filter] already done or previously too_big: {n_skip_existing}", flush=True)
    print(f"[filter] skipped as too_big now (>{args.max_params/1e6:.0f}M params): {n_skip_too_big}", flush=True)
    print(f"[filter] to run: {len(to_run)}", flush=True)
    print(f"[config] workers={args.workers}, timeout={args.timeout}s, "
          f"max_params={args.max_params/1e6:.0f}M, min_disk_gb={args.min_disk_gb}", flush=True)

    if not to_run:
        print("Nothing to do.", flush=True); return

    t_all = time.time()
    n_ok = n_fail = 0

    with LOG.open("a") as flog:
        flog.write(f"\n=== sweep_v2 start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                   f"configs={len(to_run)} workers={args.workers} "
                   f"max_params={args.max_params/1e6:.0f}M ===\n")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, c, str(OUT), args.num_calib,
                                    args.timeout, args.min_disk_gb): c
                       for c in to_run}
            for i, fut in enumerate(as_completed(futures)):
                tag, rc, elapsed, stderr_tail, disk_free = fut.result()
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                if rc == 0: n_ok += 1
                else: n_fail += 1
                msg = (f"[{i+1}/{len(to_run)}] {tag:40s} {status:12s} "
                       f"{elapsed:6.1f}s  disk={disk_free:.1f}GB")
                print(msg, flush=True)
                flog.write(msg + "\n")
                if stderr_tail and rc != 0:
                    flog.write(f"  stderr: {stderr_tail}\n")
                flog.flush()

    dt = time.time() - t_all
    print(f"\nDone in {dt/60:.1f}min: {n_ok} OK, {n_fail} FAIL, "
          f"{n_skip_existing} skipped (existing), {n_skip_too_big} skipped (too big)",
          flush=True)

    summary = {"total": len(to_run), "ok": n_ok, "fail": n_fail,
               "skipped_existing": n_skip_existing, "skipped_too_big": n_skip_too_big,
               "elapsed_min": round(dt/60, 1), "timestamp": time.time(),
               "max_params": args.max_params}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
