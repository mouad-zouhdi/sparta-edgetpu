#!/usr/bin/env python3
"""
bench_partitions.py — les 22 partitions des 8 accelerateurs, par checkpoint.

GRILLE. Une repartition des 8 accelerateurs est une facon de les decouper en
instances, chaque instance recevant un pipeline de N etages (donc le binaire
compile a N segments). A permutation pres, il y en a exactement p(8) = 22 :

    8   7+1   6+2   6+1+1   5+3   5+2+1   5+1+1+1   4+4   4+3+1   4+2+2
    4+2+1+1   4+1+1+1+1   3+3+2   3+3+1+1   3+2+2+1   3+2+1+1+1
    3+1+1+1+1+1   2+2+2+2   2+2+2+1+1   2+2+1+1+1+1   2+1+1+1+1+1+1
    1+1+1+1+1+1+1+1

Elles saturent toutes la carte. C'est une sur-ensemble strict de ce qui a ete
mesure jusqu'ici : scaling_sweepN + hybrid_sweepN couvrent les repartitions
homogenes M x N (dont celles qui laissent des accelerateurs oisifs, absentes
ici), hetero_sweepN n'en couvrait que quatre formes heuristiques.

L'ordre d'affectation des accelerateurs est canonique (0..7 dans l'ordre des
instances) : l'experience C' du 2026-08-16 a etabli que l'affectation n'a aucun
effet mesurable (F = 0,905, p = 0,94), les permutations sont donc hors grille.

PROTOCOLE, par point et par passe. Interpreteurs neufs a chaque passe.
  [C] premiere inference : une inference chainee par instance, toutes les
      instances lachees ensemble sur une barriere, chronometrage par etage.
      C'est la meme frontiere de mesure que coldstart_synth.py (autour du seul
      invoke), donc comparable aux campagnes de demarrage a froid.
  [W] chauffe.
  [L] latence : boucle fermee a une requete en vol par instance, toutes les
      instances actives simultanement. Donne la latence bout en bout de chaque
      sous-pipeline et le temps de service de chaque etage.
  [T] debit : pipeline a files bornees, sature. Donne le debit par instance et
      le temps de service de chaque etage sous charge.

Les deux phases partagent les memes interpreteurs : le chainage est manuel (un
fil par etage, files entre etages) et non le PipelinedModelRunner de pycoral,
pour trois raisons. Il donne les temps par etage, que le runner masque ; il
donne la latence sans le gonflement du runner, deja documente ; et il evite de
construire deux jeux d'interpreteurs par point.

Sortie : une ligne par (passe, checkpoint, repartition, instance).
"""
from __future__ import annotations
import argparse, csv, os, queue, random, sys, threading, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from bench_scaling import make_interp, make_input_buf
from wave_configs_v2 import discover_segments
from sweepN_configs import build_sweepN_points


# ---------------------------------------------------------------- grille -----
def partitions(n: int = 8):
    """Partitions de n, parts decroissantes. p(8) = 22."""
    def rec(rest, mx):
        if rest == 0:
            yield []
            return
        for k in range(min(rest, mx), 0, -1):
            for tail in rec(rest - k, k):
                yield [k] + tail
    return list(rec(n, n))


SHAPES = partitions(8)


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def join(xs, fmt="{:.4f}"):
    return "|".join(fmt.format(x) for x in xs)


# ------------------------------------------------------------- une instance --
class Instance:
    """Un sous-pipeline : N etages, un accelerateur et un interpreteur par etage."""

    def __init__(self, segments, tpu_ids):
        self.segments = list(segments)
        self.tpu_ids = list(tpu_ids)
        self.N = len(segments)
        self.interps, self.dels = [], []
        t0 = time.perf_counter_ns()
        for s, t in zip(self.segments, self.tpu_ids):
            it, d = make_interp(s, t)
            self.interps.append(it)
            self.dels.append(d)
        self.setup_ms = (time.perf_counter_ns() - t0) / 1e6
        # details mis en cache : les relire dans la boucle couterait une
        # construction de liste de dicts par inference
        self.ins = [it.get_input_details() for it in self.interps]
        self.outs = [it.get_output_details() for it in self.interps]
        self.first = self.ins[0][0]

    def chain(self, x, stage_ns=None):
        """Une inference chainee synchrone. Renvoie la duree bout en bout (ns)."""
        t0 = time.perf_counter_ns()
        it = self.interps[0]
        it.set_tensor(self.first["index"], x)
        ts = time.perf_counter_ns()
        it.invoke()
        tm = {d["name"]: it.get_tensor(d["index"]).copy() for d in self.outs[0]}
        if stage_ns is not None:
            stage_ns[0].append(time.perf_counter_ns() - ts)
        for k in range(1, self.N):
            itk = self.interps[k]
            ts = time.perf_counter_ns()
            for d in self.ins[k]:
                itk.set_tensor(d["index"], tm[d["name"]])
            itk.invoke()
            tm = {d["name"]: itk.get_tensor(d["index"]).copy() for d in self.outs[k]}
            if stage_ns is not None:
                stage_ns[k].append(time.perf_counter_ns() - ts)
        return time.perf_counter_ns() - t0

    def close(self):
        self.interps.clear(); self.dels.clear()
        self.ins.clear(); self.outs.clear()


# ------------------------------------------------------------- phases -------
def phase_cold(insts, buf, barrier):
    """[C] Premiere inference de chaque instance, toutes lachees ensemble."""
    res = [None] * len(insts)

    def w(j):
        inst = insts[j]
        st = [[] for _ in range(inst.N)]
        barrier.wait()
        e2e = inst.chain(buf[0:1], st)
        res[j] = (e2e / 1e6, [s[0] / 1e6 for s in st])

    th = [threading.Thread(target=w, args=(j,), daemon=True) for j in range(len(insts))]
    for t in th: t.start()
    for t in th: t.join()
    return res


def phase_latency(insts, buf, t_sec, warm, barrier):
    """[L] Boucle fermee, une requete en vol par instance, toutes concurrentes."""
    res = [None] * len(insts)
    nbuf = len(buf)

    def w(j):
        inst = insts[j]
        for i in range(warm):
            inst.chain(buf[i % nbuf:i % nbuf + 1])
        st = [[] for _ in range(inst.N)]
        e2e = []
        barrier.wait()
        dl = time.perf_counter() + t_sec
        r = 0
        while time.perf_counter() < dl:
            i = r % nbuf
            e2e.append(inst.chain(buf[i:i + 1], st))
            r += 1
        res[j] = (np.asarray(e2e, dtype=np.int64) / 1e6,
                  [np.asarray(s, dtype=np.int64) / 1e6 for s in st])

    th = [threading.Thread(target=w, args=(j,), daemon=True) for j in range(len(insts))]
    for t in th: t.start()
    for t in th: t.join()
    return res


def phase_throughput(insts, buf, t_sec, t_warm, qdepth, barrier):
    """[T] Pipeline sature, un fil et une file par etage."""
    res = [None] * len(insts)
    nbuf = len(buf)
    SENT = object()

    def w(j):
        inst = insts[j]
        N = inst.N
        qs = [queue.Queue(maxsize=qdepth) for _ in range(N + 1)]
        stage_ns = [[] for _ in range(N)]

        def stage(k):
            it = inst.interps[k]
            ins, outs = inst.ins[k], inst.outs[k]
            qi, qo = qs[k], qs[k + 1]
            acc = stage_ns[k]
            while True:
                item = qi.get()
                if item is SENT:
                    qo.put(SENT); return
                t0 = time.perf_counter_ns()
                for d in ins:
                    it.set_tensor(d["index"], item[d["name"]])
                it.invoke()
                o = {d["name"]: it.get_tensor(d["index"]).copy() for d in outs}
                acc.append(time.perf_counter_ns() - t0)
                qo.put(o)

        pops = []

        def consumer():
            qn = qs[N]
            while True:
                o = qn.get()
                if o is SENT: return
                pops.append(time.perf_counter_ns())

        th = [threading.Thread(target=stage, args=(k,), daemon=True) for k in range(N)]
        tc = threading.Thread(target=consumer, daemon=True)
        for t in th: t.start()
        tc.start()

        name = inst.first["name"]
        barrier.wait()
        t_start = time.perf_counter()
        dl = t_start + t_warm + t_sec
        i = 0
        while time.perf_counter() < dl:
            qs[0].put({name: buf[i % nbuf:i % nbuf + 1]})
            i += 1
        qs[0].put(SENT)
        for t in th: t.join()
        tc.join()

        p = np.asarray(pops, dtype=np.int64)
        # on jette la fenetre de chauffe en temps, pas en nombre : les instances
        # d'une repartition heterogene n'avancent pas au meme rythme
        if len(p) < 4:
            res[j] = (float("nan"), 0, [np.asarray([]) for _ in range(N)])
            return
        cut = p[0] + int(t_warm * 1e9)
        st = p[p >= cut]
        if len(st) < 3:
            st = p[len(p) // 2:]
        fps = (len(st) - 1) / ((st[-1] - st[0]) / 1e9)
        # meme fenetre pour les temps par etage
        keep = max(1, int(len(stage_ns[0]) * t_sec / (t_warm + t_sec)))
        res[j] = (fps, len(st),
                  [np.asarray(s[-keep:], dtype=np.int64) / 1e6 for s in stage_ns])

    th = [threading.Thread(target=w, args=(j,), daemon=True) for j in range(len(insts))]
    for t in th: t.start()
    for t in th: t.join()
    return res


# ----------------------------------------------------------------- point -----
def bench_point(seg_by_len, shape, t_lat, t_thr, warm_n, t_warm, qdepth, nbuf, seed):
    insts, tpu = [], 0
    try:
        for L in shape:
            insts.append(Instance(seg_by_len[L], list(range(tpu, tpu + L))))
            tpu += L
        first = insts[0].first
        buf = make_input_buf(first["shape"], first["dtype"], nbuf, seed=seed)

        bar = threading.Barrier(len(insts))
        cold = phase_cold(insts, buf, bar)
        lat = phase_latency(insts, buf, t_lat, warm_n, bar)
        thr = phase_throughput(insts, buf, t_thr, t_warm, qdepth, bar)

        tot = sum(r[0] for r in thr if r[0] == r[0])
        rows = []
        for j, L in enumerate(shape):
            c_e2e, c_st = cold[j]
            l_e2e, l_st = lat[j]
            f_fps, f_n, f_st = thr[j]
            rows.append(dict(
                inst_idx=j, n_stages=L,
                tpu_ids="|".join(map(str, insts[j].tpu_ids)),
                setup_ms=round(insts[j].setup_ms, 2),
                cold_e2e_ms=round(c_e2e, 4), cold_stage_ms=join(c_st),
                lat_n=len(l_e2e),
                lat_e2e_mean=round(float(np.mean(l_e2e)), 4),
                lat_e2e_median=round(float(np.median(l_e2e)), 4),
                lat_e2e_std=round(float(np.std(l_e2e)), 4),
                lat_e2e_p05=round(q(l_e2e, 5), 4), lat_e2e_p95=round(q(l_e2e, 95), 4),
                lat_e2e_p99=round(q(l_e2e, 99), 4),
                lat_e2e_min=round(float(np.min(l_e2e)), 4),
                lat_e2e_max=round(float(np.max(l_e2e)), 4),
                lat_stage_median=join([float(np.median(s)) for s in l_st]),
                lat_stage_p95=join([q(s, 95) for s in l_st]),
                thr_n=f_n, thr_fps_instance=round(f_fps, 3),
                thr_fps_total=round(tot, 3),
                thr_stage_median=join([float(np.median(s)) if len(s) else float("nan")
                                       for s in f_st]),
                thr_stage_p95=join([q(s, 95) for s in f_st]),
                thr_stage_n=join([len(s) for s in f_st], "{}"),
            ))
        return rows
    finally:
        for i in insts:
            i.close()


COLS = ["timestamp", "pass", "kind", "model", "pct", "fit_tpu", "shape",
        "n_instances", "tpu_used", "inst_idx", "n_stages", "tpu_ids",
        "setup_ms", "cold_e2e_ms", "cold_stage_ms",
        "lat_n", "lat_e2e_mean", "lat_e2e_median", "lat_e2e_std",
        "lat_e2e_p05", "lat_e2e_p95", "lat_e2e_p99", "lat_e2e_min", "lat_e2e_max",
        "lat_stage_median", "lat_stage_p95",
        "thr_n", "thr_fps_instance", "thr_fps_total",
        "thr_stage_median", "thr_stage_p95", "thr_stage_n",
        "t_lat_s", "t_thr_s", "qdepth"]


def append_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new: w.writeheader()
        for r in rows: w.writerow(r)
        f.flush(); os.fsync(f.fileno())


def done_keys(path: Path):
    if not path.exists(): return set()
    with path.open() as f:
        return {(int(r["pass"]), r["model"], int(r["pct"]), r["shape"])
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweepn-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--t-lat", type=float, default=4.0, help="s de mesure de latence")
    ap.add_argument("--t-thr", type=float, default=6.0, help="s de mesure de debit")
    ap.add_argument("--t-warm", type=float, default=1.0, help="s de chauffe phase T")
    ap.add_argument("--warm-n", type=int, default=8, help="inferences de chauffe phase L")
    ap.add_argument("--qdepth", type=int, default=4)
    ap.add_argument("--nbuf", type=int, default=16)
    ap.add_argument("--shapes", default="", help="sous-ensemble, ex '8,4+4,1+1+1+1+1+1+1+1'")
    ap.add_argument("--configs", default="", help="sous-ensemble 'model:pct,...'")
    ap.add_argument("--first-pass", type=int, default=1)
    ap.add_argument("--deadline-h", type=float, default=0.0, help="arret propre apres X h")
    a = ap.parse_args()

    pts = build_sweepN_points()
    by = {}
    for p in pts:
        if discover_segments(p, a.sweepn_root):
            by.setdefault((p.model, p.pct), {})[p.N] = list(p.segments)
    fit = {(p.model, p.pct): p.target_mb for p in pts}

    keys = sorted(k for k, v in by.items() if all(n in v for n in range(1, 9)))
    if a.configs:
        want = {tuple(s.split(":")[:2]) for s in a.configs.split(",")}
        keys = [k for k in keys if (k[0], str(k[1])) in want]
    shapes = SHAPES
    if a.shapes:
        want = {tuple(int(x) for x in s.split("+")) for s in a.shapes.split(",")}
        shapes = [s for s in SHAPES if tuple(s) in want]

    print(f"[grille] {len(keys)} checkpoints x {len(shapes)} repartitions "
          f"x {a.passes} passes = {len(keys)*len(shapes)*a.passes} points", flush=True)

    seen = done_keys(a.out_csv)
    t_start = time.perf_counter()
    dl = t_start + a.deadline_h * 3600 if a.deadline_h > 0 else float("inf")

    for pas in range(a.first_pass, a.first_pass + a.passes):
        todo = [(k, s) for k in keys for s in shapes
                if (pas, k[0], k[1], "+".join(map(str, s))) not in seen]
        # ordre melange, graine derivee de la passe : deux passes ne visitent
        # pas la grille dans le meme ordre, donc une derive thermique ou une
        # charge parasite ne se colle pas toujours aux memes points
        random.Random(1000 + pas).shuffle(todo)
        print(f"\n===== PASSE {pas} : {len(todo)} points restants =====", flush=True)
        for i, ((model, pct), shape) in enumerate(todo, 1):
            if time.perf_counter() > dl:
                print("[deadline] arret propre", flush=True); return
            lbl = "+".join(map(str, shape))
            t0 = time.perf_counter()
            try:
                rows = bench_point(by[(model, pct)], shape, a.t_lat, a.t_thr,
                                   a.warm_n, a.t_warm, a.qdepth, a.nbuf,
                                   seed=123 + pas)
            except Exception as e:
                print(f"[{pas}|{i}/{len(todo)}] {model} {pct}% {lbl} ECHEC "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            common = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), **{"pass": pas},
                          kind="sweepN", model=model, pct=pct,
                          fit_tpu=fit[(model, pct)], shape=lbl,
                          n_instances=len(shape), tpu_used=sum(shape),
                          t_lat_s=a.t_lat, t_thr_s=a.t_thr, qdepth=a.qdepth)
            append_rows(a.out_csv, [dict(common, **r) for r in rows])
            el = time.perf_counter() - t0
            print(f"[{pas}|{i}/{len(todo)}] {model} {pct}% {lbl:<16} "
                  f"{rows[0]['thr_fps_total']:8.1f} im/s  "
                  f"lat {max(r['lat_e2e_median'] for r in rows):7.2f} ms  "
                  f"froid {max(r['cold_e2e_ms'] for r in rows):7.1f} ms  "
                  f"({el:.1f}s)", flush=True)
    print(f"\ndone -> {a.out_csv}  ({(time.perf_counter()-t_start)/3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
