#!/usr/bin/env python3
"""
bench_partitions_fixedn.py — les 22 repartitions des 8 accelerateurs, a nombre
d'inferences fixe, sur les 43 checkpoints des trois campagnes multi-accelerateur.

DIFFERENCE AVEC bench_partitions.py. Celui-ci fixait des fenetres de mesure en
DUREE (4,5 s en latence, 6,5 s en debit). Consequence : le nombre d'inferences
chronometrees variait d'un facteur 20 d'un point a l'autre, mediane 107 en
latence, si bien que les percentiles hauts etaient estimes sur une poignee de
tirages et n'etaient comparables ni entre points ni entre repartitions.

Ici le nombre est fixe et identique partout :

    1000 inferences chronometrees en latence
    1000 inferences chronometrees en debit
     200 premieres inferences, chacune sur des interpreteurs neufs

Le gain porte sur les queues de distribution, pas sur les moyennes : la
dispersion de la moyenne d'un point est dominee par la derive entre sessions
(0,20 %) et non par l'echantillonnage a l'interieur d'une mesure (0,57 % par
inference, donc 0,018 % sur une moyenne de mille). Repeter les passes reste donc
la seule facon d'obtenir une barre d'erreur honnete.

CHARGE MAINTENUE. Un nombre fixe cree un piege que la fenetre en duree n'avait
pas : dans une repartition heterogene, l'instance courte atteindrait ses mille
inferences bien avant l'instance longue, et la fin de mesure de cette derniere
se ferait sur une carte a moitie libre, donc dans un regime de contention qui
n'est pas celui qu'on pretend mesurer. Toutes les instances continuent donc a
tourner jusqu'a ce que la plus lente ait fini ; seules les mille premieres de
chacune sont retenues. Cela ne coute rien, la duree de phase etant de toute
facon celle de l'instance la plus lente.

EFFET DE BORD BIENVENU. Compter les sorties au lieu de fermer une fenetre en
temps supprime le seul defaut releve par l'audit du protocole precedent : la
vidange des files apres l'arret du producteur etait comptee dans la fenetre, ce
qui la faisait deborder de 2,2 % en mediane. Ici la vidange tombe hors du compte.

DEBIT ET DISTRIBUTION. Le debit n'est pas une grandeur par inference : une passe
n'en produisait qu'une valeur par instance, dont on ne peut tirer aucune
distribution. Deux echantillonnages sont donc enregistres, tous deux gratuits
puisque les horodatages de sortie sont deja collectes : les 999 intervalles
entre sorties consecutives, qui donnent la gigue, et le debit par bloc de cent
inferences, qui donne moyenne, ecart-type et percentiles au sens usuel.

TABLEAUX BRUTS. Les series par inference sont ecrites telles quelles en .npz, un
fichier par (passe, checkpoint, repartition). Sans elles, les percentiles ne
peuvent pas etre remis en commun entre passes, ce qui etait la limite des
resumes de la campagne precedente.

Sortie : une ligne par (passe, checkpoint, repartition, instance), plus un .npz
par (passe, checkpoint, repartition).
"""
from __future__ import annotations
import argparse, csv, os, queue, random, sys, threading, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from bench_scaling import make_interp, make_input_buf
from multi_tpu_corpus import build_corpus, partitions, ALL_N

SHAPES = partitions(8)


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def join(xs, fmt="{:.4f}"):
    return "|".join(fmt.format(x) for x in xs)


def stats(a, prefix):
    """Le jeu de statistiques demande, sur une serie par inference."""
    if not len(a):
        return {f"{prefix}_{k}": float("nan") for k in
                ("mean", "median", "std", "p05", "p25", "p75", "p95", "p99",
                 "min", "max")}
    return {
        f"{prefix}_mean": round(float(np.mean(a)), 4),
        f"{prefix}_median": round(float(np.median(a)), 4),
        f"{prefix}_std": round(float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, 4),
        f"{prefix}_p05": round(q(a, 5), 4), f"{prefix}_p25": round(q(a, 25), 4),
        f"{prefix}_p75": round(q(a, 75), 4), f"{prefix}_p95": round(q(a, 95), 4),
        f"{prefix}_p99": round(q(a, 99), 4),
        f"{prefix}_min": round(float(np.min(a)), 4),
        f"{prefix}_max": round(float(np.max(a)), 4),
    }


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
        self.ins = [it.get_input_details() for it in self.interps]
        self.outs = [it.get_output_details() for it in self.interps]
        self.first = self.ins[0][0]
        # Un etage ne consomme pas forcement la seule sortie de son predecesseur :
        # quand le compilateur coupe a l'interieur d'un module a branches, un
        # etage peut reclamer un tenseur produit deux etages plus tot. Le relais
        # est donc cumulatif. `keep[k]` est l'ensemble des noms encore reclames a
        # partir de l'etage k, ce qui permet de liberer les autres tout de suite
        # et de borner la memoire des tenseurs en vol.
        self.keep = []
        for k in range(self.N):
            self.keep.append({d["name"] for kk in range(k, self.N)
                              for d in self.ins[kk]})

    def _relay(self, tm, k):
        """Ajoute les sorties de l'etage k et jette ce que plus personne ne lit."""
        for d in self.outs[k]:
            tm[d["name"]] = self.interps[k].get_tensor(d["index"]).copy()
        nxt = self.keep[k + 1] if k + 1 < self.N else set()
        for n in [n for n in tm if n not in nxt]:
            del tm[n]
        return tm

    def chain(self, x, stage_ns=None):
        """Une inference chainee synchrone. Renvoie la duree bout en bout (ns)."""
        t0 = time.perf_counter_ns()
        it = self.interps[0]
        it.set_tensor(self.first["index"], x)
        ts = time.perf_counter_ns()
        it.invoke()
        tm = self._relay({}, 0)
        if stage_ns is not None:
            stage_ns[0].append(time.perf_counter_ns() - ts)
        for k in range(1, self.N):
            itk = self.interps[k]
            ts = time.perf_counter_ns()
            for d in self.ins[k]:
                itk.set_tensor(d["index"], tm[d["name"]])
            itk.invoke()
            tm = self._relay(tm, k)
            if stage_ns is not None:
                stage_ns[k].append(time.perf_counter_ns() - ts)
        return time.perf_counter_ns() - t0

    def close(self):
        self.interps.clear(); self.dels.clear()
        self.ins.clear(); self.outs.clear()


def run_workers(fn, n):
    """Lance un fil par instance et fait remonter la premiere exception.

    Sans cela, un fil qui meurt laisse simplement un resultat a None et l'erreur
    reapparait bien plus loin sous une forme qui ne dit rien de sa cause, en
    plus de casser la barriere des autres fils.
    """
    err = []

    def wrap(j):
        try:
            fn(j)
        except BaseException as e:            # noqa: BLE001
            err.append((j, e))

    th = [threading.Thread(target=wrap, args=(j,), daemon=True) for j in range(n)]
    for t in th: t.start()
    for t in th: t.join()
    if err:
        j, e = err[0]
        raise RuntimeError(f"instance {j}: {type(e).__name__}: {e}") from e


class Countdown:
    """Barriere de fin : l'evenement se leve quand toutes les instances ont fini."""

    def __init__(self, n):
        self.n = n
        self.lock = threading.Lock()
        self.evt = threading.Event()

    def done_one(self):
        with self.lock:
            self.n -= 1
            if self.n <= 0:
                self.evt.set()


# ------------------------------------------------------------------ phases ---
def phase_cold(seg_by_len, shape, buf, reps):
    """[C] Premieres inferences, interpreteurs neufs a chaque repetition.

    Toutes les instances sont reconstruites puis lachees ensemble sur une
    barriere, exactement comme dans les campagnes de demarrage a froid : le
    chronometrage entoure le seul enchainement des invoke, pas la construction.
    """
    ni = len(shape)
    e2e = [[] for _ in range(ni)]
    setup = [[] for _ in range(ni)]
    stage = [[[] for _ in range(L)] for L in shape]
    for _ in range(reps):
        insts, tpu = [], 0
        try:
            for L in shape:
                insts.append(Instance(seg_by_len[L], list(range(tpu, tpu + L))))
                tpu += L
            bar = threading.Barrier(ni)
            res = [None] * ni

            def w(j):
                st = [[] for _ in range(insts[j].N)]
                bar.wait()
                res[j] = (insts[j].chain(buf[0:1], st) / 1e6,
                          [s[0] / 1e6 for s in st])

            run_workers(w, ni)
            for j in range(ni):
                e2e[j].append(res[j][0])
                setup[j].append(insts[j].setup_ms)
                for k, v in enumerate(res[j][1]):
                    stage[j][k].append(v)
        finally:
            for i in insts:
                i.close()
    return ([np.asarray(x) for x in e2e],
            [np.asarray(x) for x in setup],
            [[np.asarray(s) for s in inst] for inst in stage])


def phase_latency(insts, buf, n_target, warm, barrier):
    """[L] Boucle fermee, une requete en vol par instance, toutes concurrentes.

    Chaque instance chronometre ses `n_target` premieres inferences puis
    continue a tourner, sans mesurer, jusqu'a ce que la derniere ait fini : la
    carte reste chargee pendant toute la mesure de chacune.
    """
    ni = len(insts)
    res = [None] * ni
    cd = Countdown(ni)
    nbuf = len(buf)

    def w(j):
        inst = insts[j]
        for i in range(warm):
            inst.chain(buf[i % nbuf:i % nbuf + 1])
        st = [[] for _ in range(inst.N)]
        e2e = []
        barrier.wait()
        r = 0
        while True:
            i = r % nbuf
            if r < n_target:
                e2e.append(inst.chain(buf[i:i + 1], st))
                r += 1
                if r == n_target:
                    res[j] = (np.asarray(e2e, dtype=np.int64) / 1e6,
                              [np.asarray(s, dtype=np.int64) / 1e6 for s in st])
                    cd.done_one()
            else:
                if cd.evt.is_set():
                    return
                inst.chain(buf[i:i + 1])
                r += 1

    run_workers(w, ni)
    return res


def phase_throughput(insts, buf, n_target, warm_n, qdepth, barrier):
    """[T] Pipeline sature, un fil et une file par etage.

    Le consommateur compte les sorties : `warm_n` jetees puis `n_target`
    retenues. Le producteur continue a alimenter tant que toutes les instances
    n'ont pas fini, pour la meme raison que ci-dessus. La vidange des files apres
    la sentinelle tombe hors du compte, ce qui supprime le biais de fenetre du
    protocole precedent.
    """
    ni = len(insts)
    res = [None] * ni
    cd = Countdown(ni)
    nbuf = len(buf)
    SENT = object()

    def w(j):
        inst = insts[j]
        N = inst.N
        qs = [queue.Queue(maxsize=qdepth) for _ in range(N + 1)]
        stage_ns = [[] for _ in range(N)]
        need = warm_n + n_target

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
                item = inst._relay(item, k)
                acc.append(time.perf_counter_ns() - t0)
                qo.put(item)

        pops = []
        reached = threading.Event()

        def consumer():
            qn = qs[N]
            while True:
                o = qn.get()
                if o is SENT:
                    return
                pops.append(time.perf_counter_ns())
                if len(pops) == need and not reached.is_set():
                    reached.set()
                    cd.done_one()

        th = [threading.Thread(target=stage, args=(k,), daemon=True)
              for k in range(N)]
        tc = threading.Thread(target=consumer, daemon=True)
        for t in th: t.start()
        tc.start()

        name = inst.first["name"]
        barrier.wait()
        i = 0
        while not cd.evt.is_set():
            qs[0].put({name: buf[i % nbuf:i % nbuf + 1]})
            i += 1
        qs[0].put(SENT)
        for t in th: t.join()
        tc.join()

        p = np.asarray(pops[warm_n:warm_n + n_target], dtype=np.int64)
        if len(p) < 2:
            res[j] = (float("nan"), np.asarray([]), np.asarray([]),
                      [np.asarray([]) for _ in range(N)])
            return
        gaps = np.diff(p) / 1e6                       # ms entre sorties
        fps = (len(p) - 1) / ((p[-1] - p[0]) / 1e9)
        # debit par bloc : dix blocs de taille egale, pour disposer d'une
        # moyenne, d'un ecart-type et de percentiles du debit, grandeur dont une
        # mesure ne produit sinon qu'une seule valeur par instance
        w = len(p) // 10
        blocks = (np.asarray([(w - 1) / ((p[b * w + w - 1] - p[b * w]) / 1e9)
                              for b in range(10)])
                  if w >= 2 else np.asarray([]))
        win = slice(warm_n, warm_n + n_target)
        res[j] = (fps, gaps, blocks,
                  [np.asarray(s[win], dtype=np.int64) / 1e6 for s in stage_ns])

    run_workers(w, ni)
    return res


# ------------------------------------------------------------------ point ----
def probe_input(seg):
    """Details d'entree du premier etage, pour dimensionner le tampon."""
    it, d = make_interp(seg, 0)
    det = it.get_input_details()[0]
    shape, dtype = det["shape"], det["dtype"]
    del it, d
    return shape, dtype


def bench_point(seg_by_len, shape, n_lat, n_thr, n_cold, warm_lat, warm_thr,
                qdepth, nbuf, seed):
    sh, dt = probe_input(seg_by_len[shape[0]][0])
    buf = make_input_buf(sh, dt, nbuf, seed=seed)

    tphase = {}
    t0 = time.perf_counter()
    cold_e2e, cold_setup, cold_stage = phase_cold(seg_by_len, shape, buf, n_cold)
    tphase["cold"] = time.perf_counter() - t0

    insts, tpu = [], 0
    try:
        for L in shape:
            insts.append(Instance(seg_by_len[L], list(range(tpu, tpu + L))))
            tpu += L
        bar = threading.Barrier(len(insts))
        t0 = time.perf_counter()
        lat = phase_latency(insts, buf, n_lat, warm_lat, bar)
        tphase["lat"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        thr = phase_throughput(insts, buf, n_thr, warm_thr, qdepth, bar)
        tphase["thr"] = time.perf_counter() - t0
    finally:
        for i in insts:
            i.close()

    tot = sum(r[0] for r in thr if r[0] == r[0])
    rows, raw = [], {}
    for j, L in enumerate(shape):
        l_e2e, l_st = lat[j]
        f_fps, f_gaps, f_blocks, f_st = thr[j]
        c_e2e, c_setup, c_st = cold_e2e[j], cold_setup[j], cold_stage[j]
        r = dict(inst_idx=j, n_stages=L,
                 tpu_ids="|".join(map(str, range(sum(shape[:j]), sum(shape[:j]) + L))),
                 setup_ms_mean=round(float(np.mean(c_setup)), 3),
                 lat_n=len(l_e2e), thr_n=len(f_gaps) + 1, cold_n=len(c_e2e),
                 thr_fps_instance=round(float(f_fps), 3),
                 thr_fps_total=round(tot, 3),
                 lat_stage_median=join([float(np.median(s)) for s in l_st]),
                 lat_stage_p95=join([q(s, 95) for s in l_st]),
                 thr_stage_median=join([float(np.median(s)) if len(s) else float("nan")
                                        for s in f_st]),
                 thr_stage_p95=join([q(s, 95) for s in f_st]),
                 cold_stage_median=join([float(np.median(s)) for s in c_st]))
        r.update(stats(l_e2e, "lat_ms"))
        r.update(stats(c_e2e, "cold_ms"))
        r.update(stats(f_gaps, "gap_ms"))
        r.update(stats(f_blocks, "blk_fps"))
        rows.append(r)
        raw[f"lat_{j}"] = l_e2e.astype(np.float32)
        raw[f"cold_{j}"] = c_e2e.astype(np.float32)
        raw[f"gap_{j}"] = f_gaps.astype(np.float32)
        raw[f"blk_{j}"] = f_blocks.astype(np.float32)
    return rows, raw, tphase


COLS = (["timestamp", "pass", "corpus", "model", "pct", "fit_tpu", "shape",
         "n_instances", "tpu_used", "inst_idx", "n_stages", "tpu_ids",
         "setup_ms_mean", "lat_n", "thr_n", "cold_n",
         "thr_fps_instance", "thr_fps_total"]
        + [f"lat_ms_{k}" for k in ("mean", "median", "std", "p05", "p25", "p75",
                                   "p95", "p99", "min", "max")]
        + [f"cold_ms_{k}" for k in ("mean", "median", "std", "p05", "p25", "p75",
                                    "p95", "p99", "min", "max")]
        + [f"gap_ms_{k}" for k in ("mean", "median", "std", "p05", "p25", "p75",
                                   "p95", "p99", "min", "max")]
        + [f"blk_fps_{k}" for k in ("mean", "median", "std", "p05", "p25", "p75",
                                    "p95", "p99", "min", "max")]
        + ["lat_stage_median", "lat_stage_p95", "thr_stage_median",
           "thr_stage_p95", "cold_stage_median",
           "n_lat", "n_thr", "n_cold", "qdepth"])


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
        return {(int(r["pass"]), r["corpus"], r["model"], int(r["pct"]), r["shape"])
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="racine contenant tflite_sweepn/, tflite_pruned_v2/, "
                         "tflite_baseline_v2/")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="dossier des .npz par point ; omettre pour ne rien ecrire")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--first-pass", type=int, default=1)
    ap.add_argument("--n-lat", type=int, default=1000)
    ap.add_argument("--n-thr", type=int, default=1000)
    ap.add_argument("--n-cold", type=int, default=200)
    ap.add_argument("--warm-lat", type=int, default=20)
    ap.add_argument("--warm-thr", type=int, default=50)
    ap.add_argument("--qdepth", type=int, default=4)
    ap.add_argument("--nbuf", type=int, default=16)
    ap.add_argument("--corpora", default="sweepN,wave,baseline")
    ap.add_argument("--shapes", default="", help="sous-ensemble, ex '8,4+4'")
    ap.add_argument("--configs", default="", help="sous-ensemble 'corpus:model:pct,...'")
    ap.add_argument("--max-points", type=int, default=0, help="0 = pas de limite")
    ap.add_argument("--deadline-h", type=float, default=0.0)
    a = ap.parse_args()

    cps = [c for c in build_corpus(a.root, tuple(a.corpora.split(",")))
           if c.complete()]
    if a.configs:
        want = {tuple(s.split(":")) for s in a.configs.split(",")}
        cps = [c for c in cps if (c.corpus, c.model, str(c.pct)) in want]
    shapes = SHAPES
    if a.shapes:
        want = {tuple(int(x) for x in s.split("+")) for s in a.shapes.split(",")}
        shapes = [s for s in SHAPES if tuple(s) in want]

    print(f"[grille] {len(cps)} checkpoints x {len(shapes)} repartitions "
          f"x {a.passes} passes = {len(cps)*len(shapes)*a.passes} points",
          flush=True)
    print(f"[compte] latence {a.n_lat}, debit {a.n_thr}, froid {a.n_cold}",
          flush=True)

    seen = done_keys(a.out_csv)
    t_start = time.perf_counter()
    dl = t_start + a.deadline_h * 3600 if a.deadline_h > 0 else float("inf")
    ndone = 0

    for pas in range(a.first_pass, a.first_pass + a.passes):
        todo = [(c, s) for c in cps for s in shapes
                if (pas, c.corpus, c.model, c.pct, "+".join(map(str, s))) not in seen]
        # ordre melange, graine derivee de la passe : deux passes ne visitent pas
        # la grille dans le meme ordre, donc une derive thermique ou une charge
        # parasite ne se colle pas toujours aux memes points
        random.Random(1000 + pas).shuffle(todo)
        print(f"\n===== PASSE {pas} : {len(todo)} points restants =====", flush=True)
        for i, (c, shape) in enumerate(todo, 1):
            if time.perf_counter() > dl:
                print("[deadline] arret propre", flush=True); return
            if a.max_points and ndone >= a.max_points:
                print("[max-points] arret", flush=True); return
            lbl = "+".join(map(str, shape))
            t0 = time.perf_counter()
            try:
                rows, raw, tph = bench_point(c.seg, shape, a.n_lat, a.n_thr,
                                             a.n_cold, a.warm_lat, a.warm_thr,
                                             a.qdepth, a.nbuf, seed=123 + pas)
            except Exception as e:
                print(f"[{pas}|{i}/{len(todo)}] {c.tag} {lbl} ECHEC "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            common = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          **{"pass": pas}, corpus=c.corpus, model=c.model,
                          pct=c.pct, fit_tpu=c.fit_tpu if c.fit_tpu is not None else "",
                          shape=lbl, n_instances=len(shape), tpu_used=sum(shape),
                          n_lat=a.n_lat, n_thr=a.n_thr, n_cold=a.n_cold,
                          qdepth=a.qdepth)
            append_rows(a.out_csv, [dict(common, **r) for r in rows])
            if a.raw_dir:
                d = a.raw_dir / f"pass{pas}"
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(d / f"{c.tag}__{lbl}.npz", **raw)
            ndone += 1
            el = time.perf_counter() - t0
            print(f"[{pas}|{i}/{len(todo)}] {c.tag:<44} {lbl:<16} "
                  f"{rows[0]['thr_fps_total']:8.1f} im/s  "
                  f"lat {max(r['lat_ms_median'] for r in rows):7.2f} ms  "
                  f"froid {max(r['cold_ms_median'] for r in rows):7.1f} ms  "
                  f"({el:.1f}s = {tph['cold']:.0f}f+{tph['lat']:.0f}l+"
                  f"{tph['thr']:.0f}d)", flush=True)
    print(f"\ndone -> {a.out_csv}  ({(time.perf_counter()-t_start)/3600:.2f} h)",
          flush=True)


if __name__ == "__main__":
    main()
