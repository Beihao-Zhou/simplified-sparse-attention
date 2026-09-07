"""Turn recorded selection traces into locality tables.

Every (layer, KV-group) pair of a request is one independent trace, simulated
separately. Capacities are quoted as multiples of that trace's own
``k_max = max_t |S_t|`` so the required set always fits (see cache_sim); mean-
and p99-based capacities are reported in a separate table and never mixed into
the k_max one.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.cache_sim import ceiling, simulate, window  # noqa: E402

WINDOWS = [(0, 60), (60, 256), (256, 1000)]
D_HEAD = 128
BYTES_PER_ELEM = 2   # bf16 K + V handled by the factor 2 below


def load_trace(path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    lens = d["lens"].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    keys = d["keys"]
    series, recent, sizes = {}, {}, {}
    for i in range(len(lens)):
        k = (int(d["layer"][i]), int(d["b"][i]), int(d["g"][i]))
        series.setdefault(k, []).append(keys[off[i]:off[i + 1]].astype(np.int64))
        recent.setdefault(k, []).append(int(d["recent"][i]))
        sizes.setdefault(k, []).append(int(lens[i]))
    meta["prefill_tokens"] = int(meta["context_tokens"])
    return series, recent, sizes, meta


def _job(arg):
    key, steps, prefill, cap_defs, seeds = arg
    local = [(s >= prefill) for s in steps]
    kmax = max(len(s) for s in steps)
    sizes = np.array([len(s) for s in steps])
    out = []
    base = dict(layer=key[0], b=key[1], group=key[2], k_max=kmax,
                k_mean=float(sizes.mean()), k_p99=float(np.percentile(sizes, 99)),
                steps=len(steps), ceiling=ceiling(steps),
                total_requests=int(sizes.sum()))
    for k_def, k_val in cap_defs:
        for mult in (1, 2, 4):
            cap = int(round(k_val * mult))
            if cap < kmax:
                continue
            for pol in ("topk_only", "fifo", "lru", "belady"):
                r = simulate(steps, cap, pol, local_keys=local)
                out.append(_row(base, k_def, mult, cap, pol, r))
            rs = [simulate(steps, cap, "random", seed=s, local_keys=local) for s in seeds]
            out.append(_row(base, k_def, mult, cap, "random", _avg(rs)))
    return out


def _avg(rs):
    a = dict(rs[0])
    for f in ("hits", "misses", "cold_misses", "requests", "set_size",
              "carryover", "union_prev", "local", "local_miss"):
        a[f] = np.mean([r[f] for r in rs], axis=0)
    for f in ("total_hits", "total_misses", "total_cold", "hit_rate",
              "miss_rate", "cold_miss_rate", "capacity_miss_rate",
              "total_local", "total_local_miss"):
        a[f] = float(np.mean([r[f] for r in rs]))
    a["requests"] = np.asarray(a["requests"])
    a["hits"] = np.asarray(a["hits"])
    a["cold_misses"] = np.asarray(a["cold_misses"])
    a["set_size"] = np.asarray(a["set_size"])
    return a


def _row(base, k_def, mult, cap, pol, r):
    row = dict(base, k_def=k_def, mult=mult, capacity=cap, policy=pol,
               hit_rate=r["hit_rate"], miss_rate=r["miss_rate"],
               cold_miss_rate=r["cold_miss_rate"],
               capacity_miss_rate=r["capacity_miss_rate"],
               local_frac=r["total_local"] / max(1, r["total_requests"]),
               local_miss_frac=r["total_local_miss"] / max(1, r["total_requests"]))
    for lo, hi in WINDOWS:
        w = window(r, lo, hi)
        tag = f"w{lo}_{hi}"
        row[f"{tag}_hit"] = w["hit_rate"] if w else float("nan")
        row[f"{tag}_cold"] = w["cold_miss_rate"] if w else float("nan")
        row[f"{tag}_steps"] = w["n_steps"] if w else 0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default="analysis/results")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_rows, summary = [], []
    for tp in args.traces:
        name = Path(tp).stem
        series, recent, sizes, meta = load_trace(tp)
        prefill = meta["prefill_tokens"]
        kmaxes = {k: max(len(s) for s in v) for k, v in series.items()}
        allsz = np.concatenate([np.array([len(s) for s in v]) for v in series.values()])
        cap_defs_per = {
            k: [("kmax", kmaxes[k]),
                ("kmean", float(np.mean([len(s) for s in series[k]]))),
                ("kp99", float(np.percentile([len(s) for s in series[k]], 99)))]
            for k in series}
        jobs = [(k, series[k], prefill, cap_defs_per[k], list(range(args.seeds)))
                for k in series]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rows in ex.map(_job, jobs, chunksize=1):
                for r in rows:
                    r["trace"] = name
                    per_rows.append(r)

        # ---- trace-level descriptive stats ----
        car, jac = [], []
        for v in series.values():
            for t in range(1, len(v)):
                inter = len(np.intersect1d(v[t - 1], v[t], assume_unique=True))
                car.append(inter / max(1, len(v[t])))
                jac.append(inter / max(1, len(v[t - 1]) + len(v[t]) - inter))
        rec_frac = np.concatenate([np.array(recent[k]) / np.maximum(1, np.array(sizes[k]))
                                   for k in series])
        n_layers, n_groups = len({k[0] for k in series}), len({k[2] for k in series})
        summary.append(dict(
            trace=name, workload=meta["workload"], synthetic=meta["synthetic"],
            forced=meta["forced"], context_tokens=meta["context_tokens"],
            decode_steps=meta["decode_steps"], sparse_layers=n_layers,
            kv_groups=n_groups,
            k_mean=float(allsz.mean()), k_median=float(np.median(allsz)),
            k_p10=float(np.percentile(allsz, 10)), k_p90=float(np.percentile(allsz, 90)),
            k_p99=float(np.percentile(allsz, 99)), k_max=int(allsz.max()),
            k_min=int(allsz.min()),
            carryover=float(np.mean(car)), jaccard=float(np.mean(jac)),
            recent_frac=float(np.mean(rec_frac)),
            ceiling=float(np.mean([ceiling(v) for v in series.values()])),
            output_preview=meta.get("output_preview", "")[:120],
        ))
        print(f"[report] {name}: {len(series)} traces, "
              f"k mean {allsz.mean():.0f} max {allsz.max()}", flush=True)

    # ~39k rows x 40 columns; gzipped so the per-series detail behind the
    # median/p10/p90 and "worst series" numbers can be committed at ~1 MB.
    fields = sorted({k for r in per_rows for k in r})
    with gzip.open(outdir / "locality_per_trace.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace"] + [c for c in fields if c != "trace"])
        w.writeheader()
        w.writerows(per_rows)
    with open(outdir / "trace_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # ---- aggregate over (layer, group), weighted by requests ----
    agg = []
    for tr in {r["trace"] for r in per_rows}:
        for k_def in {r["k_def"] for r in per_rows}:
            for mult in (1, 2, 4):
                for pol in ("topk_only", "fifo", "lru", "random", "belady"):
                    sel = [r for r in per_rows if r["trace"] == tr and r["k_def"] == k_def
                           and r["mult"] == mult and r["policy"] == pol]
                    if not sel:
                        continue
                    wgt = np.array([r["total_requests"] for r in sel], float)
                    hr = np.array([r["hit_rate"] for r in sel])
                    row = dict(trace=tr, k_def=k_def, mult=mult, policy=pol,
                               n_series=len(sel),
                               capacity_mean=float(np.mean([r["capacity"] for r in sel])),
                               hit_rate=float((hr * wgt).sum() / wgt.sum()),
                               hit_median=float(np.median(hr)),
                               hit_p10=float(np.percentile(hr, 10)),
                               hit_p90=float(np.percentile(hr, 90)),
                               hit_min=float(hr.min()),
                               worst_series=str(sel[int(hr.argmin())]["layer"]) + "/"
                                            + str(sel[int(hr.argmin())]["group"]),
                               cold_miss_rate=float(
                                   (np.array([r["cold_miss_rate"] for r in sel]) * wgt).sum() / wgt.sum()),
                               ceiling=float((np.array([r["ceiling"] for r in sel]) * wgt).sum() / wgt.sum()))
                    for lo, hi in WINDOWS:
                        tag = f"w{lo}_{hi}"
                        v = np.array([r[f"{tag}_hit"] for r in sel])
                        s = np.array([r[f"{tag}_steps"] for r in sel], float)
                        row[f"{tag}_hit"] = float(np.nansum(v * s) / s.sum()) if s.sum() else float("nan")
                    # GPU bytes for the hot buffer: K+V, bf16, all sparse layers/groups
                    n_series = len(sel)
                    row["buffer_MiB"] = (row["capacity_mean"] * n_series * 2 * D_HEAD
                                         * BYTES_PER_ELEM / 2**20)
                    agg.append(row)
    with open(outdir / "locality_aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    print(f"[report] wrote {outdir}/locality_aggregate.csv "
          f"({len(agg)} rows), locality_per_trace.csv ({len(per_rows)} rows)")


if __name__ == "__main__":
    main()
