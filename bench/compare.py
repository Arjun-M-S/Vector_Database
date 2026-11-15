#!/usr/bin/env python3
"""FAISS (IndexHNSWFlat) and hnswlib on the same data and methodology as bench/bench_hnsw.cpp.

Run one engine per process so peak_mem_mb (this process's VmHWM) belongs to that engine alone:
    .venv/bin/python bench/compare.py --engine faiss   --out results/faiss_hnsw.csv
    .venv/bin/python bench/compare.py --engine hnswlib --out results/hnswlib.csv

Timing mirrors the C++ harness: one search call per query, each timed with perf_counter_ns;
QPS = queries / wall time of the whole loop. Unlike C++, every call also pays Python overhead
(interpreter loop, binding argument conversion, allocating the result arrays). The script prints
an estimate of that fixed per-call cost so you can see its share of the latency.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # must be set before faiss/numpy start OpenMP

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HEADER = ["engine", "M", "ef_construction", "ef_search", "k", "recall", "qps_median", "qps_min",
          "qps_max", "p50_us", "p99_us", "build_s", "threads", "peak_mem_mb"]


def load_vecs(path, dtype):
    """TEXMEX .fvecs/.ivecs -> (n, d) array. Reads in chunks so memory peaks at about one copy
    of the data, like the C++ loader; a whole-file read + copy would double peak_mem_mb."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        d = int(np.fromfile(f, np.int32, 1)[0])
        if size % (4 * (d + 1)):
            raise ValueError(f"{path}: size is not a whole number of records")
        n = size // (4 * (d + 1))
        out = np.empty((n, d), dtype)
        f.seek(0)
        for start in range(0, n, 100_000):
            rows = np.fromfile(f, dtype, min(100_000, n - start) * (d + 1)).reshape(-1, d + 1)
            if (rows[:, 0].view(np.int32) != d).any():
                raise ValueError(f"{path}: rows have different dimensions")
            out[start:start + len(rows)] = rows[:, 1:]
    return out


def peak_mem_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024  # kB -> MiB
    return -1.0


def percentile(sorted_vals, p):
    """Nearest rank, same as the C++ harness: an observed sample, no interpolation."""
    return sorted_vals[max(math.ceil(p * len(sorted_vals)), 1) - 1]


def time_queries(queries, repeats, search):
    qps, p50, p99 = [], [], []
    latency_us = [0.0] * len(queries)
    for _ in range(repeats):
        loop_start = time.perf_counter_ns()
        for i, q in enumerate(queries):
            t0 = time.perf_counter_ns()
            search(q)
            latency_us[i] = (time.perf_counter_ns() - t0) / 1e3
        qps.append(len(queries) * 1e9 / (time.perf_counter_ns() - loop_start))
        s = sorted(latency_us)
        p50.append(percentile(s, 0.50))
        p99.append(percentile(s, 0.99))
    return statistics.median(qps), min(qps), max(qps), statistics.median(p50), statistics.median(p99)


def build(engine, base, args):
    """Builds a single-threaded index on base; returns (set_ef, search(q) -> top-k ids)."""
    n, d = base.shape
    if engine == "faiss":
        import faiss
        faiss.omp_set_num_threads(1)
        index = faiss.IndexHNSWFlat(d, args.M)
        index.hnsw.efConstruction = args.ef_construction
        index.add(base)

        def set_ef(ef):
            index.hnsw.efSearch = ef
        return set_ef, lambda q: index.search(q, args.k)[1][0]

    import hnswlib
    index = hnswlib.Index(space="l2", dim=d)
    index.init_index(max_elements=n, M=args.M, ef_construction=args.ef_construction,
                     random_seed=args.seed)
    index.set_num_threads(1)
    index.add_items(base, np.arange(n), num_threads=1)
    return index.set_ef, lambda q: index.knn_query(q, k=args.k, num_threads=1)[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=["faiss", "hnswlib"])
    ap.add_argument("--data-dir", default="data/sift")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42, help="hnswlib only; faiss has no seed parameter")
    ap.add_argument("--ef-search", default="10,16,32,64,128,256,512")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out")
    args = ap.parse_args()
    engine = {"faiss": "faiss_hnsw", "hnswlib": "hnswlib"}[args.engine]
    out = Path(args.out or f"results/{engine}.csv")
    k = args.k

    # Loading is not timed.
    data_dir = Path(args.data_dir)
    name = data_dir.name
    base = load_vecs(data_dir / f"{name}_base.fvecs", np.float32)
    queries = load_vecs(data_dir / f"{name}_query.fvecs", np.float32)
    gt = load_vecs(data_dir / f"{name}_groundtruth.ivecs", np.int32)
    if queries.shape[1] != base.shape[1] or len(gt) != len(queries) or gt.shape[1] < k:
        sys.exit("dataset files disagree (dimension, query count, or ground truth < k)")
    n, d = base.shape
    qs = [queries[i:i + 1] for i in range(len(queries))]  # pre-sliced (1, d) views, so slicing isn't timed
    print(f"data: {name} base {n}x{d}, {len(qs)} queries, threads 1, cpus {','.join(map(str, sorted(os.sched_getaffinity(0))))}, python {sys.version.split()[0]}")

    if args.engine == "faiss":
        print("note: faiss has no seed parameter; --seed is ignored", file=sys.stderr)
    # A search on a k-vector index does almost no work, so its latency is roughly the fixed
    # per-call cost the C++ harness doesn't pay (binding conversion, result arrays, Python call).
    _, tiny_search = build(args.engine, base[:k], args)
    overhead = time_queries(qs, 1, tiny_search)
    print(f"python + binding per-call overhead (search on a {k}-vector index): p50 {overhead[3]:.2f} us")

    t0 = time.perf_counter()
    set_ef, search = build(args.engine, base, args)
    build_s = time.perf_counter() - t0
    mem_mb = peak_mem_mb()
    print(f"build: {build_s:.2f} s, peak RSS {mem_mb:.1f} MiB")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        print(",".join(HEADER))
        for ef in (int(x) for x in args.ef_search.split(",")):
            set_ef(ef)
            hits = sum(len(set(search(q).tolist()) & set(gt[i, :k].tolist())) for i, q in enumerate(qs))
            recall = hits / (len(qs) * k)
            for q in qs[:args.warmup]:
                search(q)
            qps_med, qps_min, qps_max, p50, p99 = time_queries(qs, args.repeats, search)
            row = [engine, args.M, args.ef_construction, ef, k, f"{recall:.4f}", f"{qps_med:.1f}",
                   f"{qps_min:.1f}", f"{qps_max:.1f}", f"{p50:.2f}", f"{p99:.2f}", f"{build_s:.2f}", 1,
                   f"{mem_mb:.1f}"]
            w.writerow(row)
            f.flush()
            print(",".join(map(str, row)))


if __name__ == "__main__":
    main()
