#!/usr/bin/env python3
"""Recall@k (x) vs QPS (y, log) from every CSV in a results dir, one line per engine.

    .venv/bin/python bench/plot.py [results_dir]   ->  <results_dir>/recall_qps.png
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed color, marker and label nudge (points) per engine, so an engine looks the same in every
# plot; the nudges keep the FAISS and hnswlib labels apart where their lines nearly overlap.
STYLE = {
    "vecdb_hnsw": ("#2a78d6", "o", 0),
    "faiss_hnsw": ("#eb6834", "s", 7),
    "hnswlib": ("#1baf7a", "^", -7),
    "vecdb_bruteforce": ("#eda100", "*", 0),
}

results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
rows = defaultdict(list)
source = {}
for path in sorted(results.glob("*.csv")):
    with open(path) as f:
        for r in csv.DictReader(f):
            # A stale CSV from another run would otherwise be merged silently into the same line.
            if source.setdefault(r["engine"], path) != path:
                sys.exit(f"engine {r['engine']} appears in both {source[r['engine']]} and {path}")
            rows[r["engine"]].append(r)
if not rows:
    sys.exit(f"no CSVs in {results}")

configs = {(r["M"], r["ef_construction"], r["k"]) for rs in rows.values() for r in rs if r["M"]}
if len(configs) > 1:
    print(f"WARNING: engines were run with different (M, efConstruction, k): {sorted(configs)}",
          file=sys.stderr)
M, efc, k = sorted(configs)[0] if configs else ("?", "?", next(iter(rows.values()))[0]["k"])

fig, ax = plt.subplots(figsize=(8, 5.5))
for engine, rs in sorted(rows.items()):
    rs.sort(key=lambda r: int(r["ef_search"] or 0))
    x = [float(r["recall"]) for r in rs]
    y = [float(r["qps_median"]) for r in rs]
    color, marker, dy = STYLE.get(engine, ("#6250d6", "D", 0))
    ax.plot(x, y, color=color, marker=marker, markersize=7 if marker != "*" else 12, linewidth=2,
            label=engine)
    # Engine name at the low-recall end, where lines are apart; they converge near recall 1.
    ax.annotate(engine, (x[0], y[0]), xytext=(-8, dy), textcoords="offset points", ha="right",
                va="center", fontsize=9, color="#333333")
    if engine == "vecdb_hnsw":  # every engine runs the same sweep, so label it once
        for r, xi, yi in zip(rs, x, y):
            ax.annotate(r["ef_search"], (xi, yi), xytext=(-6, -10), textcoords="offset points",
                        ha="right", fontsize=7, color="#666666")

ax.set_yscale("log")
ax.set_xlabel(f"recall@{k}")
ax.set_ylabel("queries per second (median of repeats, log scale)")
ax.set_title(f"Recall vs QPS — M={M}, efConstruction={efc}, k={k}, 1 thread  (grey labels: efSearch)",
             fontsize=10)
ax.grid(True, which="both", color="#dddddd", linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.margins(x=0.12)
ax.legend(frameon=False, loc="lower left")
out = results / "recall_qps.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
