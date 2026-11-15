#!/usr/bin/env bash
# Records the machine, toolchain and build flags behind a benchmark run in results/env.txt.
#   scripts/env.sh   records both Python environments: .venv (faiss) and .conda-hnswlib (hnswlib)
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/results/env.txt"
mkdir -p "$root/results"

{
  echo "date: $(date -Iseconds)"
  echo "git: $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet || echo ' (uncommitted changes)')"
  echo; echo "== uname -a"; uname -a
  echo; echo "== lscpu"; lscpu
  echo; echo "== cpu frequency governor (count per value)"
  cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c || echo "n/a"
  echo; echo "== g++ --version"; g++ --version
  echo "== cmake --version"; cmake --version
  echo; echo "== release build: compile commands (actual flags)"
  python3 -I - "$root/build/release/compile_commands.json" <<'PY'
import json, sys
try:
    for e in json.load(open(sys.argv[1])):
        if e["file"].endswith(("bench_hnsw.cpp", "hnsw.cpp", "distance.cpp", "brute_force.cpp")):
            print(e["command"])
except FileNotFoundError:
    print("build/release not configured; run: cmake --preset release")
PY
  for py in "$root/.venv/bin/python" "$root/.conda-hnswlib/bin/python"; do
    echo; echo "== python ($py)"
    [[ -x "$py" ]] || { echo "not created"; continue; }
    "$py" -I -c '
import sys, importlib.metadata as m
print(sys.version)
for p in ("numpy", "faiss-cpu", "hnswlib", "matplotlib"):
    try:
        print(p, m.version(p))
    except m.PackageNotFoundError:
        print(p, "not installed")'
  done
} > "$out"
echo "wrote $out"
