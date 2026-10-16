# vecdb

A vector search engine written from scratch in C++ (HNSW index), benchmarked against
FAISS/hnswlib, with a RAG-based system-design mock interviewer built on top.

**Status:** Phase 2 — project skeleton and dataset loader. No search yet.

## Build and test

```sh
scripts/download_data.sh                 # siftsmall (10k vectors) for tests

cmake --preset asan && cmake --build --preset asan && ctest --preset asan        # tests, with sanitizers
cmake --preset release && cmake --build --preset release                         # -O2, for benchmarks
```

Never benchmark the `asan` build: sanitizers slow code down several times over.

## Layout

- `include/vecdb/`, `src/` — the library
- `tests/` — GoogleTest tests
- `scripts/` — dataset download
- `data/` — datasets (git-ignored)

## Decision log

| # | Decision | Alternatives | Why |
|---|---|---|---|
| 1 | One squared-L2 distance kernel. `metric=cosine` means the engine normalizes vectors on insert and on query, then uses L2. | Separate L2 / inner-product / cosine kernels, as in hnswlib | On unit vectors `‖a−b‖² = 2 − 2cos θ`, so L2 and cosine rank identically: one kernel to test and optimize. Raw inner-product search (MIPS) is not needed here; add a kernel if that changes. |
| 2 | C++20 | C++17 | `std::span` lets functions take a view of a vector's floats without copying. |
| 3 | GoogleTest; system copy if installed, else a pinned download | Catch2 | Most widely used in industry; already installed. |
| 4 | Two CMake presets: `release` (`-O2`) and `asan` (debug + ASan + UBSan) | One build with flags toggled by hand | Makes it hard to benchmark a sanitized build by accident. |
| 5 | siftsmall (10k) for tests, SIFT1M for benchmarks; datasets git-ignored and fetched by script | Synthetic random vectors only | Standard datasets with published ground truth make results comparable. |
| 6 | Loader reads the whole file into one flat `std::vector<float>` | `vector<vector<float>>`; memory-mapping | Contiguous storage is cache-friendly (details to be measured in the brute-force phase). SIFT1M is ~500 MB, which fits in RAM, so mmap is unneeded complexity for now. |

## Benchmark results

None yet.

## Limitations

- Loader assumes a little-endian machine.
- Loader reads the entire file into memory.
