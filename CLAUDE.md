# CLAUDE.md — Vector Search Engine (HNSW in C++)

## What this project is
A vector search engine built from scratch in C++20: brute-force exact k-NN and an
HNSW index, benchmarked against FAISS/hnswlib. Later phases add an HTTP service and
a Python RAG / mock-interviewer layer that uses this engine for retrieval.

The developer must be able to explain and defend every line of the search-engine
core in interviews. Tutoring on the core happens in a separate web chat, not here.

## The two zones — read before touching any file

### OWNED CORE — do not write code here
These files are written by the developer, by hand:

- `include/vecdb/distance.hpp` and `src/distance.cpp` — L2 / inner product kernels, normalization
- `include/vecdb/brute_force.hpp` and `src/brute_force.cpp` — exact k-NN (ground truth)
- `include/vecdb/hnsw.hpp` and `src/hnsw.cpp` — HNSW: graph structure, level assignment,
  layer search, neighbor selection, insert, multi-layer search
- `tests/distance_test.cpp`, `tests/brute_force_test.cpp`, `tests/hnsw_test.cpp`
- Benchmark *methodology* (what is measured and how): `bench/` measurement logic

Rules for the owned core:
1. Never create, write, edit, or refactor code in these files — not even "small fixes",
   formatting, or making a failing test pass.
2. When asked to fix or debug something there, give the **smallest useful hint**:
   which function or line region, and *why* it is wrong (e.g. "check the comparator
   direction in your result heap"). No corrected code. Escalate the hint only if the
   developer is still stuck after trying.
3. If asked to write owned code, decline briefly and say this is an owned part —
   ask the developer for their approach, or suggest taking it to the tutor chat.
4. The only override is the developer typing exactly **SHOW ME**. Then: write the
   code, explain every line, and ask 2–3 questions to check understanding.
5. Reading these files, running their tests, and reporting results is always fine.
6. If a delegated change would require changing an owned interface (a signature in
   an owned header), stop and describe the needed change — don't make it.

### DELEGATED — normal pair programming
Everything else: CMake and presets, the `.fvecs`/`.ivecs` dataset loader, CI,
scripts, plotting, benchmark *harness plumbing* (timers, CLI flags, CSV output),
serialization I/O, the HTTP server, Python bindings/client, ingestion, chunking,
embeddings, LLM calls, the mock-interviewer agent, UI, deployment.

Rules for delegated work:
1. Before writing anything non-trivial, list the key design decisions and trade-offs
   (1–3 lines each) and wait for approval.
2. After writing code, give a short explanation of how it works and what the
   developer should be able to explain in an interview.
3. Move fast otherwise.

## Engineering standards
- C++20, CMake, GoogleTest via `FetchContent` (uses an installed GTest if found).
- Vectors are stored contiguously: one flat `std::vector<float>` of size `n * d`,
  with row `i` at offset `i * d`. Never `vector<vector<float>>`. Pass views with
  `std::span<const float>`.
- Correctness first, then performance. Every optimization must be measured, never
  assumed. SIMD and multithreading are later, optional stages.

## Build and test
Two presets:
- `release` — `-O2`. The ONLY build benchmark numbers may come from.
- `asan` — `-O0 -g` + AddressSanitizer + UBSan. Used for all tests.

```
cmake --preset asan && cmake --build --preset asan && ctest --preset asan
cmake --preset release    && cmake --build --preset release
```

Run the tests after every change and report failures honestly.

## Layout
```
include/vecdb/   public headers
src/          implementations
tests/        tests (`<name>_test.cpp`)
bench/        benchmark harness (not created yet)
scripts/      download_data.sh (fetches datasets)
data/         datasets (git-ignored): siftsmall (10K, tests), sift (1M, benchmarks)
```

## Metric decisions already made (see README decision log)
- Supported metrics: L2 and inner product. Cosine = normalize, then inner product.
- Squared L2 internally; sqrt only on the k returned results if values are exposed.
- Normalize at insert time (protects rankings) and at query time (protects score values).
- Zero vectors can't be normalized: reject them under cosine.

## Honesty
- Never fabricate or round up benchmark numbers. Report hardware, build flags,
  dataset, and parameters with every result.
- If a claim in the README wouldn't survive an interviewer's scrutiny, flag it.
- Keep the README's "Design decisions" log and "Limitations" section up to date.
  Remind the developer to commit at the end of each phase.
