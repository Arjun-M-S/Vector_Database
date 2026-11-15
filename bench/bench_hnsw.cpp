// Phase 5 benchmark: recall@k, QPS and per-query latency for vecdb::HNSW (efSearch sweep) and
// brute force on a TEXMEX dataset. Single-threaded. One CSV row per (engine, efSearch).
#include "vecdb/brute_force.hpp"
#include "vecdb/dataset.hpp"
#include "vecdb/hnsw.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__SANITIZE_ADDRESS__)
#define VECDB_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define VECDB_ASAN 1
#endif
#endif

namespace {

using Clock = std::chrono::steady_clock;

// Written after every timed query so the compiler cannot drop a search whose result is unused.
volatile std::size_t g_sink = 0;

const char* const kHeader =
    "engine,M,ef_construction,ef_search,k,recall,qps_median,qps_min,qps_max,p50_us,p99_us,build_s,"
    "threads,peak_mem_mb";

const char* const kUsage = R"(usage: bench_hnsw [flags]
  --data-dir DIR        reads DIR/<name>_base.fvecs, <name>_query.fvecs, <name>_groundtruth.ivecs
                        where <name> is the directory's name (default data/sift)
  --M N                 HNSW links per node (default 16)
  --ef-construction N   (default 200)
  --seed N              HNSW level RNG seed (default 42)
  --ef-search LIST      comma-separated sweep (default 10,16,32,64,128,256,512)
  --k N                 neighbors per query (default 10)
  --repeats N           timed repeats per efSearch (default 5)
  --warmup N            untimed queries before each timed measurement (default 100)
  --bf-queries N        brute-force queries to time, 0 skips brute force (default 500)
  --out FILE            CSV output (default results/bench_hnsw.csv)
)";

struct Options {
  std::filesystem::path data_dir = "data/sift";
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::uint32_t seed = 42;
  std::vector<std::size_t> ef_search = {10, 16, 32, 64, 128, 256, 512};
  std::size_t k = 10;
  std::size_t repeats = 5;
  std::size_t warmup = 100;
  std::size_t bf_queries = 500;
  std::filesystem::path out = "results/bench_hnsw.csv";
};

std::vector<std::size_t> parse_list(const std::string& s) {
  std::vector<std::size_t> v;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) v.push_back(std::stoul(item));
  return v;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") {
      std::cout << kUsage;
      std::exit(0);
    }
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
    const std::string v = argv[++i];
    if (flag == "--data-dir") o.data_dir = v;
    else if (flag == "--M") o.M = std::stoul(v);
    else if (flag == "--ef-construction") o.ef_construction = std::stoul(v);
    else if (flag == "--seed") o.seed = static_cast<std::uint32_t>(std::stoul(v));
    else if (flag == "--ef-search") o.ef_search = parse_list(v);
    else if (flag == "--k") o.k = std::stoul(v);
    else if (flag == "--repeats") o.repeats = std::stoul(v);
    else if (flag == "--warmup") o.warmup = std::stoul(v);
    else if (flag == "--bf-queries") o.bf_queries = std::stoul(v);
    else if (flag == "--out") o.out = v;
    else throw std::invalid_argument("unknown flag " + flag + "\n" + kUsage);
  }
  if (o.k == 0 || o.repeats == 0 || o.ef_search.empty())
    throw std::invalid_argument("--k and --repeats must be > 0 and --ef-search non-empty");
  return o;
}

double elapsed_s(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// Peak resident set size of this process so far (VmHWM), in MiB; -1 if /proc is unavailable.
// CPUs this process may run on (e.g. "2" under taskset -c 2), so each log records its pinning.
std::string allowed_cpus() {
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) != 0) return "unknown";
  std::string out;
  for (int c = 0; c < CPU_SETSIZE; ++c)
    if (CPU_ISSET(c, &set)) out += (out.empty() ? "" : ",") + std::to_string(c);
  return out;
}

double peak_mem_mb() {
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);)
    if (line.rfind("VmHWM:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;  // value is in kB
  return -1.0;
}

// Nearest-rank percentile of sorted samples: the smallest sample with at least p of all
// samples <= it. Always a latency that was actually observed; no interpolation.
double percentile(const std::vector<double>& sorted, double p) {
  const auto rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(sorted.size())));
  return sorted[std::max<std::size_t>(rank, 1) - 1];
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

struct Timing {
  double qps_median, qps_min, qps_max, p50_us, p99_us;
};

// Calls search(i) for i in [0, nq), `repeats` times, timing every call on its own.
// QPS = nq / wall time of the whole loop, so it includes the two clock reads per query.
template <typename Search>
Timing time_queries(std::size_t nq, std::size_t repeats, Search search) {
  std::vector<double> qps, p50, p99, latency_us(nq);
  for (std::size_t r = 0; r < repeats; ++r) {
    const auto loop_start = Clock::now();
    for (std::size_t i = 0; i < nq; ++i) {
      const auto t0 = Clock::now();
      g_sink = search(i);
      latency_us[i] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }
    qps.push_back(static_cast<double>(nq) / elapsed_s(loop_start));
    std::sort(latency_us.begin(), latency_us.end());
    p50.push_back(percentile(latency_us, 0.50));
    p99.push_back(percentile(latency_us, 0.99));
  }
  return {median(qps), *std::min_element(qps.begin(), qps.end()),
          *std::max_element(qps.begin(), qps.end()), median(p50), median(p99)};
}

// |our top-k ids ∩ first k ground-truth ids| / k, averaged over all queries.
double recall_at_k(const vecdb::HNSW& index, const vecdb::Matrix<float>& queries,
                   const vecdb::Matrix<std::int32_t>& gt, std::size_t k, std::size_t ef) {
  std::size_t hits = 0;
  for (std::size_t i = 0; i < queries.n; ++i) {
    std::vector<std::size_t> ids;
    for (const vecdb::Neighbor& nb : index.search(queries.row(i), k, ef)) ids.push_back(nb.id);
    std::sort(ids.begin(), ids.end());
    // A duplicate would be counted twice and inflate recall, so treat it as an engine bug.
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
      throw std::runtime_error("search returned a duplicate id for query " + std::to_string(i));
    const auto truth = gt.row(i).first(k);
    for (const std::size_t id : ids)
      hits += std::find(truth.begin(), truth.end(), static_cast<std::int32_t>(id)) != truth.end();
  }
  return static_cast<double>(hits) / static_cast<double>(queries.n * k);
}

void write_row(std::ofstream& csv, const std::string& engine, const std::string& M,
               const std::string& ef_construction, const std::string& ef_search, std::size_t k,
               double recall, const Timing& t, double build_s, double mem_mb) {
  std::ostringstream line;
  line << std::fixed << engine << ',' << M << ',' << ef_construction << ',' << ef_search << ','
       << k << ',' << std::setprecision(4) << recall << ',' << std::setprecision(1)
       << t.qps_median << ',' << t.qps_min << ',' << t.qps_max << ',' << std::setprecision(2)
       << t.p50_us << ',' << t.p99_us << ',' << build_s << ",1," << std::setprecision(1) << mem_mb;
  csv << line.str() << std::endl;  // flushed, so a crash mid-sweep keeps the finished rows
  std::cout << line.str() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(__OPTIMIZE__) || !defined(NDEBUG) || defined(VECDB_ASAN)
  std::cerr << "\n*** bench_hnsw: unoptimized, assert-enabled or sanitized build. Its numbers are"
               " meaningless. ***\n*** Build with: cmake --preset release && cmake --build --preset"
               " release ***\n\n";
  return 1;
#endif
  try {
    const Options o = parse_args(argc, argv);
    auto dir = o.data_dir;
    if (dir.filename().empty()) dir = dir.parent_path();  // tolerate a trailing slash
    const std::string name = dir.filename().string();

    // Loading is not timed.
    const auto base = vecdb::load_fvecs(dir / (name + "_base.fvecs"));
    const auto queries = vecdb::load_fvecs(dir / (name + "_query.fvecs"));
    const auto gt = vecdb::load_ivecs(dir / (name + "_groundtruth.ivecs"));
    if (queries.d != base.d || gt.n != queries.n || gt.d < o.k)
      throw std::runtime_error("dataset files disagree (dimension, query count, or ground truth < k)");
    std::cout << "data: " << name << " base " << base.n << "x" << base.d << ", " << queries.n
              << " queries, ground truth " << gt.n << "x" << gt.d << ", threads 1, cpus "
              << allowed_cpus() << '\n';

    if (!o.out.parent_path().empty()) std::filesystem::create_directories(o.out.parent_path());
    std::ofstream csv(o.out);
    if (!csv) throw std::runtime_error("cannot write " + o.out.string());
    csv << kHeader << std::endl;
    std::cout << kHeader << '\n';

    // Brute force first, so its peak_mem_mb is the data alone rather than data + HNSW index.
    if (o.bf_queries > 0) {
      const std::size_t nq = std::min(o.bf_queries, queries.n);
      auto bf = [&](std::size_t i) { return vecdb::brute_force_knn(base, queries.row(i), o.k).front().id; };
      for (std::size_t i = 0; i < std::min(o.warmup, nq); ++i) g_sink = bf(i);
      const Timing t = time_queries(nq, o.repeats, bf);
      // Recall 1.0 by definition: brute force is exact (checked against ground truth in tests).
      write_row(csv, "vecdb_bruteforce", "", "", "", o.k, 1.0, t, 0.0, peak_mem_mb());
    }

    vecdb::HNSW index(base.d, o.seed, o.M, o.ef_construction);
    const auto build_start = Clock::now();
    for (std::size_t i = 0; i < base.n; ++i) {
      if (index.insert(base.row(i)) != i)
        throw std::runtime_error("insert returned an id != row index; ground-truth ids would not match");
      if ((i + 1) % 100000 == 0)
        std::cerr << "  inserted " << (i + 1) << "/" << base.n << " (" << elapsed_s(build_start) << " s)\n";
    }
    const double build_s = elapsed_s(build_start);
    const double mem_mb = peak_mem_mb();

    for (const std::size_t ef : o.ef_search) {
      const double recall = recall_at_k(index, queries, gt, o.k, ef);
      auto search = [&](std::size_t i) { return index.search(queries.row(i), o.k, ef).front().id; };
      for (std::size_t i = 0; i < std::min(o.warmup, queries.n); ++i) g_sink = search(i);
      const Timing t = time_queries(queries.n, o.repeats, search);
      write_row(csv, "vecdb_hnsw", std::to_string(o.M), std::to_string(o.ef_construction),
                std::to_string(ef), o.k, recall, t, build_s, mem_mb);
    }
  } catch (const std::exception& e) {
    std::cerr << "bench_hnsw: " << e.what() << '\n';
    return 1;
  }
}
