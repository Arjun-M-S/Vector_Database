#include "vecdb/dataset.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Writes rows in the .fvecs/.ivecs record layout: int32 dim, then the values.
template <typename T>
fs::path write_vecs(const std::string& name, const std::vector<std::vector<T>>& rows) {
  const fs::path path = fs::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary);
  for (const auto& row : rows) {
    const auto dim = static_cast<std::int32_t>(row.size());
    out.write(reinterpret_cast<const char*>(&dim), 4);
    out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(4 * row.size()));
  }
  return path;
}

}  // namespace

TEST(Dataset, LoadsFvecsIntoFlatArray) {
  const auto path = write_vecs<float>("vecdb_ok.fvecs", {{1, 2, 3}, {4, 5, 6}});
  const auto m = vecdb::load_fvecs(path);
  EXPECT_EQ(m.n, 2u);
  EXPECT_EQ(m.d, 3u);
  EXPECT_EQ(m.data, (std::vector<float>{1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(m.row(1)[0], 4.0f);
  EXPECT_EQ(m.row(1).size(), 3u);
}

TEST(Dataset, LoadsIvecs) {
  const auto path = write_vecs<std::int32_t>("vecdb_ok.ivecs", {{7, 8}, {9, 10}, {11, 12}});
  const auto m = vecdb::load_ivecs(path);
  EXPECT_EQ(m.n, 3u);
  EXPECT_EQ(m.d, 2u);
  EXPECT_EQ(m.row(2)[1], 12);
}

TEST(Dataset, RejectsMissingFile) {
  EXPECT_THROW(vecdb::load_fvecs("/nonexistent/nope.fvecs"), std::runtime_error);
}

TEST(Dataset, RejectsTruncatedFile) {
  const auto path = write_vecs<float>("vecdb_trunc.fvecs", {{1, 2, 3}, {4, 5, 6}});
  fs::resize_file(path, fs::file_size(path) - 4);
  EXPECT_THROW(vecdb::load_fvecs(path), std::runtime_error);
}

TEST(Dataset, RejectsRowsOfDifferentDimension) {
  // Same total size as two 3-d rows, but the second row claims a different dimension.
  const auto path = write_vecs<float>("vecdb_ragged.fvecs", {{1, 2, 3}, {4, 5, 6}});
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    const std::int32_t wrong = 2;
    f.seekp(16);
    f.write(reinterpret_cast<const char*>(&wrong), 4);
  }
  EXPECT_THROW(vecdb::load_fvecs(path), std::runtime_error);
}

TEST(Dataset, LoadsSiftSmall) {
  const fs::path dir = fs::path(VECDB_DATA_DIR) / "siftsmall";
  if (!fs::exists(dir / "siftsmall_base.fvecs")) GTEST_SKIP() << "run scripts/download_data.sh";
  const auto base = vecdb::load_fvecs(dir / "siftsmall_base.fvecs");
  const auto query = vecdb::load_fvecs(dir / "siftsmall_query.fvecs");
  const auto truth = vecdb::load_ivecs(dir / "siftsmall_groundtruth.ivecs");
  EXPECT_EQ(base.n, 10000u);
  EXPECT_EQ(base.d, 128u);
  EXPECT_EQ(query.n, 100u);
  EXPECT_EQ(query.d, 128u);
  EXPECT_EQ(truth.n, 100u);
  EXPECT_EQ(truth.d, 100u);
}
