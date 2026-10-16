#include "vecdb/dataset.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace vecdb {
namespace {

static_assert(sizeof(float) == 4 && sizeof(std::int32_t) == 4);

// .fvecs and .ivecs share one layout and differ only in the 4-byte element type.
template <typename T>
Matrix<T> load_vecs(const std::filesystem::path &path) {
  auto fail = [&](const std::string& why) {
    return std::runtime_error(path.string() + ": " + why);
  };

  std::ifstream in(path, std::ios::binary);
  if (!in) throw fail("cannot open");
  const auto file_size = std::filesystem::file_size(path);

  std::int32_t dim = 0;
  if (!in.read(reinterpret_cast<char*>(&dim), 4)) throw fail("file is empty");
  if (dim <= 0) throw fail("bad dimension " + std::to_string(dim));

  Matrix<T> m;
  m.d = static_cast<std::size_t>(dim);
  const std::size_t record_bytes = 4 + 4 * m.d;
  if (file_size % record_bytes != 0) throw fail("size is not a whole number of records");
  m.n = file_size / record_bytes;
  m.data.resize(m.n * m.d);

  in.seekg(0);
  for (std::size_t i = 0; i < m.n; ++i) {
    std::int32_t row_dim = 0;
    in.read(reinterpret_cast<char*>(&row_dim), 4);
    if (row_dim != dim) throw fail("row " + std::to_string(i) + " has a different dimension");
    in.read(reinterpret_cast<char*>(m.data.data() + i * m.d), static_cast<std::streamsize>(4 * m.d));
  }
  if (!in) throw fail("read failed");
  return m;
}

}  // namespace

Matrix<float> load_fvecs(const std::filesystem::path& path) { return load_vecs<float>(path); }
Matrix<std::int32_t> load_ivecs(const std::filesystem::path& path) {
  return load_vecs<std::int32_t>(path);
}

}  // namespace vecdb


