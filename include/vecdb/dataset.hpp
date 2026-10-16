#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace vecdb {

// n rows of d values each, stored back to back in one contiguous array:
// row i occupies data[i*d .. i*d + d).
template <typename T>
struct Matrix {
  std::vector<T> data;
  std::size_t n = 0;
  std::size_t d = 0;

  std::span<const T> row(std::size_t i) const { return {data.data() + i * d, d}; }
};

// TEXMEX formats (SIFT1M etc.): each record is an int32 dimension followed by
// that many 4-byte values (float32 for .fvecs, int32 for .ivecs), little-endian.
// Throws std::runtime_error if the file is missing or malformed.
Matrix<float> load_fvecs(const std::filesystem::path& path);
Matrix<std::int32_t> load_ivecs(const std::filesystem::path& path);

}  // namespace vecdb
