#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ivf_cli {
  enum class InsertInputPolicy { Streaming, Staged };
  inline InsertInputPolicy parse_insert_input_policy(std::string_view name) {
    if (name == "streaming")
      return InsertInputPolicy::Streaming;
    if (name == "staged")
      return InsertInputPolicy::Staged;
    throw std::invalid_argument(
        "Insertion input policy must be streaming or staged");
  }
  inline uint64_t insert_staging_bytes(uint64_t rows, uint32_t dim,
                                       uint32_t element_bytes) {
    if (!dim || !element_bytes || rows > UINT64_MAX / dim ||
        rows * dim > uint64_t(PTRDIFF_MAX) / element_bytes)
      throw std::invalid_argument(
          "Insertion staging size overflow/invalid dimension");
    return rows * dim * element_bytes;
  }
  // Preflight only. Caller admits the composed workload budget first; this
  // local check also refuses before allocation or reader/progress callbacks.
  // Preserve counted-ID order and validate every coordinate before returning.
  template<class T, class Read, class Progress>
  std::vector<T> stage_insert_rows(std::span<const uint32_t> ids, uint32_t dim,
                                   uint64_t allowance, Read read,
                                   Progress progress) {
    const auto bytes = insert_staging_bytes(ids.size(), dim, sizeof(T));
    if (bytes > allowance)
      throw std::invalid_argument(
          "Insertion staging exceeds workspace allowance");
    std::vector<T> result(size_t(bytes / sizeof(T)));
    for (size_t i = 0; i < ids.size(); ++i) {
      std::span<T> row(result.data() + i * dim, dim);
      read(ids[i], row);
      for (T value : row)
        if (!std::isfinite(double(value)))
          throw std::invalid_argument("Nonfinite insert coordinate");
      progress(i + 1);
    }
    return result;
  }
}  // namespace ivf_cli
