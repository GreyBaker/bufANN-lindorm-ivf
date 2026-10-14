#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ivf_cli {
  inline constexpr uint32_t max_warmup_queries = 100000;
  inline void               validate_warmup_count(uint32_t count) {
    if (count > max_warmup_queries)
      throw std::invalid_argument("Warmup queries must be in [0, 100000]");
  }
  // Inputs depend ONLY on the frozen base-trained full-coordinate centroids,
  // dtype and requested count. No final query/truth/result feeds this policy.
  // The caller admits one reusable row and existing query scratch first.
  template<class T, class Query, class Progress>
  uint32_t run_centroid_warmup(std::span<const float> centroids, uint32_t dim,
                               uint32_t count, Query query, Progress progress) {
    validate_warmup_count(count);
    if (!count)
      return 0;
    if (!dim || centroids.empty() || centroids.size() % dim ||
        centroids.size() / dim > INT32_MAX)
      throw std::invalid_argument("Invalid warmup centroid shape");
    const auto     nlist = centroids.size() / dim;
    std::vector<T> row(dim);
    uint32_t       completed = 0;
    for (uint32_t i = 0; i < count; ++i) {
      const auto selected = uint64_t(i) * nlist / count;
      for (uint32_t d = 0; d < dim; ++d) {
        const auto value = centroids[size_t(selected) * dim + d];
        if (!std::isfinite(value))
          throw std::invalid_argument("Nonfinite warmup centroid");
        if constexpr (std::is_integral_v<T>)
          row[d] = T(std::clamp(std::round(double(value)),
                                double(std::numeric_limits<T>::lowest()),
                                double(std::numeric_limits<T>::max())));
        else
          row[d] = T(value);
      }
      query(std::span<const T>(row));
      progress(++completed);
    }
    return completed;
  }
}  // namespace ivf_cli
