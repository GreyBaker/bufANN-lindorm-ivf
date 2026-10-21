#pragma once

#include <cstddef>
#include <cstdint>

namespace diskann::ivf::detail {
  // Unchecked numeric primitives: callers retain buffer/value validation and
  // their own nonfinite-result policy. Keep ADC in float and in chunk order;
  // accumulating in double or regrouping contributions changes candidate ties.
  inline float adc_distance_unchecked(const uint8_t* code,
                                      const float* distances,
                                      uint32_t chunks) {
    float result = 0;
    for (uint32_t c = 0; c < chunks; ++c)
      result += distances[std::size_t(c) * 256 + code[c]];
    return result;
  }

  // Exact/objective L2 only, not graph float kernels, PCA or ADC. Convert each
  // operand before subtraction and accumulate from zero in dimension order.
  template<typename T, typename U>
  double squared_l2_double_unchecked(const T* a, const U* b, uint32_t dim) {
    double result = 0;
    for (uint32_t d = 0; d < dim; ++d) {
      const double difference = double(a[d]) - double(b[d]);
      result += difference * difference;
    }
    return result;
  }

  template<typename Candidate>
  bool candidate_nearer(const Candidate& a, const Candidate& b) {
    return a.squared_distance < b.squared_distance ||
           (a.squared_distance == b.squared_distance && a.id < b.id);
  }
}  // namespace diskann::ivf::detail
