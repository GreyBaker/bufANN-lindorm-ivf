#pragma once

#include <cstdint>
#include <random>

namespace diskann::ivf::detail {
  // Positive bound, uint32_t-sized result. Preserve the rejection threshold
  // and mt19937_64 draw sequence used by training and calibration sampling.
  inline uint32_t bounded_random(std::mt19937_64& rng, uint64_t bound) {
    const uint64_t threshold = (uint64_t(0) - bound) % bound;
    uint64_t       value;
    do {
      value = rng();
    } while (value < threshold);
    return static_cast<uint32_t>(value % bound);
  }
}  // namespace diskann::ivf::detail
