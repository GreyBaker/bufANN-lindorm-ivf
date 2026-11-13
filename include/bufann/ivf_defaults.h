#pragma once

#include <cstdint>

namespace diskann::ivf::defaults {
  inline constexpr uint32_t nlist = 400000;
  inline constexpr uint32_t nprobe = 96;
  inline constexpr uint32_t build_threads = 16;
  inline constexpr uint32_t query_threads = 8;
  inline constexpr uint32_t mixed_threads = 16;
  inline constexpr double memory_budget_gb = 96;
  inline constexpr uint64_t memory_budget_bytes = 96ull << 30;
  inline constexpr uint32_t page_size = 4096;
  inline constexpr uint32_t buffer_frames = 262144;  // 1 GiB page payload.
  inline constexpr uint32_t warmup_queries = 64;
  // PCA covariance at 500K x 128D alone counts 4.128B operations.
  // Independent of exact-distance and assignment-search work guards.
  inline constexpr uint64_t routing_work_limit = 10000000000ull;
}  // namespace diskann::ivf::defaults
