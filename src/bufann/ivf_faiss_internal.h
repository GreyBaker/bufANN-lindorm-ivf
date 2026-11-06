#pragma once
#include "bufann/ivf_patched_graph.h"
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <memory>

namespace diskann::ivf::faiss_detail {
  using Index = faiss::IndexHNSWFlat;
  using Hit = PatchedGraphHit;
  // One exact-size cached table per calling thread, exclusively leased per
  // operation. Reentrant calls get temporary independent storage.
  class VisitedLease {
   public:
    explicit VisitedLease(size_t count);
    ~VisitedLease();
    VisitedLease(const VisitedLease&) = delete;
    VisitedLease& operator=(const VisitedLease&) = delete;
    faiss::VisitedTable& table() { return *table_; }
   private:
    std::unique_ptr<faiss::VisitedTable> temporary_;
    faiss::VisitedTable* table_ = nullptr;
    uint8_t epoch_ = 0;
  };
  size_t retained_visited_bytes();
  std::unique_ptr<Index> build(uint32_t dim, std::span<const float> values,
                               PatchedGraphOptions options);
  void save(const Index& index, const std::string& path);
  std::unique_ptr<Index> load(const std::string& path, uint32_t dim,
                              std::span<const float> values,
                              PatchedGraphOptions options);
  // Uses unmodified upstream HNSW::search, not the high-level global-stat wrapper.
  std::vector<Hit> search(const Index& index, std::span<const float> query,
                           uint32_t count, uint32_t breadth,
                           std::span<const uint32_t> entries = {},
                           std::span<const float> projected = {},
                           uint32_t projected_dim = 0,
                            uint64_t* computations = nullptr, bool seeded = false,
                            double* kernel_us = nullptr);
  std::vector<Hit> search_layer(const Index& index, faiss::DistanceComputer& dc,
                                 uint32_t count, uint32_t breadth, int level,
                                 std::span<const uint32_t> entries, uint32_t exclude);
  std::unique_ptr<faiss::DistanceComputer> distances(const Index& index);
  float* coordinates(Index& index);
  const float* coordinates(const Index& index);
}
