#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace diskann::ivf {
  struct PatchedGraphHit {
    uint32_t id;
    double   distance;
  };
  struct PatchedGraphOptions {
    uint32_t degree = 16, breadth = 200, seed = 100;
    uint64_t memory_bytes = 1ull << 30, work_limit = 1000000000;
  };
  struct PatchedGraphDiagnostics {
    uint32_t unreachable_before = 0, unreachable_after = 0, zero_indegree = 0;
    uint32_t reinserted_nodes = 0, refined_nodes = 0;
  };
  // Thin ownership/measurement adapter for patched hnswlib, not a native graph.
  // Refinement is serial/quiescent; query contexts are per-call and read-only.
  class PatchedCentroidGraph {
   public:
    static std::unique_ptr<PatchedCentroidGraph> build(
        uint32_t dim, std::span<const float> vectors,
        PatchedGraphOptions options = {});
    static std::unique_ptr<PatchedCentroidGraph> load(
        const std::string& path, uint32_t dim, std::span<const float> vectors,
        PatchedGraphOptions options, PatchedGraphDiagnostics diagnostics);
    ~PatchedCentroidGraph();
    std::vector<PatchedGraphHit> search(std::span<const float> query,
                                        uint32_t count, uint32_t breadth,
                                        std::span<const uint32_t> entries = {},
                                        std::span<const float> projected = {},
                                        uint32_t  projected_dim = 0,
                                        uint64_t* computations = nullptr,
                                        bool      seeded = false) const;
    // Training-only, quiescent: snapshot old neighbors, replace EVERY vector,
    // then one serial ordinary reinsertion per stable ID. No W10 refinement.
    void refresh(std::span<const float> vectors);
    void refine(uint32_t breadth);
    void configure_projection(
        uint32_t dim);  // Quiescent setup before any readers.
    void                    save(const std::string& path) const;
    PatchedGraphDiagnostics diagnostics() const {
      return stats_;
    }

   private:
    PatchedCentroidGraph(uint32_t dim, uint32_t count,
                         PatchedGraphOptions options);
    void measure();
    struct Impl;
    uint32_t                dim_, count_;
    PatchedGraphOptions     options_;
    PatchedGraphDiagnostics stats_;
    std::unique_ptr<Impl>   impl_;
  };
}  // namespace diskann::ivf
