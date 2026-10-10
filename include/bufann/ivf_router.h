#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>
#include "bufann/ivf_centroids.h"

namespace diskann::ivf {
  // The only accepted/written producer label, independent of artifact framing.
  // It identifies the stock comparison dependency, not every IVF graph mode.
  inline constexpr const char* router_backend_id = "hnswlib-0.8.0-stock1";
  enum class RouterMode {
    Default,
    Hnswlib,
    Exact,
    PatchedFull,
    Refined,
    PCA,
    PCARefined
  };
  RouterMode  parse_router_mode(const std::string& value);
  const char* router_mode_name(RouterMode mode);
  struct RoutingRequest {
    RouterMode mode = RouterMode::Default;
    double     rerank_factor = 0;
    uint32_t   breadth = 0;
  };
  struct RoutingStats {
    double   projection_us = 0, graph_us = 0, centroid_rerank_us = 0;
    uint64_t distance_evaluations = 0;
    bool     distance_count_available = false;
  };
  struct RouterOptions {
    uint32_t   degree = 16;
    uint32_t   construction_breadth = 200;
    uint32_t   query_breadth = 128;
    uint32_t   insert_breadth = 64;
    uint32_t   seed = 100;
    uint64_t   memory_budget_bytes = 1024ull * 1024 * 1024;
    bool       native_modes = false;
    RouterMode mode = RouterMode::Hnswlib;  // Persisted build/insert default.
    uint32_t   refine_breadth = 128;
    bool       refine_reverse_edges = false;
    double     pca_variance = .95, router_rerank_factor = 3;
    uint64_t   native_work_limit = 1000000000;
  };
  struct CentroidHit {
    uint32_t list_id;
    double   squared_distance;
    bool     operator==(const CentroidHit&) const = default;
  };

  // Immutable centroid router: HNSW/exact plus optional patched/refined/PCA
  // modes. No vector-level graph or posting list ownership. Serial
  // construction; concurrent search is supported, save requires quiescence.
  class CentroidRouter {
   public:
    static std::unique_ptr<CentroidRouter> build(
        uint32_t dim, std::span<const float> centroids,
        const RouterOptions& options = {});
    // New files: prefix.centroids.bin, prefix.hnsw, prefix.router; optimized
    // modes also save prefix.patched.hnsw, prefix.refined.hnsw and prefix.pca.
    // Trusted, native 64-bit little-endian artifacts, not crash recovery.
    void save(const std::string& prefix) const;
    static std::unique_ptr<CentroidRouter> load(
        const std::string& prefix,
        uint64_t           memory_budget_bytes = 1024ull * 1024 * 1024);
    ~CentroidRouter();
    CentroidRouter(const CentroidRouter&) = delete;
    CentroidRouter& operator=(const CentroidRouter&) = delete;

    std::vector<CentroidHit> search(std::span<const float> query,
                                    uint32_t nprobe, bool exact = false,
                                    RoutingStats*  stats = nullptr,
                                    RoutingRequest request = {}) const;
    uint32_t assign(std::span<const float> vector, bool exact = false) const;
    template<typename T>
    std::vector<CentroidHit> search(std::span<const T> query, uint32_t nprobe,
                                    bool           exact = false,
                                    RoutingStats*  stats = nullptr,
                                    RoutingRequest request = {}) const {
      static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
      if (query.size() != dim())
        return search(std::span<const float>{}, nprobe, exact);
      std::vector<float> converted(query.begin(), query.end());
      return search(converted, nprobe, exact, stats, request);
    }
    template<typename T>
    uint32_t assign(std::span<const T> vector, bool exact = false) const {
      static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
      if (vector.size() != dim())
        return assign(std::span<const float>{}, exact);
      std::vector<float> converted(vector.begin(), vector.end());
      return assign(converted, exact);
    }
    uint32_t dim() const {
      return _dim;
    }
    uint32_t nlist() const {
      return static_cast<uint32_t>(_centroids.size() / _dim);
    }
    const RouterOptions& options() const {
      return _options;
    }
    std::span<const float> centroids() const {
      return _centroids;
    }
    uint32_t projected_dim() const;
    double   retained_variance() const;
    bool     pca_degenerate() const;
    uint32_t unreachable_before(bool refined) const;
    uint32_t unreachable_after(bool refined) const;
    uint32_t reinserted_nodes(bool refined) const;
    uint32_t zero_indegree(bool refined) const;
    bool     has_native_modes() const {
      return _options.native_modes;
    }
    // Preflight allowance, not measured RSS or a concurrent-search memory cap.
    uint64_t        estimated_bytes() const;
    static uint64_t estimate_build_bytes(uint32_t dim, uint32_t nlist,
                                         const RouterOptions& options = {});

   private:
    CentroidRouter(uint32_t dim, std::span<const float> centroids,
                   const RouterOptions& options);
    std::vector<CentroidHit> route(std::span<const float> query, uint32_t count,
                                   uint32_t breadth, bool exact,
                                   RoutingStats*  stats = nullptr,
                                   RoutingRequest request = {}) const;
    struct Impl;
    uint32_t              _dim;
    RouterOptions         _options;
    std::vector<float>    _centroids;
    std::unique_ptr<Impl> _impl;
  };

  // Shared native sampler/means/empty recovery, using rebuilt centroid HNSW for
  // each assignment pass. Router insert_breadth controls training assignment.
  // No graph reuse/previous-assignment entry points. Diagnostics report
  // assigned sample SSE; total graph distance evaluations are unavailable.
  CentroidTraining train_centroids_hnsw(const std::string&   path,
                                        const BuildConfig&   config,
                                        const RouterOptions& options = {},
                                        const TrainerLimits& limits = {});
  // Select exact, rebuilt/reused full-coordinate controls or LV/LV+ PCA
  // assignment. Training breadth/rerank/refinement settings are independent of
  // final routing. Training graphs/PCA are never persisted or shared with
  // frozen query routers.
  CentroidTraining train_centroids_ann(const std::string&   path,
                                       const BuildConfig&   config,
                                       const RouterOptions& options = {},
                                       const TrainerLimits& limits = {});
}  // namespace diskann::ivf
