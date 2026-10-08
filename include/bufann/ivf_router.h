#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>
#include "bufann/ivf_centroids.h"

namespace diskann::ivf {
struct RouterOptions {
    uint32_t degree = 16;
    uint32_t construction_breadth = 200;
    uint32_t query_breadth = 128;
    uint32_t insert_breadth = 64;
    uint32_t seed = 100;
    uint64_t memory_budget_bytes = 1024ull * 1024 * 1024;
};
struct CentroidHit {
    uint32_t list_id;
    double squared_distance;
    bool operator==(const CentroidHit&) const = default;
};

// Immutable centroid-only HNSW. No vector-level graph or posting list ownership.
// Serial construction; concurrent search is supported, save requires quiescence.
class CentroidRouter {
 public:
    static std::unique_ptr<CentroidRouter> build(uint32_t dim,
        std::span<const float> centroids, const RouterOptions& options = {});
    // Three new files: prefix.centroids.bin, prefix.hnsw, prefix.router.
    // Trusted, native 64-bit little-endian artifacts, not crash recovery.
    void save(const std::string& prefix) const;
    static std::unique_ptr<CentroidRouter> load(const std::string& prefix,
        uint64_t memory_budget_bytes = 1024ull * 1024 * 1024);
    ~CentroidRouter();
    CentroidRouter(const CentroidRouter&) = delete;
    CentroidRouter& operator=(const CentroidRouter&) = delete;

    std::vector<CentroidHit> search(std::span<const float> query, uint32_t nprobe,
                                    bool exact = false) const;
    uint32_t assign(std::span<const float> vector, bool exact = false) const;
    template<typename T>
    std::vector<CentroidHit> search(std::span<const T> query, uint32_t nprobe,
                                    bool exact = false) const {
        static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
        if (query.size() != dim()) return search(std::span<const float>{}, nprobe, exact);
        std::vector<float> converted(query.begin(), query.end());
        return search(converted, nprobe, exact);
    }
    template<typename T>
    uint32_t assign(std::span<const T> vector, bool exact = false) const {
        static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
        if (vector.size() != dim()) return assign(std::span<const float>{}, exact);
        std::vector<float> converted(vector.begin(), vector.end());
        return assign(converted, exact);
    }
    uint32_t dim() const { return _dim; }
    uint32_t nlist() const { return static_cast<uint32_t>(_centroids.size() / _dim); }
    const RouterOptions& options() const { return _options; }
    std::span<const float> centroids() const { return _centroids; }
    // Preflight allowance, not measured RSS or a concurrent-search memory cap.
    uint64_t estimated_bytes() const;
    static uint64_t estimate_build_bytes(uint32_t dim, uint32_t nlist,
                                        const RouterOptions& options = {});

 private:
    CentroidRouter(uint32_t dim, std::span<const float> centroids, const RouterOptions& options);
    std::vector<CentroidHit> route(std::span<const float> query, uint32_t count,
                                  uint32_t breadth, bool exact) const;
    struct Impl;
    uint32_t _dim;
    RouterOptions _options;
    std::vector<float> _centroids;
    std::unique_ptr<Impl> _impl;
};

// Shared native sampler/means/empty recovery, using rebuilt centroid HNSW for
// each assignment pass. Router insert_breadth controls training assignment.
// No graph reuse/previous-assignment entry points. Diagnostics report assigned
// sample SSE; total graph distance evaluations are unavailable.
CentroidTraining train_centroids_hnsw(const std::string& path, const BuildConfig& config,
    const RouterOptions& options = {}, const TrainerLimits& limits = {});
} // namespace diskann::ivf
