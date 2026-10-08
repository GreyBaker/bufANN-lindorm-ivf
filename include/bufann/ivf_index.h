#pragma once

#include <array>
#include "bufann/ivf_artifacts.h"
#include "bufann/ivf_search.h"

namespace diskann::ivf {
inline constexpr std::array<const char*, 8> index_component_suffixes{
    ".centroids.bin", ".hnsw", ".router", ".lists", ".heap", ".heap.meta", ".heap.meta.ivf", ".pq.bin"};

struct IndexManifest {
    BuildConfig training;
    RouterOptions routing;
    HeapOptions heap;
    uint32_t source_rows = 0;
    uint32_t first_id = 0;
    uint32_t chunks = 0;
    // Full source-file identities, including reserved-tail bytes. External PQ
    // training provenance/code-to-vector parity is NOT independently verified.
    std::array<ArtifactDigest, 3> sources{}; // vectors, pivots, codes
    std::array<ArtifactDigest, 8> components{};
};
struct IndexOpenOptions {
    uint32_t buffer_frames = 64;
    uint64_t memory_budget_bytes = 1024ull * 1024 * 1024;
};

// Self-contained immutable bundle, fixed suffixes relative to its prefix.
// Manifest is written last; no atomic publication, rollback or crash recovery.
// Files must remain trusted and immutable during build/load/query. Full artifact
// checksums are integrity checks, not authentication. Linux heap backend required.
template<typename T>
class NativeIndex {
 public:
    static void build(const std::string& prefix, const std::string& vectors,
                      const std::string& pivots, const std::string& codes,
                      const BuildConfig& training, const HeapOptions& heap,
                      const RouterOptions& routing = {}, uint32_t first_id = 0);
    static std::unique_ptr<NativeIndex> open(const std::string& prefix,
                                            const IndexOpenOptions& options = {});
    const IndexManifest& manifest() const { return _manifest; }
    uint64_t estimated_open_bytes() const { return _estimated_open_bytes; }
    uint64_t estimated_query_bytes(const SearchConfig& config, bool exact_routing = false) const;
    QueryResult query(std::span<const T> vector, const SearchConfig& config,
                      const QueryLimits& limits = {}) const;
 private:
    NativeIndex() = default;
    IndexManifest _manifest;
    uint64_t _estimated_open_bytes = 0;
    std::unique_ptr<CentroidRouter> _router;
    std::unique_ptr<PostingLists> _lists;
    std::unique_ptr<PQCodebook<T>> _pq;
    std::unique_ptr<HeapStorage<T>> _heap;
};
extern template class NativeIndex<float>;
extern template class NativeIndex<int8_t>;
extern template class NativeIndex<uint8_t>;
} // namespace diskann::ivf
