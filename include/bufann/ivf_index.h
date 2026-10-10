#pragma once

#include <array>
#include "bufann/ivf_artifacts.h"
#include "bufann/ivf_search.h"

namespace diskann::ivf {
template<typename T> class MutableIndex;
inline constexpr std::array<const char*, 8> index_component_suffixes{
    ".centroids.bin", ".hnsw", ".router", ".lists", ".heap", ".heap.meta", ".heap.meta.ivf", ".pq.bin"};
inline constexpr std::array<const char*, 10> mutable_component_suffixes{
    ".centroids.bin", ".hnsw", ".router", ".mlists", ".heap", ".heap.meta", ".heap.meta.ivf", ".pq.bin", ".heap.state", ".origin"};
inline constexpr std::array<const char*, 3> native_router_suffixes{".patched.hnsw", ".refined.hnsw", ".pca"};

struct IndexManifest {
    uint32_t format_version = 3; // Preserve legacy producer framing in mutable origin checkpoints.
    std::string router_backend = "hnswlib-0.8.0-prefetch1";
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
    std::array<ArtifactDigest, 3> native_components{}; // Optimized-mode bundles, version 2/3/4.
    uint32_t pca_retained_dim = 0; // Actual fitted projection, digest-bound below.
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
                      const RouterOptions& routing = {}, uint32_t first_id = 0,
                      TrainingStats* training_stats = nullptr);
    static std::unique_ptr<NativeIndex> open(const std::string& prefix,
                                            const IndexOpenOptions& options = {});
    const IndexManifest& manifest() const { return _manifest; }
    const CentroidRouter& router() const { return *_router; }
    uint64_t estimated_open_bytes() const { return _estimated_open_bytes; }
    uint64_t estimated_query_bytes(const SearchConfig& config, bool exact_routing = false) const;
    QueryResult query(std::span<const T> vector, const SearchConfig& config,
                      const QueryLimits& limits = {}) const;
  private:
    friend class MutableIndex<T>;
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

struct MutableIndexOptions {
    uint32_t buffer_frames = 64;
    uint64_t memory_budget_bytes = 1024ull * 1024 * 1024;
    uint64_t list_memory_budget_bytes = 0; // 0 uses remaining retained-object budget.
};
struct MutationLimits {
    uint64_t workspace_bytes = 64ull * 1024 * 1024;
    bool exact_routing = false;
};

// Native mutable session, forked onto a verified private heap file. Original bundle
// artifacts remain immutable. Mutable checkpoints are separate immutable families;
// reopen copies onto a new private heap/PQ sidecar. CLI updates use this API.
// Frozen router/codebook; logically deleted/aborted physical slots are retained.
template<typename T>
class MutableIndex {
  public:
    static std::unique_ptr<MutableIndex> fork(const std::string& immutable_prefix,
        const std::string& new_private_heap_path, const MutableIndexOptions& options = {});
    static std::unique_ptr<MutableIndex> reopen(const std::string& checkpoint_prefix,
        const std::string& new_private_heap_path, const MutableIndexOptions& options = {});
    // Exclusive lifecycle admission waits for current operations to quiesce;
    // new operations resume afterward. NEW prefix only, manifest written last.
    // Preserve partial failures; no fsync/atomic generation/crash recovery.
    void checkpoint(const std::string& new_prefix, uint64_t workspace_bytes = 64ull * 1024 * 1024);
    uint64_t estimated_checkpoint_bytes() const;
    const IndexManifest& origin_manifest() const { return _origin; }
    const CentroidRouter& router() const { return *_router; }
    uint32_t size() const { return _lists->size(); }
    uint32_t retained_heap_records() const { return _heap->retained_size(); }
    uint64_t retained_heap_bytes() const { return _heap->retained_file_bytes(); }
    uint64_t estimated_fork_bytes() const { return _fork_bytes; }
    uint64_t estimated_retained_bytes() const { return _other_bytes + _lists->estimated_bytes(); }
    bool failed() const { return _failed.load() || _heap->failed(); }
    bool contains(uint32_t id) const { return _lists->contains(id); }
    uint64_t missing_deletes() const { return _lists->missing_deletes(); }
    ListSnapshot copy_list(uint32_t list, uint64_t workspace_bytes = 64ull * 1024 * 1024) const {
        std::shared_lock admission(_lifecycle);
        return _lists->copy_list(list, workspace_bytes);
    }
    uint64_t estimated_mutation_bytes(bool exact_routing = false) const;
    void insert(uint32_t id, std::span<const T> coordinates, const MutationLimits& limits = {});
    bool erase(uint32_t id);
    uint64_t estimated_query_bytes(const SearchConfig& config, bool exact_routing = false) const;
    QueryResult query(std::span<const T>, const SearchConfig&, const QueryLimits& = {}) const;
    QueryResult rerank_candidates(std::span<const T>, std::span<const PQCandidate>,
        const SearchConfig&, const QueryLimits& = {}) const;
  private:
    MutableIndex() = default;
    void require_healthy() const;
    void validate_checkpoint_membership() const;
    mutable std::shared_mutex _lifecycle;
    IndexManifest _origin;
    uint64_t _fork_bytes = 0, _other_bytes = 0;
    std::atomic<bool> _failed{false};
    std::string _pq_artifact_path; // Owned immutable private-heap + .pq.bin sidecar.
    std::unique_ptr<CentroidRouter> _router;
    std::unique_ptr<PQCodebook<T>> _pq;
    std::unique_ptr<MutablePostingLists> _lists;
    std::unique_ptr<HeapStorage<T>> _heap;
};
extern template class MutableIndex<float>;
extern template class MutableIndex<int8_t>;
extern template class MutableIndex<uint8_t>;
} // namespace diskann::ivf
