#pragma once

#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "bufann/inplace_backend.h"
#include "bufann/ivf_io.h"

namespace diskann::ivf {
template<typename T> class MutableIndex;

struct HeapOptions {
    uint32_t dim = 0;
    uint32_t id_capacity = 0;
    uint32_t page_size = 4096;
    uint32_t buffer_frames = 64;
};
struct BatchCopyResult {
    std::vector<uint8_t> present;
    uint64_t backend_batch_calls = 0; // Logical API calls, NOT disk reads or IOPS.
    uint64_t heap_pages_read = 0; // Backend thread-local page-read delta, not device IOPS.
};

// Graph-free typed storage. Immutable build/checkpoint operations require
// quiescent access. Opt-in mutable sessions serialize backend writes/copies and
// logical-state operations. MutableIndex owns a separate checkpoint family;
// the legacy immutable checkpoint method refuses mutable sessions.
// IDs are bounded external IDs and also the backend node IDs (never remapped).
// A failed stream or backend mutation poisons checkpointing; discard that build.
template<typename T>
class HeapStorage {
 public:
    static std::unique_ptr<HeapStorage> create(const HeapOptions& options,
                                              const std::string& heap_path);
    static std::unique_ptr<HeapStorage> reopen(const HeapOptions& options,
                                              const std::string& heap_path,
                                              const std::string& snapshot_path);
    // S05 opt-in session: owns a NEW heap or an explicitly private copy of an
    // immutable v1 heap. Mutable checkpoint composition is private to MutableIndex.
    static std::unique_ptr<HeapStorage> create_mutable(const HeapOptions& options, const std::string& heap_path);
    void prepare_insert(uint32_t id, std::span<const T> coordinates);
    // Logical publication only (no page I/O), but may wait for an admitted copy/
    // write section. Call before publishing the corresponding list entry, after
    // its growth/code-copy failure points. Never holds heap lock then list lock.
    void publish_insert(uint32_t id);
    void cancel_insert(uint32_t id); // Prepared -> Aborted; retain slot for retry.
    bool retire(uint32_t id); // Logical delete; never reclaim/reuse committed IDs.
    bool mutable_mode() const { return _mutable_mode; }
    uint32_t retained_size() const { return _store.num_active(); } // Includes deleted/aborted records.
    uint64_t retained_file_bytes() const;
    uint32_t id_capacity() const { return _options.id_capacity; }
    void append(uint32_t id, std::span<const T> coordinates);
    // Stream all rows, or an explicit base prefix, assigning first_id + row.
    // Empty heap only; input must remain immutable during the operation.
    void load_vectors(const std::string& path, uint32_t first_id = 0,
                      std::optional<uint32_t> row_count = std::nullopt);
    bool contains(uint32_t id) const;
    // Caller owns >= aligned_dim() elements at a 64-byte-aligned address.
    // No heap pointer escapes the guard; missing IDs leave output untouched.
    bool copy_coordinates(uint32_t id, std::span<T> output);
    // Real backend/libaio batch pins, bounded by pool capacity and kPinBatchMax.
    // Each row starts on a 64-byte boundary; stride >= aligned_dim(). Missing
    // IDs leave their row untouched. All adapter pin/copy sections serialize
    // initially to avoid pool exhaustion under concurrent batches.
    BatchCopyResult copy_coordinates_batch(std::span<const uint32_t> ids,
                                          std::span<T> output, size_t stride);
    uint32_t copy_batch_limit() const;
    // New snapshot path only; writes backend metadata plus .ivf descriptor.
    // Clean shutdown only, not atomic heap generations or crash/WAL recovery.
    void checkpoint(const std::string& snapshot_path);
    uint32_t size() const { return _mutable_mode ? _live_count.load() : _store.num_active(); }
    uint32_t aligned_dim() const { return _store.aligned_dim(); }
    uint32_t dim() const { return _options.dim; }
    bool failed() const { return _failed; }
    bool read_only() const { return _read_only; }

  private:
    friend class MutableIndex<T>;
    static std::unique_ptr<HeapStorage> reopen_physical(const HeapOptions& options,
        const std::string& heap_path, const std::string& snapshot_path, const char* descriptor_magic);
    // Only the native session may reopen a copy, after its exclusive claim and
    // component-content digest checks. Never make an immutable source writable.
    static std::unique_ptr<HeapStorage> reopen_mutable_copy(const HeapOptions& options,
        const std::string& private_heap_path, const std::string& immutable_snapshot_path);
    static std::unique_ptr<HeapStorage> reopen_mutable_checkpoint_copy(const HeapOptions& options,
        const std::string& private_heap_path, const std::string& snapshot_path, const std::string& state_path);
    void checkpoint_mutable_copy(const std::string& new_heap_path, const std::string& snapshot_path,
        const std::string& state_path);
    void validate_retained_coordinates();
    explicit HeapStorage(const HeapOptions& options, std::string path);
    static DataType type();
    void init();
    void require_writable() const;
    bool contains_unlocked(uint32_t id) const;
    void write_coordinates(uint32_t id, std::span<const T> coordinates, bool allocate);
    enum class MutableState : uint8_t { Absent, Prepared, Live, Deleted, Aborted };
    HeapOptions _options;
    std::string _heap_path;
    inplace::InPlaceGraphStore _store;
    mutable std::mutex _copy_mutex; // All mutable backend metadata/write/copy sections serialize.
    std::atomic<bool> _failed{false};
    std::atomic<bool> _read_only{false};
    bool _mutable_mode = false; // Set before publishing the object; immutable afterward.
    std::vector<MutableState> _mutable_states; // Protected by _copy_mutex, allocated once.
    std::atomic<uint32_t> _live_count{0};
};

extern template class HeapStorage<float>;
extern template class HeapStorage<int8_t>;
extern template class HeapStorage<uint8_t>;

} // namespace diskann::ivf
