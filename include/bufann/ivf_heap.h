#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "bufann/inplace_backend.h"
#include "bufann/ivf_io.h"

namespace diskann::ivf {

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

// Build-time graph-free storage. Mutations/checkpoints require exclusive,
// quiescent access; concurrent reads are supported after build/reopen only.
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
    uint32_t size() const { return _store.num_active(); }
    uint32_t aligned_dim() const { return _store.aligned_dim(); }
    uint32_t dim() const { return _options.dim; }
    bool failed() const { return _failed; }
    bool read_only() const { return _read_only; }

 private:
    explicit HeapStorage(const HeapOptions& options, std::string path);
    static DataType type();
    void init();
    void require_writable() const;
    HeapOptions _options;
    std::string _heap_path;
    inplace::InPlaceGraphStore _store;
    std::mutex _copy_mutex;
    bool _failed = false;
    bool _read_only = false;
};

extern template class HeapStorage<float>;
extern template class HeapStorage<int8_t>;
extern template class HeapStorage<uint8_t>;

} // namespace diskann::ivf
