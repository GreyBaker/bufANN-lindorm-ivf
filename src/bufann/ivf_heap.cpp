#include "bufann/ivf_heap.h"

#include <bit>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace diskann::ivf {
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Claim a new artifact without ever truncating an existing file/symlink.
void claim_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    check(fd >= 0, "IVF artifact already exists or cannot be created");
    const int result = ::close(fd);
    check(result == 0, "IVF artifact close failed");
}

uint64_t file_hash(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    check(bool(in), "Cannot read IVF snapshot");
    uint64_t hash = 14695981039346656037ull;
    char buffer[8192];
    while (in.read(buffer, sizeof(buffer)) || in.gcount())
        for (std::streamsize i = 0; i < in.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= 1099511628211ull;
        }
    check(in.eof() && !in.bad(), "IVF snapshot read failed");
    return hash;
}

uint32_t little_u32(const unsigned char* bytes) {
    return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
           uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
}

uint64_t little_u64(const unsigned char* bytes) {
    return uint64_t(little_u32(bytes)) | uint64_t(little_u32(bytes + 4)) << 32;
}

// Bound/check the existing v4 snapshot framing before handing it to the
// backend loader (whose legacy reader does not check every body read).
// These offsets describe the existing little-endian Linux IPGSNP4 header.
void check_snapshot_layout(const std::string& path, const HeapOptions& options,
                           size_t elem_size, uint64_t count, uint64_t heap_bytes) {
    std::ifstream in(path, std::ios::binary);
    unsigned char h[112]{};
    in.read(reinterpret_cast<char*>(h), sizeof(h));
    const uint64_t rid_count = little_u64(h + 64);
    const uint64_t active_cap = little_u64(h + 72);
    const uint64_t page_dirs = little_u64(h + 80);
    const uint64_t space_pages = little_u64(h + 88);
    const uint64_t pages = little_u32(h + 56);
    const uint64_t aligned = (uint64_t(options.dim) + 7) / 8 * 8;
    const uint64_t slots = little_u32(h + 36);
    check(bool(in) && std::memcmp(h, "IPGSNP4", 8) == 0 && little_u32(h + 8) == 4 &&
          little_u32(h + 12) == options.dim && little_u32(h + 16) == aligned &&
          little_u32(h + 20) == 0 && little_u32(h + 24) == elem_size &&
          little_u32(h + 28) == options.page_size &&
          little_u32(h + 32) == 8 + aligned * elem_size &&
          slots > 0 && slots <= UINT16_MAX &&
          8 + (slots + 7) / 8 + slots * (8 + aligned * elem_size) <= options.page_size &&
          little_u32(h + 48) == 0 && little_u32(h + 52) == count &&
          rid_count == count && active_cap <= options.id_capacity &&
          rid_count <= active_cap && pages <= heap_bytes / options.page_size &&
          page_dirs <= pages && space_pages <= page_dirs && little_u64(h + 96) < 64,
          "Invalid IVF backend snapshot layout");
    // RID wire entries: ID(4) + NodeRID(8). Active flags(1) + tags(4),
    // page-directory entries(4), space-page IDs(4), entry pool/reservoir(296).
    const uint64_t expected = 112 + rid_count * 12 + active_cap * 5 +
                              page_dirs * 4 + space_pages * 4 + 296;
    check(std::filesystem::file_size(path) == expected,
          "Truncated or trailing IVF backend snapshot");
}
} // namespace

template<typename T>
DataType HeapStorage<T>::type() {
    if constexpr (std::is_same_v<T, float>) return DataType::Float;
    if constexpr (std::is_same_v<T, int8_t>) return DataType::Int8;
    return DataType::UInt8;
}

template<typename T>
HeapStorage<T>::HeapStorage(const HeapOptions& options, std::string path)
    : _options(options), _heap_path(std::move(path)) {
    static_assert(std::endian::native == std::endian::little,
                  "BufANN snapshots require a little-endian host");
    check(options.dim > 0 && options.dim <= std::numeric_limits<uint32_t>::max() - 7,
          "Invalid IVF heap dimension");
    check(options.id_capacity > 0 && options.id_capacity < inplace::INVALID_NODE,
          "Invalid IVF ID capacity");
    check(options.buffer_frames >= 2, "IVF heap requires at least two frames");
    check(options.page_size >= 512 && options.page_size <= 65536 &&
          options.page_size % 512 == 0, "Invalid IVF page size");
    const uint64_t aligned = (uint64_t(options.dim) + 7) / 8 * 8;
    // Page header + at least one bitmap byte + one zero-degree slot.
    check(17 + aligned * sizeof(T) <= options.page_size,
          "IVF coordinate slot exceeds page size");
    check(!_heap_path.empty(), "Empty IVF heap path");
}

template<typename T>
void HeapStorage<T>::init() {
    _store.init(_options.dim, 0, sizeof(T), _options.page_size,
                _options.buffer_frames, _heap_path, 32, 100, false,
                _options.id_capacity);
    _store.set_page_data_locks_enabled(true);
}

template<typename T>
std::unique_ptr<HeapStorage<T>> HeapStorage<T>::create(
    const HeapOptions& options, const std::string& heap_path) {
    auto heap = std::unique_ptr<HeapStorage>(new HeapStorage(options, heap_path));
    claim_file(heap_path);
    heap->init();
    return heap;
}

template<typename T>
void HeapStorage<T>::require_writable() const {
    check(!_failed, "IVF heap build failed; discard it rather than checkpointing");
    check(!_read_only, "Reopened IVF heap is read-only in S02");
}

template<typename T>
bool HeapStorage<T>::contains(uint32_t id) const {
    return id < _options.id_capacity && _store.is_active(id);
}

template<typename T>
void HeapStorage<T>::append(uint32_t id, std::span<const T> coordinates) {
    require_writable();
    check(id < _options.id_capacity, "IVF external ID exceeds capacity");
    check(!contains(id), "Duplicate IVF external ID");
    check(coordinates.size() == _options.dim, "IVF coordinate dimension mismatch");
    if constexpr (std::is_same_v<T, float>)
        for (float value : coordinates)
            check(std::isfinite(value), "Non-finite IVF coordinate");
    std::vector<T> padded(aligned_dim(), T{});
    std::copy(coordinates.begin(), coordinates.end(), padded.begin());
    try {
        _store.allocate_node(id);
        {
            inplace::NodeRID rid;
            auto frame = _store.pin_page_for_node(id, inplace::WRITE, &rid);
            check(frame.valid(), "IVF allocated page unavailable");
            auto guard = frame.write_guard();
            auto node = guard.write_node(rid);
            check(node.valid() && node.node_id() == id, "IVF allocated slot mismatch");
            std::memcpy(node.coords_bytes(), padded.data(), padded.size() * sizeof(T));
            node.set_degree(0); // Marks the coordinate write dirty as well.
            guard.commit();
        }
        check(_store.set_node_tag(id, id), "IVF external tag assignment failed");
        _store.publish_node(id); // Last operation; no list yet exists in S02.
    } catch (...) {
        _failed = true;
        throw;
    }
}

template<typename T>
void HeapStorage<T>::load_vectors(const std::string& path, uint32_t first_id,
                                 std::optional<uint32_t> row_count) {
    require_writable();
    check(size() == 0, "IVF streaming load requires an empty heap");
    try {
        const auto metadata = inspect_vectors(path, type(), _options.dim);
        const uint32_t rows = row_count.value_or(metadata.rows);
        check(rows > 0 && rows <= metadata.rows, "Invalid IVF input prefix count");
        check(first_id < _options.id_capacity &&
              rows <= _options.id_capacity - first_id,
              "IVF input rows exceed ID capacity");
        std::ifstream in(path, std::ios::binary);
        unsigned char header[8]{};
        in.read(reinterpret_cast<char*>(header), sizeof(header));
        check(bool(in) && little_u32(header) == metadata.rows &&
              little_u32(header + 4) == metadata.cols, "IVF input header changed");
        std::vector<T> row(_options.dim);
        for (uint32_t r = 0; r < rows; ++r) {
            in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
            check(bool(in), "Truncated IVF vector stream");
            append(first_id + r, row);
        }
        if (rows == metadata.rows)
            check(in.peek() == std::char_traits<char>::eof() && !in.bad(),
                  "IVF vector stream has trailing data or I/O error");
    } catch (...) {
        _failed = true;
        throw;
    }
}

template<typename T>
bool HeapStorage<T>::copy_coordinates(uint32_t id, std::span<T> output) {
    check(output.size() >= aligned_dim() && output.data() != nullptr &&
          reinterpret_cast<uintptr_t>(output.data()) % 64 == 0,
          "IVF coordinate output must be padded and 64-byte aligned");
    std::lock_guard<std::mutex> lock(_copy_mutex);
    if (!contains(id)) return false;
    inplace::NodeRID rid;
    auto frame = _store.pin_page_for_node(id, inplace::READ, &rid);
    check(frame.valid(), "IVF coordinate page unavailable");
    auto guard = frame.read_guard();
    auto node = guard.read_node(rid);
    check(node.valid() && node.node_id() == id && node.degree() == 0,
          "IVF coordinate slot mismatch");
    std::memcpy(output.data(), node.coords_bytes(), size_t(aligned_dim()) * sizeof(T));
    return true;
}

template<typename T>
uint32_t HeapStorage<T>::copy_batch_limit() const {
    const auto shard_frames = _store.min_buffer_shard_frames();
    return std::min(inplace::BufferPool::kPinBatchMax, std::max(1u, shard_frames - 1));
}

template<typename T>
BatchCopyResult HeapStorage<T>::copy_coordinates_batch(std::span<const uint32_t> ids,
                                                       std::span<T> output, size_t stride) {
    check(stride >= aligned_dim() && stride <= SIZE_MAX / sizeof(T) && stride * sizeof(T) % 64 == 0,
          "IVF batch coordinate stride must be padded and 64-byte aligned");
    check(ids.size() <= SIZE_MAX / stride && output.size() >= ids.size() * stride &&
          (ids.empty() || (output.data() != nullptr && reinterpret_cast<uintptr_t>(output.data()) % 64 == 0)),
          "Invalid IVF batch coordinate output");
    BatchCopyResult result;
    result.present.resize(ids.size(), 0);
    std::lock_guard<std::mutex> lock(_copy_mutex);
    const auto reads_before = inplace::thread_physical_reads_count();
    // All requested pages could map to one shard. Keep one spare per shard
    // when possible; one-frame shards necessarily use single-page batches.
    const size_t limit = copy_batch_limit();
    for (size_t begin = 0; begin < ids.size(); begin += limit) {
        const size_t count = std::min(limit, ids.size() - begin);
        std::array<inplace::PinnedFrame, inplace::BufferPool::kPinBatchMax> frames;
        std::array<inplace::NodeRID, inplace::BufferPool::kPinBatchMax> rids;
        _store.pin_pages_for_nodes_batch(ids.data() + begin, count, inplace::READ, frames.data(), rids.data());
        ++result.backend_batch_calls;
        for (size_t i = 0; i < count; ++i) {
            if (!contains(ids[begin + i])) continue;
            check(frames[i].valid(), "IVF batch coordinate page unavailable");
            // One guard at a time: duplicate IDs/pages must not recursively
            // acquire the same shared mutex. RAII frames release every pin.
            auto guard = frames[i].read_guard();
            auto node = guard.read_node(rids[i]);
            check(node.valid() && node.node_id() == ids[begin + i] && node.degree() == 0,
                  "IVF batch coordinate slot mismatch");
            std::memcpy(output.data() + (begin + i) * stride, node.coords_bytes(), size_t(aligned_dim()) * sizeof(T));
            result.present[begin + i] = 1;
        }
    }
    result.heap_pages_read = inplace::thread_physical_reads_count() - reads_before;
    return result;
}

template<typename T>
void HeapStorage<T>::checkpoint(const std::string& snapshot_path) {
    require_writable();
    // Checkpoint paths must be distinct from the heap and any existing file.
    // Claim all artifacts before writing so accidental path reuse cannot
    // truncate a heap, another snapshot, or user data.
    check(!std::filesystem::exists(snapshot_path + ".ivf"),
          "IVF checkpoint descriptor already exists");
    claim_file(snapshot_path);
    try {
        _store.flush();
        _store.save_snapshot(snapshot_path);
        const auto meta_size = std::filesystem::file_size(snapshot_path);
        const auto heap_size = std::filesystem::file_size(_heap_path);
        check_snapshot_layout(snapshot_path, _options, sizeof(T), size(), heap_size);
        const auto hash = file_hash(snapshot_path);
        claim_file(snapshot_path + ".ivf");
        std::ofstream out(snapshot_path + ".ivf", std::ios::binary);
        out.imbue(std::locale::classic());
        out << "IVFHEAP1 " << data_type_name(type()) << ' ' << _options.dim << ' '
            << _options.id_capacity << ' ' << _options.page_size << ' ' << size()
            << ' ' << meta_size << ' ' << heap_size << ' ' << hash << '\n';
        out.close();
        check(bool(out), "IVF checkpoint descriptor write failed");
        _read_only = true; // S02 artifacts are immutable after checkpointing.
    } catch (...) {
        _failed = true;
        throw;
    }
}

template<typename T>
std::unique_ptr<HeapStorage<T>> HeapStorage<T>::reopen(
    const HeapOptions& options, const std::string& heap_path,
    const std::string& snapshot_path) {
    auto heap = std::unique_ptr<HeapStorage>(new HeapStorage(options, heap_path));
    std::ifstream in(snapshot_path + ".ivf", std::ios::binary);
    in.imbue(std::locale::classic());
    std::string magic, dtype;
    uint64_t dim = 0, cap = 0, page = 0, count = 0, meta_size = 0, heap_size = 0, hash = 0;
    in >> magic >> dtype >> dim >> cap >> page >> count >> meta_size >> heap_size >> hash;
    check(bool(in) && magic == "IVFHEAP1" && dtype == data_type_name(type()) &&
          dim == options.dim && cap == options.id_capacity && page == options.page_size &&
          count <= cap, "Invalid or incompatible IVF checkpoint descriptor");
    in >> std::ws;
    check(in.eof(), "Trailing IVF checkpoint descriptor data");
    check(std::filesystem::file_size(snapshot_path) == meta_size &&
          std::filesystem::file_size(heap_path) == heap_size &&
          heap_size % options.page_size == 0 && file_hash(snapshot_path) == hash,
          "IVF checkpoint file size/hash mismatch");
    check_snapshot_layout(snapshot_path, options, sizeof(T), count, heap_size);
    heap->init(); // Only after existing files/descriptor have been validated.
    heap->_store.load_snapshot(snapshot_path);
    check(heap->size() == count && heap->_store.n_chunks() == 0 &&
          heap->_store.max_degree() == 0, "IVF checkpoint state mismatch");
    heap->_read_only = true;
    return heap;
}

template class HeapStorage<float>;
template class HeapStorage<int8_t>;
template class HeapStorage<uint8_t>;
} // namespace diskann::ivf
