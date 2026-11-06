#include "bufann/ivf_heap.h"
#include "bufann/ivf_fetch_timing.h"
#include "bufann/ivf_artifacts.h"

#include <bit>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace diskann::ivf {
  namespace {
    void check(bool condition, const char* message) {
      if (!condition)
        throw std::runtime_error(message);
    }

    uint32_t little_u32(const unsigned char* bytes) {
      return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
             uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
    }

    uint64_t little_u64(const unsigned char* bytes) {
      return uint64_t(little_u32(bytes)) | uint64_t(little_u32(bytes + 4))
                                               << 32;
    }

    // Bound/check the existing v4 snapshot framing before handing it to the
    // backend loader (whose legacy reader does not check every body read).
    // These offsets describe the existing little-endian Linux IPGSNP4 header.
    void check_snapshot_layout(const std::string& path,
                               const HeapOptions& options, size_t elem_size,
                               uint64_t count, uint64_t heap_bytes) {
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
      check(bool(in) && std::memcmp(h, "IPGSNP4", 8) == 0 &&
                little_u32(h + 8) == 4 && little_u32(h + 12) == options.dim &&
                little_u32(h + 16) == aligned && little_u32(h + 20) == 0 &&
                little_u32(h + 24) == elem_size &&
                little_u32(h + 28) == options.page_size &&
                little_u32(h + 32) == 8 + aligned * elem_size && slots > 0 &&
                slots <= UINT16_MAX &&
                8 + (slots + 7) / 8 + slots * (8 + aligned * elem_size) <=
                    options.page_size &&
                little_u32(h + 48) == 0 && little_u32(h + 52) == count &&
                rid_count == count && active_cap <= options.id_capacity &&
                rid_count <= active_cap &&
                pages <= heap_bytes / options.page_size && page_dirs <= pages &&
                space_pages <= page_dirs && little_u64(h + 96) < 64,
            "Invalid IVF backend snapshot layout");
      // RID wire entries: ID(4) + NodeRID(8). Active flags(1) + tags(4),
      // page-directory entries(4), space-page IDs(4), entry
      // pool/reservoir(296).
      const uint64_t expected = 112 + rid_count * 12 + active_cap * 5 +
                                page_dirs * 4 + space_pages * 4 + 296;
      check(std::filesystem::file_size(path) == expected,
            "Truncated or trailing IVF backend snapshot");
    }
  }  // namespace

  template<typename T>
  DataType HeapStorage<T>::type() {
    if constexpr (std::is_same_v<T, float>)
      return DataType::Float;
    if constexpr (std::is_same_v<T, int8_t>)
      return DataType::Int8;
    return DataType::UInt8;
  }

  template<typename T>
  HeapStorage<T>::HeapStorage(const HeapOptions& options, std::string path)
      : _options(options), _heap_path(std::move(path)) {
    static_assert(std::endian::native == std::endian::little,
                  "BufANN snapshots require a little-endian host");
    check(options.dim > 0 &&
              options.dim <= std::numeric_limits<uint32_t>::max() - 7,
          "Invalid IVF heap dimension");
    check(
        options.id_capacity > 0 && options.id_capacity < inplace::INVALID_NODE,
        "Invalid IVF ID capacity");
    check(options.buffer_frames >= 2, "IVF heap requires at least two frames");
    check(options.page_size >= 512 && options.page_size <= 65536 &&
              options.page_size % 512 == 0,
          "Invalid IVF page size");
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
    _pins = std::make_unique<detail::PinAdmission>(
        std::max(1u, _store.min_buffer_shard_frames() - 1));
  }

  template<typename T>
  std::unique_ptr<HeapStorage<T>> HeapStorage<T>::create(
      const HeapOptions& options, const std::string& heap_path) {
    auto heap =
        std::unique_ptr<HeapStorage>(new HeapStorage(options, heap_path));
    claim_artifact(heap_path, 0644);
    heap->init();
    return heap;
  }

  template<typename T>
  void HeapStorage<T>::require_writable() const {
    check(!_failed, "IVF heap failed; discard the build/session");
    check(!_read_only, "Reopened immutable IVF heap is read-only");
  }

  template<typename T>
  void HeapStorage<T>::poison(std::exception_ptr error) noexcept {
    _failed = true;
    _pins->poison(error);
  }

  template<typename T>
  std::unique_lock<std::mutex> HeapStorage<T>::lock_writer() {
    _writer_waiters.fetch_add(1, std::memory_order_acq_rel);
    try {
      std::unique_lock lock(_writer_mutex);
      _writer_waiters.fetch_sub(1, std::memory_order_acq_rel);
      return lock;
    } catch (...) {
      _writer_waiters.fetch_sub(1, std::memory_order_acq_rel);
      throw;
    }
  }

  template<typename T>
  bool HeapStorage<T>::contains(uint32_t id) const {
    std::lock_guard lock(_copy_mutex);
    return contains_unlocked(id);
  }
  template<typename T>
  bool HeapStorage<T>::contains_unlocked(uint32_t id) const {
    return id < _options.id_capacity &&
           (_mutable_mode ? _mutable_states[id] == MutableState::Live
                          : _store.is_active(id));
  }
  template<typename T>
  uint64_t HeapStorage<T>::retained_file_bytes() const {
    std::lock_guard lock(_copy_mutex);
    return std::filesystem::file_size(_heap_path);
  }

  template<typename T>
  void HeapStorage<T>::append(uint32_t id, std::span<const T> coordinates) {
    auto writer = lock_writer();
    {
      std::lock_guard lock(_copy_mutex);
      require_writable();
      check(!_mutable_mode, "Mutable IVF heap requires prepared insertion");
      check(id < _options.id_capacity, "IVF external ID exceeds capacity");
      check(!contains_unlocked(id), "Duplicate IVF external ID");
    }
    write_coordinates(id, coordinates, true);
  }
  template<typename T>
  void HeapStorage<T>::write_coordinates(uint32_t           id,
                                         std::span<const T> coordinates,
                                         bool allocate, bool sequential) {
    check(coordinates.size() == _options.dim,
          "IVF coordinate dimension mismatch");
    if constexpr (std::is_same_v<T, float>)
      for (float value : coordinates)
        check(std::isfinite(value), "Non-finite IVF coordinate");
    std::vector<T> padded(aligned_dim(), T{});
    std::copy(coordinates.begin(), coordinates.end(), padded.begin());
    try {
      uint64_t wait = 0;
      auto     pins = _pins->acquire(1, wait);
      try {
        if (allocate) {
          if (sequential)
            _store.allocate_node_sequential(id);
          else
            _store.allocate_node(id);
        }
        {
          inplace::NodeRID rid;
          auto frame = _store.pin_page_for_node(id, inplace::WRITE, &rid);
          check(frame.valid(), "IVF allocated page unavailable");
          auto guard = frame.write_guard();
          auto node = guard.write_node(rid);
          check(node.valid() && node.node_id() == id,
                "IVF allocated slot mismatch");
          std::memcpy(node.coords_bytes(), padded.data(),
                      padded.size() * sizeof(T));
          node.set_degree(0);  // Marks the coordinate write dirty as well.
          guard.commit();
        }
        check(_store.set_node_tag(id, id),
              "IVF external tag assignment failed");
        if (allocate)
          _store.publish_node(id);  // Physical readiness; mutable logical
                                    // visibility is separate.
      } catch (...) {
        poison(std::current_exception());
        throw;
      }
    } catch (...) {
      poison(std::current_exception());
      throw;
    }
  }

  template<typename T>
  std::unique_ptr<HeapStorage<T>> HeapStorage<T>::create_mutable(
      const HeapOptions& options, const std::string& path) {
    auto heap = std::unique_ptr<HeapStorage>(new HeapStorage(options, path));
    heap->_mutable_states.resize(options.id_capacity, MutableState::Absent);
    heap->_mutable_mode = true;
    claim_artifact(path, 0644);
    heap->init();
    return heap;
  }
  template<typename T>
  std::unique_ptr<HeapStorage<T>> HeapStorage<T>::reopen_mutable_copy(
      const HeapOptions& options, const std::string& path,
      const std::string& snapshot) {
    auto heap = reopen(options, path, snapshot);
    heap->_mutable_states.resize(options.id_capacity, MutableState::Absent);
    uint32_t live = 0;
    for (uint32_t id = 0; id < options.id_capacity; ++id)
      if (heap->_store.is_active(id)) {
        heap->_mutable_states[id] = MutableState::Live;
        ++live;
      }
    check(live == heap->_store.num_active(),
          "Mutable IVF heap initial ledger/population mismatch");
    heap->_live_count.store(live);
    heap->_mutable_mode = true;
    heap->_read_only = false;
    return heap;
  }
  template<typename T>
  void HeapStorage<T>::prepare_insert(uint32_t           id,
                                      std::span<const T> coordinates) {
    auto         writer = lock_writer();
    MutableState state;
    {
      std::lock_guard lock(_copy_mutex);
      require_writable();
      check(_mutable_mode, "IVF heap is not a mutable session");
      check(id < id_capacity(), "IVF external ID exceeds capacity");
      state = _mutable_states[id];
      check(state == MutableState::Absent || state == MutableState::Aborted,
            "Mutable IVF heap ID already prepared/live/deleted; reuse refused");
    }
    // IVF owns/serializes this heap; fill one page instead of cycling across
    // the backend's 256-page growth batch. Aborted retries retain their slot.
    write_coordinates(id, coordinates, state == MutableState::Absent, true);
    std::lock_guard lock(_copy_mutex);
    require_writable();
    _mutable_states[id] = MutableState::Prepared;
  }
  template<typename T>
  void HeapStorage<T>::publish_insert(uint32_t id) {
    std::lock_guard lock(_copy_mutex);
    require_writable();
    check(_mutable_mode && id < id_capacity() &&
              _mutable_states[id] == MutableState::Prepared,
          "Mutable IVF heap publication requires a prepared ID");
    _mutable_states[id] = MutableState::Live;
    _live_count.fetch_add(1);
  }
  template<typename T>
  void HeapStorage<T>::cancel_insert(uint32_t id) {
    auto            writer = lock_writer();
    std::lock_guard lock(_copy_mutex);
    check(_mutable_mode && id < id_capacity() &&
              (_mutable_states[id] == MutableState::Prepared ||
               _mutable_states[id] == MutableState::Aborted),
          "Mutable IVF heap cancellation requires an unpublished prepared ID");
    _mutable_states[id] = MutableState::Aborted;
  }
  template<typename T>
  bool HeapStorage<T>::retire(uint32_t id) {
    std::lock_guard lock(_copy_mutex);
    require_writable();
    check(_mutable_mode && id < id_capacity(),
          "Invalid mutable IVF heap deletion");
    if (_mutable_states[id] != MutableState::Live)
      return false;
    _mutable_states[id] = MutableState::Deleted;
    _live_count.fetch_sub(1);
    return true;
  }

  template<typename T>
  void HeapStorage<T>::load_vectors(const std::string& path, uint32_t first_id,
                                    std::optional<uint32_t> row_count) {
    require_writable();
    check(!_mutable_mode, "Mutable IVF heap does not support streaming load");
    check(size() == 0, "IVF streaming load requires an empty heap");
    try {
      const auto     metadata = inspect_vectors(path, type(), _options.dim);
      const uint32_t rows = row_count.value_or(metadata.rows);
      check(rows > 0 && rows <= metadata.rows,
            "Invalid IVF input prefix count");
      check(first_id < _options.id_capacity &&
                rows <= _options.id_capacity - first_id,
            "IVF input rows exceed ID capacity");
      std::ifstream in(path, std::ios::binary);
      unsigned char header[8]{};
      in.read(reinterpret_cast<char*>(header), sizeof(header));
      check(bool(in) && little_u32(header) == metadata.rows &&
                little_u32(header + 4) == metadata.cols,
            "IVF input header changed");
      std::vector<T> row(_options.dim);
      const auto     progress_start = std::chrono::steady_clock::now();
      for (uint32_t r = 0; r < rows; ++r) {
        in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
        check(bool(in), "Truncated IVF vector stream");
        // Finish each physical page before moving on. Round-robin append would
        // cycle across 256 allocation pages even with only 64 buffer frames,
        // repeatedly reading/writing every partially filled page. Source IDs,
        // validation, publication and framing stay unchanged. Generic append
        // retains its round-robin placement; mutable IVF uses page filling too.
        auto writer = lock_writer();
        write_coordinates(first_id + r, row, true, true);
        if (rows >= 1000000 && ((r + 1) % 1000000 == 0 || r + 1 == rows))
          std::clog << "[IVF heap] rows=" << r + 1 << '/' << rows
                    << " elapsed_s="
                    << std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - progress_start)
                           .count()
                    << std::endl;
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
    {
      std::lock_guard lock(_copy_mutex);
      check(!_failed, "IVF coordinate heap failed");
      if (!contains_unlocked(id))
        return false;
    }
    try {
      uint64_t             wait = 0;
      auto                 pins = _pins->acquire(1, wait);
      inplace::NodeRID     rid;
      inplace::PinnedFrame frame;
      try {
        frame = _store.pin_page_for_node(id, inplace::READ, &rid);
        check(frame.valid(), "IVF coordinate page unavailable");
        auto guard = frame.read_guard();
        auto node = guard.read_node(rid);
        check(node.valid() && node.node_id() == id && node.degree() == 0,
              "IVF coordinate slot mismatch");
        std::lock_guard lock(_copy_mutex);
        check(!_failed, "IVF coordinate heap failed");
        if (!contains_unlocked(id))
          return false;
        std::memcpy(output.data(), node.coords_bytes(),
                    size_t(aligned_dim()) * sizeof(T));
        return true;
      } catch (...) {
        poison(std::current_exception());
        throw;
      }
    } catch (...) {
      poison(std::current_exception());
      throw;
    }
  }

  template<typename T>
  HeapIOCounters HeapStorage<T>::io_counters() const {
    const auto& stats = _store.stats();
    return {stats.page_fault_total.load(std::memory_order_relaxed) +
                stats.batch_misses_total.load(std::memory_order_relaxed),
            stats.pages_flushed.load(std::memory_order_relaxed),
            stats.dirty_evictions.load(std::memory_order_relaxed),
            stats.dirty_page_bytes_flushed.load(std::memory_order_relaxed),
            stats.page_in_ns.load(std::memory_order_relaxed) +
                stats.batch_io_ns_total.load(std::memory_order_relaxed),
            stats.eviction_flush_ns.load(std::memory_order_relaxed)};
  }

  template<typename T>
  uint32_t HeapStorage<T>::copy_batch_limit() const {
    // Leave room for a second batch or a writer, without depending on worker
    // count. The aggregate cap remains safe if all requests hash to one shard.
    return std::min(inplace::BufferPool::kPinBatchMax,
                    std::max(1u, _pins->capacity() / 2));
  }

  template<typename T>
  BatchCopyResult HeapStorage<T>::copy_coordinates_batch(
      std::span<const uint32_t> ids, std::span<T> output, size_t stride) {
    check(stride >= aligned_dim() && stride <= SIZE_MAX / sizeof(T) &&
              stride * sizeof(T) % 64 == 0,
          "IVF batch coordinate stride must be padded and 64-byte aligned");
    check(ids.size() <= SIZE_MAX / stride &&
              output.size() >= ids.size() * stride &&
              (ids.empty() ||
               (output.data() != nullptr &&
                reinterpret_cast<uintptr_t>(output.data()) % 64 == 0)),
          "Invalid IVF batch coordinate output");
    BatchCopyResult result;
    result.present.resize(ids.size(), 0);
    const auto reads_before = inplace::thread_physical_reads_count();
    // All requested pages could map to one shard. Keep one spare per shard
    // when possible; one-frame shards necessarily use single-page batches.
    const size_t limit = copy_batch_limit();
    try {
      for (size_t begin = 0; begin < ids.size(); begin += limit) {
        const size_t count = std::min(limit, ids.size() - begin);
        std::array<uint32_t, inplace::BufferPool::kPinBatchMax> requested;
        {
          uint64_t wait = 0;
          auto     lock = detail::timed_mutex_lock(_copy_mutex, wait);
          result.adapter_mutex_wait_ns += wait;
          check(!_failed, "IVF coordinate heap failed");
          for (size_t i = 0; i < count; ++i)
            requested[i] = contains_unlocked(ids[begin + i])
                               ? ids[begin + i]
                               : inplace::INVALID_NODE;
        }
        uint64_t admission_wait = 0;
        auto     pins = _pins->acquire(uint32_t(count), admission_wait);
        result.pin_admission_wait_ns += admission_wait;
        std::array<inplace::PinnedFrame, inplace::BufferPool::kPinBatchMax>
                                                                        frames;
        std::array<inplace::NodeRID, inplace::BufferPool::kPinBatchMax> rids;
        const auto pin_begin = std::chrono::steady_clock::now();
        try {
          _store.pin_pages_for_nodes_batch(requested.data(), count,
                                           inplace::READ, frames.data(),
                                           rids.data());
          result.backend_pin_wall_ns += detail::elapsed_nanoseconds(pin_begin);
          ++result.backend_batch_calls;
          for (size_t i = 0; i < count; ++i) {
            if (requested[i] == inplace::INVALID_NODE)
              continue;
            check(frames[i].valid(), "IVF batch coordinate page unavailable");
            // One guard at a time: duplicate IDs/pages must not recursively
            // acquire the same shared mutex. RAII frames release every pin.
            auto guard = frames[i].read_guard();
            auto node = guard.read_node(rids[i]);
            check(node.valid() && node.node_id() == ids[begin + i] &&
                      node.degree() == 0,
                  "IVF batch coordinate slot mismatch");
            uint64_t wait = 0;
            auto     lock = detail::timed_mutex_lock(_copy_mutex, wait);
            result.adapter_mutex_wait_ns += wait;
            check(!_failed, "IVF coordinate heap failed");
            if (!contains_unlocked(ids[begin + i]))
              continue;
            std::memcpy(output.data() + (begin + i) * stride,
                        node.coords_bytes(), size_t(aligned_dim()) * sizeof(T));
            result.present[begin + i] = 1;
          }
        } catch (...) {
          // Close admission before frames/credits unwind, including cached
          // contenders and validation failures that did not poison the pool.
          poison(std::current_exception());
          throw;
        }
      }
    } catch (...) {
      poison(std::current_exception());
      throw;
    }
    result.heap_pages_read =
        inplace::thread_physical_reads_count() - reads_before;
    return result;
  }

  template<typename T>
  void HeapStorage<T>::checkpoint(const std::string& snapshot_path) {
    auto writer = lock_writer();
    require_writable();
    check(!_mutable_mode,
          "Mutable IVF heap snapshots are unsupported; use writeback only");
    // Checkpoint paths must be distinct from the heap and any existing file.
    // Claim all artifacts before writing so accidental path reuse cannot
    // truncate a heap, another snapshot, or user data.
    check(!std::filesystem::exists(snapshot_path + ".ivf"),
          "IVF checkpoint descriptor already exists");
    claim_artifact(snapshot_path, 0644);
    try {
      _store.flush();
      _store.save_snapshot(snapshot_path);
      const auto meta_size = std::filesystem::file_size(snapshot_path);
      const auto heap_size = std::filesystem::file_size(_heap_path);
      check_snapshot_layout(snapshot_path, _options, sizeof(T), size(),
                            heap_size);
      const auto hash = artifact_hash_file(snapshot_path);
      claim_artifact(snapshot_path + ".ivf", 0644);
      std::ofstream out(snapshot_path + ".ivf", std::ios::binary);
      out.imbue(std::locale::classic());
      out << "IVFHEAP1 " << data_type_name(type()) << ' ' << _options.dim << ' '
          << _options.id_capacity << ' ' << _options.page_size << ' ' << size()
          << ' ' << meta_size << ' ' << heap_size << ' ' << hash << '\n';
      out.close();
      check(bool(out), "IVF checkpoint descriptor write failed");
      _read_only =
          true;  // Immutable artifacts become read-only after checkpointing.
    } catch (...) {
      _failed = true;
      throw;
    }
  }

  template<typename T>
  void HeapStorage<T>::writeback(uint64_t workspace_bytes) {
    auto writer = lock_writer();
    require_writable();
    check(_mutable_mode, "Writeback requires a mutable session");
    check(workspace_bytes >= writeback_workspace_bytes(),
          "IVF writeback exceeds workspace allowance");
    try {
      while (_store.dirty_page_count())
        check(_store.flush_dirty_budget(32) != 0,
              "Quiescent IVF writeback made no progress");
    } catch (...) {
      poison(std::current_exception());
      throw;
    }
  }

  template<typename T>
  std::unique_ptr<HeapStorage<T>> HeapStorage<T>::reopen(
      const HeapOptions& options, const std::string& heap_path,
      const std::string& snapshot_path) {
    auto heap =
        std::unique_ptr<HeapStorage>(new HeapStorage(options, heap_path));
    check(std::filesystem::file_size(snapshot_path + ".ivf") <= 1024,
          "IVF heap descriptor exceeds framing limit");
    std::ifstream in(snapshot_path + ".ivf", std::ios::binary);
    in.imbue(std::locale::classic());
    std::string magic, dtype;
    uint64_t    dim = 0, cap = 0, page = 0, count = 0, meta_size = 0,
             heap_size = 0, hash = 0;
    in >> magic >> dtype >> dim >> cap >> page >> count >> meta_size >>
        heap_size >> hash;
    check(bool(in) && magic == "IVFHEAP1" &&
              dtype == data_type_name(type()) && dim == options.dim &&
              cap == options.id_capacity && page == options.page_size &&
              count <= cap,
          "Invalid or incompatible IVF checkpoint descriptor");
    in >> std::ws;
    check(in.eof(), "Trailing IVF checkpoint descriptor data");
    check(std::filesystem::file_size(snapshot_path) == meta_size &&
              std::filesystem::file_size(heap_path) == heap_size &&
              heap_size % options.page_size == 0 &&
              artifact_hash_file(snapshot_path) == hash,
          "IVF checkpoint file size/hash mismatch");
    check_snapshot_layout(snapshot_path, options, sizeof(T), count, heap_size);
    heap->init();  // Only after existing files/descriptor have been validated.
    heap->_store.load_snapshot(snapshot_path);
    check(heap->size() == count && heap->_store.n_chunks() == 0 &&
              heap->_store.max_degree() == 0,
          "IVF checkpoint state mismatch");
    heap->_read_only = true;
    return heap;
  }

  template class HeapStorage<float>;
  template class HeapStorage<int8_t>;
  template class HeapStorage<uint8_t>;
}  // namespace diskann::ivf
