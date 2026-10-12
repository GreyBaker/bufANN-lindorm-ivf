#include "bufann/ivf_heap.h"
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
    template<typename T>
    T read(std::istream& in) {
      T value{};
      in.read(reinterpret_cast<char*>(&value), sizeof(value));
      check(bool(in), "Truncated mutable IVF heap artifact");
      return value;
    }
    template<typename T>
    void write(std::ostream& out, T value) {
      out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    // Validate all legacy metadata before its loader follows array/page
    // indices. Every retained record is physically active; the IVF ledger
    // defines visibility.
    void validate_retained_metadata(const std::string&       path,
                                    const HeapOptions&       options,
                                    size_t                   element_bytes,
                                    std::span<const uint8_t> states,
                                    uint32_t                 retained,
                                    const std::string&       heap_path) {
      const auto heap_bytes = std::filesystem::file_size(heap_path);
      check_snapshot_layout(path, options, element_bytes, retained, heap_bytes);
      std::ifstream in(path, std::ios::binary);
      unsigned char h[112]{};
      in.read(reinterpret_cast<char*>(h), sizeof(h));
      const auto pages = little_u32(h + 56), slots = little_u32(h + 36);
      const auto active_cap = little_u64(h + 72), dirs = little_u64(h + 80),
                 spaces = little_u64(h + 88);
      check(dirs == pages && pages == heap_bytes / options.page_size &&
                little_u32(h + 40) == 0 && little_u32(h + 44) == 0 &&
                little_u64(h + 104) == 0,
            "Invalid mutable IVF heap reserved fields/page directory");
      std::vector<uint8_t>  seen(options.id_capacity, 0);
      std::vector<uint64_t> physical_slots;
      physical_slots.reserve(retained);
      std::vector<uint16_t> occupied(size_t(dirs), 0);
      for (uint32_t r = 0; r < retained; ++r) {
        const auto id = read<uint32_t>(in), page = read<uint32_t>(in);
        const auto slot = read<uint16_t>(in);
        const auto active = read<uint8_t>(in), pq_ready = read<uint8_t>(in);
        check(id < active_cap && page < dirs && slot < slots && active == 1 &&
                  pq_ready == 0 && !seen[id] && states[id] != 0,
              "Invalid/duplicate mutable IVF physical locator");
        seen[id] = 1;
        ++occupied[page];
        physical_slots.push_back(uint64_t(page) * slots + slot);
      }
      std::sort(physical_slots.begin(), physical_slots.end());
      check(std::adjacent_find(physical_slots.begin(), physical_slots.end()) ==
                physical_slots.end(),
            "Aliased mutable IVF physical slots");
      for (uint32_t id = 0; id < options.id_capacity; ++id)
        check(bool(seen[id]) == (states[id] != 0),
              "Mutable IVF ledger/physical membership mismatch");
      for (uint64_t id = 0; id < active_cap; ++id)
        check(read<uint8_t>(in) == seen[id],
              "Mutable IVF physical active flags mismatch");
      for (uint64_t id = 0; id < active_cap; ++id)
        check(read<uint32_t>(in) ==
                  (seen[id] ? uint32_t(id) : inplace::INVALID_TAG),
              "Mutable IVF physical tag mismatch");
      for (uint64_t page = 0; page < dirs; ++page) {
        const auto total = read<uint16_t>(in), used = read<uint16_t>(in);
        check(total == slots && used == occupied[page] && used <= slots,
              "Mutable IVF heap page-directory mismatch");
      }
      std::vector<uint8_t> free_pages(size_t(dirs), 0);
      for (uint64_t p = 0; p < spaces; ++p) {
        const auto page = read<uint32_t>(in);
        check(page < dirs && !free_pages[page] && occupied[page] < slots,
              "Invalid mutable IVF free-page set");
        free_pages[page] = 1;
      }
      for (uint64_t page = 0; page < dirs; ++page)
        check(bool(free_pages[page]) == (occupied[page] < slots),
              "Incomplete mutable IVF free-page set");
      for (uint32_t i = 0; i < 74;
           ++i) {  // Existing 10-entry pool + 64-entry reservoir.
        const auto id = read<uint32_t>(in);
        check(id == inplace::INVALID_NODE ||
                  (id < options.id_capacity && seen[id]),
              "Invalid mutable IVF reservoir ID");
      }
      // Bind allocator state to actual page bitmaps, including spare pages and
      // bitmap tail bits. Otherwise a rehashed ghost slot could break later
      // inserts.
      std::ifstream              heap(heap_path, std::ios::binary);
      std::vector<unsigned char> page(options.page_size);
      size_t                     cursor = 0;
      const auto                 bitmap_bytes = (slots + 7) / 8;
      const auto                 slot_bytes =
          8 + ((uint64_t(options.dim) + 7) / 8 * 8) * element_bytes;
      for (uint32_t pid = 0; pid < pages; ++pid) {
        heap.read(reinterpret_cast<char*>(page.data()), page.size());
        check(bool(heap) && (uint32_t(page[0]) | uint32_t(page[1]) << 8) ==
                                (occupied[pid] ? slots : 0),
              "Mutable IVF heap page-header mismatch");
        for (uint32_t slot = 0; slot < bitmap_bytes * 8; ++slot) {
          const uint64_t key = uint64_t(pid) * slots + slot;
          const bool     expected = slot < slots &&
                                cursor < physical_slots.size() &&
                                physical_slots[cursor] == key;
          check(bool((page[8 + slot / 8] >> (slot % 8)) & 1u) == expected,
                "Mutable IVF heap bitmap/locator mismatch");
          if (expected) {
            const auto* header =
                page.data() + 8 + bitmap_bytes + size_t(slot * slot_bytes);
            check(header[0] == 0 && header[1] == 0 && header[2] == 0 &&
                      header[3] == 0,
                  "Mutable IVF retained slot has edges/flags/padding");
            ++cursor;
          }
        }
      }
      check(cursor == physical_slots.size(),
            "Mutable IVF heap bitmap population mismatch");
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
    check(!_failed,
          "IVF heap build failed; discard it rather than checkpointing");
    check(!_read_only, "Reopened immutable IVF heap is read-only");
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
    std::lock_guard lock(_copy_mutex);
    require_writable();
    check(!_mutable_mode, "Mutable IVF heap requires prepared insertion");
    check(id < _options.id_capacity, "IVF external ID exceeds capacity");
    check(!contains_unlocked(id), "Duplicate IVF external ID");
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
      check(_store.set_node_tag(id, id), "IVF external tag assignment failed");
      if (allocate)
        _store.publish_node(
            id);  // Physical readiness; mutable logical visibility is separate.
    } catch (...) {
      _failed = true;
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
    std::lock_guard lock(_copy_mutex);
    require_writable();
    check(_mutable_mode, "IVF heap is not a mutable session");
    check(id < id_capacity(), "IVF external ID exceeds capacity");
    const auto state = _mutable_states[id];
    check(state == MutableState::Absent || state == MutableState::Aborted,
          "Mutable IVF heap ID already prepared/live/deleted; reuse refused");
    write_coordinates(id, coordinates, state == MutableState::Absent);
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
        // validation, publication and framing stay unchanged; mutable append
        // retains its existing placement policy.
        std::lock_guard lock(_copy_mutex);
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
    std::lock_guard<std::mutex> lock(_copy_mutex);
    if (!contains_unlocked(id))
      return false;
    inplace::NodeRID rid;
    auto             frame = _store.pin_page_for_node(id, inplace::READ, &rid);
    check(frame.valid(), "IVF coordinate page unavailable");
    auto guard = frame.read_guard();
    auto node = guard.read_node(rid);
    check(node.valid() && node.node_id() == id && node.degree() == 0,
          "IVF coordinate slot mismatch");
    std::memcpy(output.data(), node.coords_bytes(),
                size_t(aligned_dim()) * sizeof(T));
    return true;
  }

  template<typename T>
  uint32_t HeapStorage<T>::copy_batch_limit() const {
    const auto shard_frames = _store.min_buffer_shard_frames();
    return std::min(inplace::BufferPool::kPinBatchMax,
                    std::max(1u, shard_frames - 1));
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
    std::lock_guard<std::mutex> lock(_copy_mutex);
    const auto reads_before = inplace::thread_physical_reads_count();
    // All requested pages could map to one shard. Keep one spare per shard
    // when possible; one-frame shards necessarily use single-page batches.
    const size_t limit = copy_batch_limit();
    for (size_t begin = 0; begin < ids.size(); begin += limit) {
      const size_t count = std::min(limit, ids.size() - begin);
      std::array<inplace::PinnedFrame, inplace::BufferPool::kPinBatchMax>
                                                                      frames;
      std::array<inplace::NodeRID, inplace::BufferPool::kPinBatchMax> rids;
      _store.pin_pages_for_nodes_batch(ids.data() + begin, count, inplace::READ,
                                       frames.data(), rids.data());
      ++result.backend_batch_calls;
      for (size_t i = 0; i < count; ++i) {
        if (!contains_unlocked(ids[begin + i]))
          continue;
        check(frames[i].valid(), "IVF batch coordinate page unavailable");
        // One guard at a time: duplicate IDs/pages must not recursively
        // acquire the same shared mutex. RAII frames release every pin.
        auto guard = frames[i].read_guard();
        auto node = guard.read_node(rids[i]);
        check(node.valid() && node.node_id() == ids[begin + i] &&
                  node.degree() == 0,
              "IVF batch coordinate slot mismatch");
        std::memcpy(output.data() + (begin + i) * stride, node.coords_bytes(),
                    size_t(aligned_dim()) * sizeof(T));
        result.present[begin + i] = 1;
      }
    }
    result.heap_pages_read =
        inplace::thread_physical_reads_count() - reads_before;
    return result;
  }

  template<typename T>
  void HeapStorage<T>::checkpoint(const std::string& snapshot_path) {
    require_writable();
    check(!_mutable_mode,
          "Mutable IVF heap checkpoint format is not implemented; no snapshot "
          "published");
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
  void HeapStorage<T>::checkpoint_mutable_copy(const std::string& path,
                                               const std::string& snapshot,
                                               const std::string& state_path) {
    std::lock_guard lock(_copy_mutex);
    require_writable();
    check(_mutable_mode, "Mutable IVF checkpoint requires a mutable heap");
    uint32_t live = 0, retained = 0;
    for (uint32_t id = 0; id < id_capacity(); ++id) {
      const auto state = _mutable_states[id];
      check(state != MutableState::Prepared,
            "Mutable IVF checkpoint has prepared inserts");
      live += state == MutableState::Live;
      retained += state != MutableState::Absent;
      check((state != MutableState::Absent) == _store.is_active(id),
            "Mutable IVF checkpoint physical-state mismatch");
    }
    check(live == size() && retained == retained_size(),
          "Mutable IVF checkpoint heap-count mismatch");
    try {
      _store.flush();
    } catch (...) {
      _failed = true;
      throw;
    }
    // The output is a NEW immutable family, not a rewrite/freeze of this
    // session.
    claim_artifact(path, 0644);
    {
      std::ifstream input(_heap_path, std::ios::binary);
      std::ofstream output(path, std::ios::binary);
      check(bool(input) && bool(output), "Cannot copy mutable IVF heap");
      char bytes[8192];
      while (input.read(bytes, sizeof(bytes)) || input.gcount())
        output.write(bytes, input.gcount());
      check(input.eof() && !input.bad(), "Mutable IVF heap copy read failure");
      output.close();
      check(bool(output), "Mutable IVF heap copy write failure");
    }
    claim_artifact(snapshot, 0644);
    _store.save_snapshot(snapshot);
    const auto heap_bytes = std::filesystem::file_size(path),
               meta_bytes = std::filesystem::file_size(snapshot);
    check_snapshot_layout(snapshot, _options, sizeof(T), retained, heap_bytes);
    claim_artifact(snapshot + ".ivf", 0644);
    std::ofstream descriptor(snapshot + ".ivf", std::ios::binary);
    descriptor.imbue(std::locale::classic());
    // Distinct magic prevents the immutable adapter from treating retained
    // tombstones/aborted records as live. The mandatory ledger defines
    // visibility.
    descriptor << "IVFHEAPM1 " << data_type_name(type()) << ' ' << dim() << ' '
               << id_capacity() << ' ' << _options.page_size << ' ' << retained
               << ' ' << meta_bytes << ' ' << heap_bytes << ' '
               << artifact_hash_file(snapshot) << '\n';
    descriptor.close();
    check(bool(descriptor), "Mutable IVF heap descriptor write failure");
    claim_artifact(state_path, 0644);
    std::ofstream state(state_path, std::ios::binary);
    state.write("IVFMHS1\0", 8);
    for (uint32_t value : {1u, uint32_t(type()), dim(), id_capacity(),
                           _options.page_size, live, retained})
      write(state, value);
    for (auto value : _mutable_states)
      write(state, uint8_t(value));
    state.close();
    check(bool(state), "Mutable IVF heap ledger write failure");
    const auto bytes = 36ull + id_capacity();
    check(std::filesystem::file_size(state_path) == bytes,
          "Mutable IVF heap ledger framing mismatch");
    const auto    hash = artifact_hash_prefix(state_path, bytes);
    std::ofstream trailer(state_path, std::ios::binary | std::ios::app);
    write(trailer, hash);
    trailer.close();
    check(bool(trailer), "Mutable IVF heap ledger checksum write failure");
  }

  template<typename T>
  std::unique_ptr<HeapStorage<T>>
  HeapStorage<T>::reopen_mutable_checkpoint_copy(
      const HeapOptions& options, const std::string& path,
      const std::string& snapshot, const std::string& state_path) {
    check(std::filesystem::file_size(state_path) == 44ull + options.id_capacity,
          "Invalid mutable IVF heap ledger length");
    std::ifstream in(state_path, std::ios::binary);
    char          magic[8]{};
    in.read(magic, sizeof(magic));
    check(bool(in) && std::equal(magic, magic + 8, "IVFMHS1\0") &&
              read<uint32_t>(in) == 1 &&
              read<uint32_t>(in) == uint32_t(type()) &&
              read<uint32_t>(in) == options.dim &&
              read<uint32_t>(in) == options.id_capacity &&
              read<uint32_t>(in) == options.page_size,
          "Invalid/incompatible mutable IVF heap ledger header");
    const auto expected_live = read<uint32_t>(in),
               expected_retained = read<uint32_t>(in);
    check(expected_live <= expected_retained &&
              expected_retained <= options.id_capacity,
          "Invalid mutable IVF heap ledger counts");
    std::vector<uint8_t> states(options.id_capacity);
    uint32_t             live = 0, retained = 0;
    for (auto& state : states) {
      state = read<uint8_t>(in);
      check(state == uint8_t(MutableState::Absent) ||
                state == uint8_t(MutableState::Live) ||
                state == uint8_t(MutableState::Deleted) ||
                state == uint8_t(MutableState::Aborted),
            "Invalid/pending mutable IVF heap ledger state");
      live += state == uint8_t(MutableState::Live);
      retained += state != uint8_t(MutableState::Absent);
    }
    check(live == expected_live && retained == expected_retained &&
              read<uint64_t>(in) ==
                  artifact_hash_prefix(state_path, 36ull + options.id_capacity),
          "Mutable IVF heap ledger counts/checksum mismatch");
    validate_retained_metadata(snapshot, options, sizeof(T), states, retained,
                               path);
    auto heap = reopen_physical(options, path, snapshot, "IVFHEAPM1");
    heap->_mutable_states.resize(options.id_capacity);
    for (uint32_t id = 0; id < options.id_capacity; ++id)
      heap->_mutable_states[id] = static_cast<MutableState>(states[id]);
    heap->_mutable_mode = true;
    heap->_read_only = false;
    heap->_live_count = live;
    heap->validate_retained_coordinates();
    return heap;
  }

  template<typename T>
  void HeapStorage<T>::validate_retained_coordinates() {
    std::lock_guard lock(_copy_mutex);
    for (uint32_t id = 0; id < id_capacity(); ++id)
      if (_mutable_states[id] != MutableState::Absent) {
        inplace::NodeRID rid;
        auto frame = _store.pin_page_for_node(id, inplace::READ, &rid);
        check(frame.valid(),
              "Mutable IVF retained coordinate page unavailable");
        auto guard = frame.read_guard();
        auto node = guard.read_node(rid);
        check(node.valid() && node.node_id() == id && node.degree() == 0,
              "Mutable IVF retained coordinate slot mismatch");
        for (uint32_t d = 0; d < aligned_dim(); ++d) {
          T value{};
          std::memcpy(&value, node.coords_bytes() + size_t(d) * sizeof(T),
                      sizeof(T));
          check(std::isfinite(double(value)) && (d < dim() || value == T{}),
                "Invalid mutable IVF retained coordinate/padding");
        }
      }
  }

  template<typename T>
  std::unique_ptr<HeapStorage<T>> HeapStorage<T>::reopen(
      const HeapOptions& options, const std::string& heap_path,
      const std::string& snapshot_path) {
    return reopen_physical(options, heap_path, snapshot_path, "IVFHEAP1");
  }
  template<typename T>
  std::unique_ptr<HeapStorage<T>> HeapStorage<T>::reopen_physical(
      const HeapOptions& options, const std::string& heap_path,
      const std::string& snapshot_path, const char* descriptor_magic) {
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
    check(bool(in) && magic == descriptor_magic &&
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
