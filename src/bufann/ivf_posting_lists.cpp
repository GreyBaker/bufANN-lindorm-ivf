#include "bufann/ivf_posting_lists.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include "bufann/ivf_artifacts.h"

namespace diskann::ivf {
  namespace {
    void check(bool ok, const char* message) {
      if (!ok)
        throw std::invalid_argument(message);
    }
    uint64_t mul(uint64_t a, uint64_t b) {
      check(!b || a <= UINT64_MAX / b, "Mutable IVF list workspace overflow");
      return a * b;
    }
    void add(uint64_t& a, uint64_t b) {
      check(a <= UINT64_MAX - b, "Mutable IVF list workspace overflow");
      a += b;
    }
    void validate(const MutableListOptions& options) {
      check(options.nlist > 0 && options.nlist <= INT32_MAX &&
                options.chunks > 0 && options.chunks <= INT32_MAX &&
                options.first_id < options.id_capacity &&
                options.memory_budget_bytes > 0,
            "Invalid mutable IVF list options");
    }
    uint64_t buffer_bytes(uint64_t capacity, uint32_t chunks) {
      uint64_t bytes = mul(capacity, uint64_t(chunks) + sizeof(uint32_t));
      add(bytes,
          128);  // Two allocations' conservative header/alignment allowance.
      check(bytes <= SIZE_MAX,
            "Mutable IVF list buffer exceeds addressable memory");
      return bytes;
    }
    bool nearer(const PQCandidate& a, const PQCandidate& b) {
      return a.squared_distance < b.squared_distance ||
             (a.squared_distance == b.squared_distance && a.id < b.id);
    }
    template<typename T>
    T read(std::istream& in) {
      T value{};
      in.read(reinterpret_cast<char*>(&value), sizeof(value));
      check(bool(in), "Truncated mutable IVF lists");
      return value;
    }
    template<typename T>
    void write(std::ostream& out, T value) {
      out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    constexpr uint64_t        mutable_header_bytes = 44;
    MutableListCheckpointInfo checkpoint_header(
        std::istream& in, const MutableListOptions& expected) {
      validate(expected);
      char magic[8]{};
      in.read(magic, sizeof(magic));
      check(bool(in) && std::equal(magic, magic + 8, "IVFMPL1\0") &&
                read<uint32_t>(in) == 1,
            "Invalid mutable IVF list header/version");
      check(read<uint32_t>(in) == expected.nlist &&
                read<uint32_t>(in) == expected.chunks &&
                read<uint32_t>(in) == expected.first_id &&
                read<uint32_t>(in) == expected.id_capacity,
            "Mutable IVF list shape mismatch");
      MutableListCheckpointInfo result;
      result.live = read<uint32_t>(in);
      result.deleted = read<uint32_t>(in);
      result.missing_deletes = read<uint64_t>(in);
      check(uint64_t(result.live) + result.deleted <=
                uint64_t(expected.id_capacity) - expected.first_id,
            "Invalid mutable IVF list counts");
      return result;
    }
    uint64_t checkpoint_file_bytes(const MutableListOptions& options,
                                   uint32_t                  live) {
      uint64_t bytes = mutable_header_bytes + 8 +
                       uint64_t(options.id_capacity) - options.first_id;
      add(bytes, mul(options.nlist, 12));
      add(bytes, mul(live, uint64_t(options.chunks) + 4));
      return bytes;
    }
  }  // namespace

  uint64_t MutablePostingLists::estimate_empty_bytes(
      const MutableListOptions& options) {
    validate(options);
    uint64_t bytes = sizeof(MutablePostingLists) + 65536;
    add(bytes, mul(options.nlist, sizeof(List)));
    add(bytes,
        mul(uint64_t(options.id_capacity) - options.first_id, sizeof(Locator)));
    add(bytes, mul(options.nlist, buffer_bytes(16, options.chunks)));
    check(bytes <= SIZE_MAX, "Mutable IVF lists exceed addressable memory");
    return bytes;
  }

  std::unique_ptr<MutablePostingLists> MutablePostingLists::create(
      const MutableListOptions& options) {
    return std::unique_ptr<MutablePostingLists>(
        new MutablePostingLists(options, nullptr));
  }

  std::unique_ptr<MutablePostingLists> MutablePostingLists::from_immutable(
      const PostingLists& input, uint64_t budget) {
    check(input.size() > 0,
          "Cannot convert empty/moved-from immutable IVF lists");
    return std::unique_ptr<MutablePostingLists>(
        new MutablePostingLists({input.nlist(), input.chunks(),
                                 input.id_capacity(), input.first_id(), budget},
                                &input));
  }

  MutablePostingLists::MutablePostingLists(const MutableListOptions& options,
                                           const PostingLists*       input,
                                           bool allocate_buffers)
      : _options(options) {
    uint64_t bytes = estimate_empty_bytes(options);
    if (input)
      add(bytes,
          mul(input->size(), uint64_t(options.chunks) + sizeof(uint32_t)));
    if (!allocate_buffers)
      bytes -= mul(options.nlist, buffer_bytes(16, options.chunks));
    check(bytes <= options.memory_budget_bytes && bytes <= SIZE_MAX,
          "Mutable IVF lists exceed declared memory estimate budget");
    _estimated_bytes = _peak_estimated_bytes = bytes;
    _lists = std::make_unique<List[]>(options.nlist);
    _locator = std::make_unique<Locator[]>(uint64_t(options.id_capacity) -
                                           options.first_id);
    if (!allocate_buffers)
      return;
    for (uint32_t c = 0; c < options.nlist; ++c) {
      auto&      list = _lists[c];
      const auto ids = input ? input->ids(c) : std::span<const uint32_t>{};
      const auto codes = input ? input->codes(c) : std::span<const uint8_t>{};
      list.capacity = uint64_t(ids.size()) + 16;
      buffer_bytes(list.capacity,
                   options.chunks);  // Checked before array allocation.
      list.ids =
          std::make_unique<uint32_t[]>(static_cast<size_t>(list.capacity));
      list.codes = std::make_unique<uint8_t[]>(
          static_cast<size_t>(mul(list.capacity, options.chunks)));
      check(codes.size() == ids.size() * uint64_t(options.chunks),
            "Invalid immutable IVF list code shape");
      if (!ids.empty()) {
        std::copy(ids.begin(), ids.end(), list.ids.get());
        std::copy(codes.begin(), codes.end(), list.codes.get());
      }
      list.length = static_cast<uint32_t>(ids.size());
      for (uint32_t s = 0; s < list.length; ++s) {
        auto& loc = _locator[relative_id(list.ids[s])];
        check(loc.state.load(std::memory_order_relaxed) == State::Absent,
              "Duplicate immutable IVF list ID");
        loc.list = c;
        loc.slot = s;
        loc.state.store(State::Live, std::memory_order_release);
      }
    }
    _live_count.store(input ? input->size() : 0, std::memory_order_release);
  }

  size_t MutablePostingLists::relative_id(uint32_t id) const {
    check(id >= first_id() && id < id_capacity(),
          "Mutable IVF external ID out of range");
    return size_t(id - first_id());
  }
  MutablePostingLists::List& MutablePostingLists::checked_list(uint32_t list) {
    check(list < nlist(), "Invalid mutable IVF list ID");
    return _lists[list];
  }
  const MutablePostingLists::List& MutablePostingLists::checked_list(
      uint32_t list) const {
    check(list < nlist(), "Invalid mutable IVF list ID");
    return _lists[list];
  }

  void MutablePostingLists::reserve_bytes(uint64_t bytes) {
    std::lock_guard lock(_budget_mutex);
    check(bytes <= _options.memory_budget_bytes - _estimated_bytes,
          "Mutable IVF list growth exceeds declared memory estimate budget");
    _estimated_bytes += bytes;
    _peak_estimated_bytes = std::max(_peak_estimated_bytes, _estimated_bytes);
  }
  void MutablePostingLists::release_bytes(uint64_t bytes) noexcept {
    std::lock_guard lock(_budget_mutex);
    _estimated_bytes -= bytes;
  }
  uint64_t MutablePostingLists::estimated_bytes() const {
    std::lock_guard lock(_budget_mutex);
    return _estimated_bytes;
  }
  uint64_t MutablePostingLists::peak_estimated_bytes() const {
    std::lock_guard lock(_budget_mutex);
    return _peak_estimated_bytes;
  }

  void MutablePostingLists::grow(List& list) {
    uint64_t capacity = list.capacity;
    add(capacity, 16);  // Exact +16 policy, not std::vector's geometric growth.
    const auto replacement = buffer_bytes(capacity, chunks());
    const auto previous = buffer_bytes(list.capacity, chunks());
    reserve_bytes(
        replacement);  // Old and new arrays can coexist on many lists.
    try {
      auto ids = std::make_unique<uint32_t[]>(static_cast<size_t>(capacity));
      auto codes = std::make_unique<uint8_t[]>(
          static_cast<size_t>(mul(capacity, chunks())));
      std::copy_n(list.ids.get(), list.length, ids.get());
      std::copy_n(list.codes.get(), size_t(list.length) * chunks(),
                  codes.get());
      list.ids.swap(ids);
      list.codes.swap(codes);
      list.capacity = capacity;
      ids.reset();
      codes.reset();  // Release actual old arrays before accounting.
    } catch (...) {
      release_bytes(replacement);
      throw;
    }
    release_bytes(previous);
  }

  MutablePostingLists::InsertReservation MutablePostingLists::reserve_insert(
      uint32_t id) {
    auto& loc = _locator[relative_id(id)];
    State expected = State::Absent;
    check(loc.state.compare_exchange_strong(expected, State::Reserved,
                                            std::memory_order_acq_rel),
          "Mutable IVF ID already live, reserved or deleted; reuse is refused");
    _pending_insertions.fetch_add(1, std::memory_order_release);
    return InsertReservation(this, id);
  }
  void MutablePostingLists::cancel_reserved(uint32_t id) noexcept {
    // Decrement before allowing another reservation, avoiding a transient
    // double count of this ID when canceled operations are immediately retried.
    _pending_insertions.fetch_sub(1, std::memory_order_release);
    _locator[id - first_id()].state.store(State::Absent,
                                          std::memory_order_release);
  }
  MutablePostingLists::InsertReservation::~InsertReservation() {
    cancel();
  }
  MutablePostingLists::InsertReservation::InsertReservation(
      InsertReservation&& other) noexcept
      : _owner(std::exchange(other._owner, nullptr)), _id(other._id) {
  }
  MutablePostingLists::InsertReservation&
  MutablePostingLists::InsertReservation::operator=(
      InsertReservation&& other) noexcept {
    if (this != &other) {
      cancel();
      _owner = std::exchange(other._owner, nullptr);
      _id = other._id;
    }
    return *this;
  }
  void MutablePostingLists::InsertReservation::cancel() noexcept {
    if (auto* owner = std::exchange(_owner, nullptr))
      owner->cancel_reserved(_id);
  }
  void MutablePostingLists::InsertReservation::commit(
      uint32_t list, std::span<const uint8_t> code) {
    commit_ready(list, code, {});
  }
  void MutablePostingLists::InsertReservation::commit_ready(
      uint32_t list, std::span<const uint8_t> code,
      const std::function<void()>& make_ready) {
    check(_owner != nullptr,
          "Mutable IVF insertion reservation already consumed");
    auto* owner = std::exchange(_owner, nullptr);
    try {
      owner->publish_reserved(_id, list, code, make_ready);
    } catch (...) {
      owner->cancel_reserved(_id);
      throw;
    }
  }
  void MutablePostingLists::insert(uint32_t id, uint32_t owner,
                                   std::span<const uint8_t> code) {
    reserve_insert(id).commit(owner, code);
  }
  void MutablePostingLists::publish_reserved(
      uint32_t id, uint32_t owner, std::span<const uint8_t> code,
      const std::function<void()>& make_ready) {
    auto& loc = _locator[relative_id(id)];
    auto& list = checked_list(owner);
    check(code.size() == chunks(), "Invalid mutable IVF insert code length");
    std::unique_lock lock(list.mutex);
    if (list.length == list.capacity)
      grow(list);
    check(list.length < UINT32_MAX, "Mutable IVF list length overflow");
    const auto slot = list.length;
    list.ids[slot] = id;
    std::copy(code.begin(), code.end(),
              list.codes.get() + size_t(slot) * chunks());
    if (make_ready)
      make_ready();  // List length/state still unchanged if this throws.
    // No throwing operations remain after readiness returns successfully.
    loc.list = owner;
    loc.slot = slot;
    ++list.length;
    _live_count.fetch_add(1, std::memory_order_release);
    loc.state.store(State::Live, std::memory_order_release);
    _pending_insertions.fetch_sub(1, std::memory_order_release);
  }

  bool MutablePostingLists::erase(uint32_t id) {
    auto& loc = _locator[relative_id(id)];
    if (loc.state.load(std::memory_order_acquire) != State::Live) {
      _missing_deletes.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    // A Live observation acquires the owner's one-time publication. Owner never
    // changes afterward (including after deletion); slot needs the list lock.
    auto&            list = checked_list(loc.list);
    std::unique_lock lock(list.mutex);
    if (loc.state.load(std::memory_order_acquire) != State::Live) {
      _missing_deletes.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    check(loc.slot < list.length && list.ids[loc.slot] == id,
          "Mutable IVF locator mismatch");
    const uint32_t last = list.length - 1;
    if (loc.slot != last) {
      const auto moved = list.ids[last];
      list.ids[loc.slot] = moved;
      std::copy_n(list.codes.get() + size_t(last) * chunks(), chunks(),
                  list.codes.get() + size_t(loc.slot) * chunks());
      _locator[relative_id(moved)].slot = loc.slot;  // Same owning-list lock.
    }
    --list.length;
    loc.state.store(State::Deleted, std::memory_order_release);
    _live_count.fetch_sub(1, std::memory_order_release);
    return true;
  }

  std::optional<ListLocation> MutablePostingLists::location(uint32_t id) const {
    const auto& loc = _locator[relative_id(id)];
    if (loc.state.load(std::memory_order_acquire) != State::Live)
      return std::nullopt;
    const auto&      list = checked_list(loc.list);
    std::shared_lock lock(list.mutex);
    if (loc.state.load(std::memory_order_acquire) != State::Live)
      return std::nullopt;
    check(loc.slot < list.length && list.ids[loc.slot] == id,
          "Mutable IVF locator mismatch");
    return ListLocation{loc.list, loc.slot};
  }
  bool MutablePostingLists::contains(uint32_t id) const {
    return _locator[relative_id(id)].state.load(std::memory_order_acquire) ==
           State::Live;
  }
  bool MutablePostingLists::deleted(uint32_t id) const {
    return _locator[relative_id(id)].state.load(std::memory_order_acquire) ==
           State::Deleted;
  }

  void MutablePostingLists::save_checkpoint(const std::string& path) const {
    static_assert(std::endian::native == std::endian::little);
    check(pending_insertions() == 0,
          "Mutable IVF checkpoint requires no pending reservations");
    uint32_t live = 0, deleted_count = 0;
    for (uint32_t id = first_id(); id < id_capacity(); ++id) {
      const auto state = _locator[id - first_id()].state.load();
      check(state != State::Reserved,
            "Mutable IVF checkpoint contains a reserved ID");
      live += state == State::Live;
      deleted_count += state == State::Deleted;
    }
    check(live == size(), "Mutable IVF checkpoint live-count mismatch");
    claim_artifact(path);
    std::ofstream out(path, std::ios::binary);
    out.write("IVFMPL1\0", 8);
    for (uint32_t value : {1u, nlist(), chunks(), first_id(), id_capacity(),
                           live, deleted_count})
      write(out, value);
    write(out, missing_deletes());
    for (uint32_t id = first_id(); id < id_capacity(); ++id)
      write(out, uint8_t(_locator[id - first_id()].state.load()));
    uint64_t entries = 0;
    for (uint32_t c = 0; c < nlist(); ++c) {
      const auto&      list = _lists[c];
      std::shared_lock lock(list.mutex);
      write(out, list.length);
      write(out, list.capacity);
      for (uint32_t s = 0; s < list.length; ++s) {
        const auto  id = list.ids[s];
        const auto& loc = _locator[relative_id(id)];
        check(loc.state.load() == State::Live && loc.list == c && loc.slot == s,
              "Mutable IVF checkpoint locator mismatch");
        write(out, id);
      }
      out.write(reinterpret_cast<const char*>(list.codes.get()),
                size_t(list.length) * chunks());
      entries += list.length;
    }
    out.close();
    check(bool(out) && entries == live,
          "Mutable IVF list checkpoint write/population failure");
    const auto bytes = checkpoint_file_bytes(_options, live);
    check(std::filesystem::file_size(path) == bytes - 8,
          "Mutable IVF list checkpoint length mismatch");
    const auto    hash = artifact_hash_prefix(path, bytes - 8);
    std::ofstream trailer(path, std::ios::binary | std::ios::app);
    write(trailer, hash);
    trailer.close();
    check(bool(trailer), "Mutable IVF list checksum write failure");
  }

  MutableListCheckpointInfo MutablePostingLists::inspect_checkpoint(
      const std::string& path, const MutableListOptions& expected) {
    static_assert(std::endian::native == std::endian::little);
    std::ifstream in(path, std::ios::binary);
    auto          info = checkpoint_header(in, expected);
    const auto    bytes = checkpoint_file_bytes(expected, info.live);
    check(bytes <= INT64_MAX && std::filesystem::file_size(path) == bytes,
          "Invalid mutable IVF list checkpoint length");
    in.seekg(static_cast<std::streamoff>(bytes - 8));
    check(read<uint64_t>(in) == artifact_hash_prefix(path, bytes - 8),
          "Mutable IVF list checksum mismatch");
    in.seekg(mutable_header_bytes);
    uint32_t live = 0, deleted_count = 0;
    for (uint32_t id = expected.first_id; id < expected.id_capacity; ++id) {
      const auto state = read<uint8_t>(in);
      check(state == uint8_t(State::Absent) || state == uint8_t(State::Live) ||
                state == uint8_t(State::Deleted),
            "Invalid/pending mutable IVF checkpoint state");
      live += state == uint8_t(State::Live);
      deleted_count += state == uint8_t(State::Deleted);
    }
    check(live == info.live && deleted_count == info.deleted,
          "Mutable IVF checkpoint state counts mismatch");
    info.estimated_bytes =
        estimate_empty_bytes(expected) -
        mul(expected.nlist, buffer_bytes(16, expected.chunks));
    uint64_t entries = 0;
    for (uint32_t c = 0; c < expected.nlist; ++c) {
      const auto length = read<uint32_t>(in);
      const auto capacity = read<uint64_t>(in);
      check(length <= live && capacity >= std::max<uint64_t>(16, length) &&
                capacity <=
                    uint64_t(expected.id_capacity) - expected.first_id + 16,
            "Invalid mutable IVF checkpoint capacity");
      add(info.estimated_bytes, buffer_bytes(capacity, expected.chunks));
      entries += length;
      const auto payload = mul(length, uint64_t(expected.chunks) + 4);
      check(uint64_t(in.tellg()) <= bytes - 8 &&
                payload <= bytes - 8 - uint64_t(in.tellg()),
            "Mutable IVF list payload exceeds framing");
      in.seekg(static_cast<std::streamoff>(payload), std::ios::cur);
    }
    check(entries == live && uint64_t(in.tellg()) == bytes - 8 &&
              info.estimated_bytes <= SIZE_MAX,
          "Mutable IVF checkpoint list totals/framing mismatch");
    return info;
  }

  std::unique_ptr<MutablePostingLists> MutablePostingLists::load_checkpoint(
      const std::string& path, const MutableListOptions& expected) {
    const auto info = inspect_checkpoint(path, expected);
    check(info.estimated_bytes <= expected.memory_budget_bytes,
          "Mutable IVF checkpoint exceeds list allocation allowance");
    auto result = std::unique_ptr<MutablePostingLists>(
        new MutablePostingLists(expected, nullptr, false));
    std::ifstream in(path, std::ios::binary);
    checkpoint_header(in, expected);
    for (uint32_t id = expected.first_id; id < expected.id_capacity; ++id) {
      const auto state = static_cast<State>(read<uint8_t>(in));
      result->_locator[id - expected.first_id].state.store(
          state == State::Live ? State::Reserved : state);
    }
    for (uint32_t c = 0; c < expected.nlist; ++c) {
      auto& list = result->_lists[c];
      list.length = read<uint32_t>(in);
      list.capacity = read<uint64_t>(in);
      result->reserve_bytes(buffer_bytes(list.capacity, expected.chunks));
      list.ids = std::make_unique<uint32_t[]>(size_t(list.capacity));
      list.codes = std::make_unique<uint8_t[]>(
          size_t(mul(list.capacity, expected.chunks)));
      for (uint32_t s = 0; s < list.length; ++s) {
        const auto id = read<uint32_t>(in);
        auto&      loc = result->_locator[result->relative_id(id)];
        check(loc.state.load() == State::Reserved,
              "Duplicate/non-live mutable IVF checkpoint ID");
        list.ids[s] = id;
        loc.list = c;
        loc.slot = s;
        loc.state.store(State::Live);
      }
      in.read(reinterpret_cast<char*>(list.codes.get()),
              size_t(list.length) * expected.chunks);
      check(bool(in), "Truncated mutable IVF checkpoint codes");
    }
    for (uint32_t id = expected.first_id; id < expected.id_capacity; ++id)
      check(result->_locator[id - expected.first_id].state.load() !=
                State::Reserved,
            "Mutable IVF live locator has no list entry");
    result->_live_count = info.live;
    result->_missing_deletes = info.missing_deletes;
    check(result->estimated_bytes() == info.estimated_bytes,
          "Mutable IVF checkpoint allocation estimate mismatch");
    return result;
  }
  ListSnapshot MutablePostingLists::copy_list(uint32_t owner,
                                              uint64_t workspace) const {
    const auto&      list = checked_list(owner);
    std::shared_lock lock(list.mutex);
    uint64_t bytes = mul(list.length, uint64_t(chunks()) + sizeof(uint32_t));
    add(bytes, 1024);
    check(bytes <= workspace && bytes <= SIZE_MAX,
          "Mutable IVF list copy exceeds workspace allowance");
    return {
        {list.ids.get(), list.ids.get() + list.length},
        {list.codes.get(), list.codes.get() + size_t(list.length) * chunks()},
        list.capacity};
  }

  ListScan MutablePostingLists::scan(std::span<const uint32_t> owners,
                                     std::span<const float>    distances,
                                     uint32_t                  budget,
                                     uint64_t workspace) const {
    check(budget > 0 && distances.size() == uint64_t(chunks()) * 256 &&
              owners.size() <= nlist(),
          "Invalid mutable IVF scan buffers/budget");
    // Use the full possible ID range, not a racing current-live-count estimate:
    // inserts can increase the scan's candidate count after this preflight.
    const uint32_t limit = std::min(budget, id_capacity() - first_id());
    uint64_t       bytes = mul(limit, 48);
    add(bytes, mul(owners.size(), 4));
    add(bytes, 1024);
    check(bytes <= workspace && bytes <= SIZE_MAX,
          "Mutable IVF scan exceeds workspace allowance");
    for (float distance : distances)
      check(std::isfinite(distance) && distance >= 0,
            "Invalid mutable IVF ADC distance table");
    std::vector<uint32_t> selected(owners.begin(), owners.end());
    std::sort(selected.begin(), selected.end());
    check(selected.empty() || selected.back() < nlist(),
          "Invalid mutable IVF scan list ID");
    check(
        std::adjacent_find(selected.begin(), selected.end()) == selected.end(),
        "Duplicate mutable IVF scan list ID");
    std::vector<PQCandidate> best;
    best.reserve(limit);
    ListScan result;
    for (const auto owner : owners) {
      const auto&      list = checked_list(owner);
      std::shared_lock lock(list.mutex);
      for (uint32_t r = 0; r < list.length; ++r) {
        float distance = 0;
        for (uint32_t c = 0; c < chunks(); ++c)
          distance +=
              distances[size_t(c) * 256 + list.codes[size_t(r) * chunks() + c]];
        check(std::isfinite(distance), "Mutable IVF ADC sum overflow");
        const PQCandidate candidate{list.ids[r], distance};
        if (best.size() < limit) {
          best.push_back(candidate);
          std::push_heap(best.begin(), best.end(), nearer);
        } else if (nearer(candidate, best.front())) {
          std::pop_heap(best.begin(), best.end(), nearer);
          best.back() = candidate;
          std::push_heap(best.begin(), best.end(), nearer);
        }
      }
      result.scanned_codes += list.length;
    }
    result.candidates.resize(best.size());
    for (size_t i = best.size(); i > 0; --i) {
      std::pop_heap(best.begin(), best.end(), nearer);
      result.candidates[i - 1] = best.back();
      best.pop_back();
    }
    return result;
  }
}  // namespace diskann::ivf
