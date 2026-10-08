#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

#include "bufann/ivf_lists.h"

namespace diskann::ivf {
struct MutableListOptions {
    uint32_t nlist = 0;
    uint32_t chunks = 0;
    uint32_t id_capacity = 0; // Exclusive external-ID upper bound.
    uint32_t first_id = 0;
    uint64_t memory_budget_bytes = 1024ull * 1024 * 1024;
};
struct ListLocation {
    uint32_t list;
    uint32_t slot;
    bool operator==(const ListLocation&) const = default;
};
struct ListSnapshot {
    std::vector<uint32_t> ids;
    std::vector<uint8_t> codes;
    uint64_t capacity_slots = 0;
};
struct MutableListCheckpointInfo {
    uint32_t live = 0, deleted = 0;
    uint64_t missing_deletes = 0, estimated_bytes = 0;
};

// S05 component seam, NOT a heap transaction or persisted NativeIndex. Caller
// supplies already encoded global PQ8 bytes. MutableIndex composes readiness/
// publication and quiescent checkpoints separately; benchmark mutations remain disabled.
// Lists/locator storage is stable; IDs cannot be reused after a successful erase.
class MutablePostingLists {
  public:
    // Move-only per-ID reservation for routing/PQ/heap work performed before
    // list publication. Destruction/cancel or failed commit releases the ID for
    // retry. The list store MUST outlive its guards; a guard is not thread-safe.
    // This rolls back list visibility only, not any caller-owned heap writes.
    class InsertReservation {
      public:
        ~InsertReservation();
        InsertReservation(InsertReservation&& other) noexcept;
        InsertReservation& operator=(InsertReservation&& other) noexcept;
        InsertReservation(const InsertReservation&) = delete;
        InsertReservation& operator=(const InsertReservation&) = delete;
        void commit(uint32_t list, std::span<const uint8_t> code);
        // Composition seam: invoke a short readiness publication after growth/
        // copy succeeds, before list visibility, with owning-list lock held.
        // Callback must not acquire a list lock or perform page I/O. It must
        // either fail without side effects, or return with no later failure.
        void commit_ready(uint32_t list, std::span<const uint8_t> code, const std::function<void()>& make_ready);
        void cancel() noexcept;
      private:
        friend class MutablePostingLists;
        InsertReservation(MutablePostingLists* owner, uint32_t id) : _owner(owner), _id(id) {}
        MutablePostingLists* _owner;
        uint32_t _id;
    };

    static uint64_t estimate_empty_bytes(const MutableListOptions& options);
    static std::unique_ptr<MutablePostingLists> create(const MutableListOptions& options);
    // Borrowed immutable input is copied, not changed. Estimate excludes that
    // caller-owned CSR input/router/codebook, which must be budgeted separately.
    static std::unique_ptr<MutablePostingLists> from_immutable(const PostingLists& input,
        uint64_t memory_budget_bytes = 1024ull * 1024 * 1024);
    // Separate mutable framing: arbitrary membership/order, capacities and
    // tombstones. Caller MUST quiesce operations and reservations during save.
    void save_checkpoint(const std::string& path) const;
    static MutableListCheckpointInfo inspect_checkpoint(const std::string& path, const MutableListOptions& expected);
    static std::unique_ptr<MutablePostingLists> load_checkpoint(const std::string& path, const MutableListOptions& expected);
    bool deleted(uint32_t id) const;
    MutablePostingLists(const MutablePostingLists&) = delete;
    MutablePostingLists& operator=(const MutablePostingLists&) = delete;

    uint32_t nlist() const { return _options.nlist; }
    uint32_t chunks() const { return _options.chunks; }
    uint32_t first_id() const { return _options.first_id; }
    uint32_t id_capacity() const { return _options.id_capacity; }
    uint32_t size() const { return _live_count.load(std::memory_order_acquire); }
    uint32_t pending_insertions() const { return _pending_insertions.load(std::memory_order_acquire); }
    uint64_t missing_deletes() const { return _missing_deletes.load(std::memory_order_relaxed); }
    // Conservative accounted allocations, not RSS/hard allocator limits. Growth
    // reserves both old and replacement arrays before allocation under a short
    // global budget mutex; unrelated lists otherwise mutate independently.
    uint64_t estimated_bytes() const;
    uint64_t peak_estimated_bytes() const;

    // Per-ID CAS reservation -> owning-list exclusive lock -> release publication.
    // Invalid/previously used IDs are rejected; prepublication failures permit
    // retry. No caller buffers/pointers are retained. No callbacks/heap locks.
    InsertReservation reserve_insert(uint32_t id);
    void insert(uint32_t id, uint32_t list, std::span<const uint8_t> code);
    // Missing, deleted or currently reserved insertion: idempotent no-op. A
    // delete overlapping a pending insert can linearize before its publication.
    bool erase(uint32_t id);
    // Advisory copied location, not a durable handle across concurrent swaps.
    std::optional<ListLocation> location(uint32_t id) const;
    bool contains(uint32_t id) const;
    ListSnapshot copy_list(uint32_t list, uint64_t workspace_bytes = 64ull * 1024 * 1024) const;
    // Shared lock only while scanning each list. Return copied candidates; no
    // pointers survive unlock. Multi-list scans are NOT collection snapshots;
    // deletion after scanning requires heap/live revalidation during reranking.
    ListScan scan(std::span<const uint32_t> lists, std::span<const float> distances,
        uint32_t budget, uint64_t workspace_bytes = 64ull * 1024 * 1024) const;

  private:
    enum class State : uint8_t { Absent, Reserved, Live, Deleted };
    struct Locator {
        std::atomic<State> state{State::Absent};
        uint32_t list = UINT32_MAX; // Immutable after Live release publication.
        uint32_t slot = UINT32_MAX; // Protected by owning list's mutex.
    };
    struct List {
        mutable std::shared_mutex mutex;
        std::unique_ptr<uint32_t[]> ids;
        std::unique_ptr<uint8_t[]> codes;
        uint32_t length = 0;
        uint64_t capacity = 0;
    };
    MutablePostingLists(const MutableListOptions& options, const PostingLists* input, bool allocate_buffers = true);
    size_t relative_id(uint32_t id) const;
    List& checked_list(uint32_t list);
    const List& checked_list(uint32_t list) const;
    void grow(List& list); // Owning list exclusive lock already held.
    void publish_reserved(uint32_t id, uint32_t list, std::span<const uint8_t> code,
        const std::function<void()>& make_ready);
    void cancel_reserved(uint32_t id) noexcept;
    void reserve_bytes(uint64_t bytes);
    void release_bytes(uint64_t bytes) noexcept;

    MutableListOptions _options;
    std::unique_ptr<List[]> _lists;
    std::unique_ptr<Locator[]> _locator;
    std::atomic<uint32_t> _live_count{0};
    std::atomic<uint32_t> _pending_insertions{0};
    std::atomic<uint64_t> _missing_deletes{0};
    mutable std::mutex _budget_mutex;
    uint64_t _estimated_bytes = 0;
    uint64_t _peak_estimated_bytes = 0;
};
} // namespace diskann::ivf
