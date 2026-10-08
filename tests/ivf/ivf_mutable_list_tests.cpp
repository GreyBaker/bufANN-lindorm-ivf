#include "bufann/ivf_posting_lists.h"
#include "bufann/ivf_artifacts.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace {
using namespace diskann::ivf;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<typename F> void rejects(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument rejection");
}
std::array<uint8_t, 3> code(uint32_t id) {
    return {uint8_t(id * 7), uint8_t(id * 11 + 1), uint8_t(id * 13 + 2)};
}
std::vector<float> table() {
    std::vector<float> result(3 * 256);
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t i = 0; i < 256; ++i) result[c * 256 + i] = float((int(i) - int(c * 13 + 70)) * (int(i) - int(c * 13 + 70)));
    return result;
}
float score(uint32_t id, const std::vector<float>& distances) {
    auto bytes = code(id); float result = 0;
    for (uint32_t c = 0; c < bytes.size(); ++c) result += distances[c * 256 + bytes[c]];
    return result;
}
bool nearer(const PQCandidate& a, const PQCandidate& b) {
    return a.squared_distance < b.squared_distance ||
        (a.squared_distance == b.squared_distance && a.id < b.id);
}
void audit(const MutablePostingLists& lists) {
    require(lists.pending_insertions() == 0, "quiescent audit found leaked reservation");
    std::vector<uint8_t> seen(lists.id_capacity() - lists.first_id());
    uint32_t total = 0;
    for (uint32_t c = 0; c < lists.nlist(); ++c) {
        const auto copy = lists.copy_list(c);
        require(copy.ids.size() <= copy.capacity_slots && copy.codes.size() == copy.ids.size() * lists.chunks(), "copy shape mismatch");
        for (uint32_t s = 0; s < copy.ids.size(); ++s) {
            const auto id = copy.ids[s];
            require(id >= lists.first_id() && id < lists.id_capacity(), "out-of-range list member");
            require(!seen[id - lists.first_id()], "duplicate list member"); seen[id - lists.first_id()] = 1;
            require(lists.contains(id) && lists.location(id) == ListLocation{c, s}, "live locator mismatch");
            const auto bytes = code(id);
            require(std::equal(bytes.begin(), bytes.end(), copy.codes.begin() + size_t(s) * lists.chunks()), "moved/copied code bytes changed");
            ++total;
        }
    }
    require(total == lists.size(), "live-count/list sum mismatch");
    for (uint32_t r = 0; r < seen.size(); ++r)
        require(lists.contains(lists.first_id() + r) == bool(seen[r]), "locator state differs from membership");
}

void basic() {
    const MutableListOptions options{3, 3, 200, 100};
    auto lists = MutablePostingLists::create(options);
    require(lists->size() == 0 && lists->estimated_bytes() == MutablePostingLists::estimate_empty_bytes(options), "empty estimate/count mismatch");
    for (uint32_t c = 0; c < 3; ++c) require(lists->copy_list(c).capacity_slots == 16, "empty list lacks 16 spare slots");
    for (uint32_t id = 100; id < 140; ++id) {
        auto bytes = code(id); lists->insert(id, 0, bytes); bytes.fill(0);
        const auto capacity = lists->copy_list(0).capacity_slots;
        require(capacity == (id < 116 ? 16 : id < 132 ? 32 : 48), "growth was not exactly +16");
        audit(*lists);
    }
    const auto accounted = lists->estimated_bytes();
    require(lists->erase(100) && lists->location(139) == ListLocation{0, 0}, "first deletion did not move/update last locator");
    audit(*lists);
    require(lists->erase(120) && lists->location(138) == ListLocation{0, 20}, "middle deletion did not update moved locator");
    audit(*lists);
    require(lists->erase(137), "last deletion refused"); audit(*lists);
    require(!lists->erase(100) && !lists->erase(199) && lists->missing_deletes() == 2, "missing delete is not idempotent/counted");
    require(!lists->location(100) && !lists->contains(100), "deleted ID remained live");
    rejects([&] { lists->insert(100, 2, code(100)); }); // Tombstone reuse forbidden.
    rejects([&] { lists->insert(139, 2, code(139)); }); // Duplicate live ID.
    require(lists->estimated_bytes() == accounted && lists->copy_list(0).capacity_slots == 48, "delete shrank/reallocated capacity");
    for (uint32_t id = 100; id < 140; ++id) if (lists->contains(id)) require(lists->erase(id), "clear failed");
    require(lists->size() == 0 && lists->copy_list(0).ids.empty(), "clear left members"); audit(*lists);
    lists->insert(140, 1, code(140));
    require(lists->erase(140) && lists->copy_list(1).ids.empty(), "only-entry deletion failed");
    rejects([&] { lists->insert(99, 0, code(99)); });
    rejects([&] { lists->insert(200, 0, code(200)); });
    rejects([&] { lists->erase(UINT32_MAX); });
    rejects([&] { lists->contains(99); });
    rejects([&] { lists->location(200); });
    rejects([&] { lists->insert(150, 3, code(150)); });
    const auto bytes = code(150);
    rejects([&] { lists->insert(150, 2, std::span<const uint8_t>(bytes).first(2)); });
    lists->insert(150, 2, bytes); audit(*lists); // Validation did not consume ID.
    rejects([&] { lists->copy_list(3); });
    rejects([&] { lists->copy_list(2, 1024 + 7 - 1); });
    require(lists->copy_list(2, 1024 + 7).ids == std::vector<uint32_t>{150}, "exact copy workspace boundary refused");
    auto invalid = options; invalid.nlist = 0; rejects([&] { MutablePostingLists::create(invalid); });
    invalid = options; invalid.chunks = 0; rejects([&] { MutablePostingLists::create(invalid); });
    invalid = options; invalid.first_id = invalid.id_capacity; rejects([&] { MutablePostingLists::create(invalid); });
    invalid = options; invalid.memory_budget_bytes = MutablePostingLists::estimate_empty_bytes(options) - 1;
    rejects([&] { MutablePostingLists::create(invalid); });
    invalid = {uint32_t(INT32_MAX), uint32_t(INT32_MAX), UINT32_MAX, 0, UINT64_MAX};
    rejects([&] { MutablePostingLists::estimate_empty_bytes(invalid); }); // Checked arithmetic, no huge allocation.
    auto high = MutablePostingLists::create({2, 3, UINT32_MAX, UINT32_MAX - 32});
    high->insert(UINT32_MAX - 1, 1, code(UINT32_MAX - 1));
    require(high->location(UINT32_MAX - 1) == ListLocation{1, 0}, "last valid external ID remapped");
    audit(*high);
}

void budget_failure() {
    MutableListOptions options{2, 3, 64};
    options.memory_budget_bytes = MutablePostingLists::estimate_empty_bytes(options);
    auto lists = MutablePostingLists::create(options);
    for (uint32_t id = 0; id < 16; ++id) lists->insert(id, 0, code(id));
    const auto bytes = lists->estimated_bytes();
    rejects([&] { lists->insert(16, 0, code(16)); });
    require(lists->size() == 16 && !lists->contains(16) && !lists->location(16) &&
            lists->estimated_bytes() == bytes && lists->copy_list(0).capacity_slots == 16, "failed growth changed membership/capacity/budget");
    lists->insert(16, 1, code(16)); audit(*lists); // Reservation rolled back.
    options.memory_budget_bytes += 32 * 7 + 128;
    auto boundary = MutablePostingLists::create(options);
    for (uint32_t id = 0; id < 17; ++id) boundary->insert(id, 0, code(id));
    require(boundary->peak_estimated_bytes() == options.memory_budget_bytes &&
            boundary->estimated_bytes() == bytes + 16 * 7, "growth peak did not account for old/new buffers");
    audit(*boundary);
}

void reservations() {
    auto lists = MutablePostingLists::create({2, 3, 64});
    const auto distances = table();
    const std::array<uint32_t, 2> all{0, 1};
    {
        auto pending = lists->reserve_insert(10);
        require(lists->pending_insertions() == 1 && lists->size() == 0 && !lists->contains(10) && !lists->location(10) &&
                lists->scan(all, distances, 64).candidates.empty(), "pending insertion became visible before publication");
        rejects([&] { lists->reserve_insert(10); });
        require(!lists->erase(10) && lists->missing_deletes() == 1, "pending insertion delete should linearize before publication");
    }
    require(lists->pending_insertions() == 0 && !lists->contains(10), "destructor failed to cancel reservation");
    auto pending = lists->reserve_insert(10);
    auto moved = std::move(pending);
    rejects([&] { pending.commit(0, code(10)); });
    moved.commit(0, code(10));
    require(lists->contains(10) && lists->pending_insertions() == 0, "moved reservation did not publish");
    rejects([&] { moved.commit(0, code(10)); });
    audit(*lists);
    try {
        auto failed_heap = lists->reserve_insert(11);
        throw std::runtime_error("simulated caller heap write failure");
    } catch (const std::runtime_error&) {}
    require(lists->pending_insertions() == 0 && !lists->contains(11), "caller failure did not unwind pending reservation");
    lists->insert(11, 1, code(11)); audit(*lists);
    auto a = lists->reserve_insert(12), b = lists->reserve_insert(13);
    a = std::move(b); // Cancel 12, take ownership of 13.
    require(lists->pending_insertions() == 1, "move assignment leaked prior reservation");
    lists->insert(12, 0, code(12)); a.commit(1, code(13)); audit(*lists);
    auto invalid = lists->reserve_insert(14);
    rejects([&] { invalid.commit(0, {}); });
    require(lists->pending_insertions() == 0 && !lists->contains(14), "failed commit retained reservation");
    lists->insert(14, 0, code(14)); audit(*lists);
    auto canceled = lists->reserve_insert(15);
    canceled.cancel(); canceled.cancel(); // Idempotent, no pending-count underflow.
    require(lists->pending_insertions() == 0, "double cancel corrupted pending count");
    lists->insert(15, 1, code(15)); audit(*lists);
    // Hand off a reservation to a worker: the parent observes only the pending
    // state until that worker commits; guard ownership is never concurrently used.
    auto handed_off = lists->reserve_insert(16);
    std::thread publisher([guard = std::move(handed_off)]() mutable { guard.commit(1, code(16)); });
    publisher.join();
    require(lists->contains(16) && lists->pending_insertions() == 0, "cross-thread reservation handoff failed");
    audit(*lists);
    auto readiness = lists->reserve_insert(17);
    bool called = false;
    struct ReadinessFailure {};
    bool propagated = false;
    try {
        readiness.commit_ready(0, code(17), [&] {
            called = true;
            require(!lists->contains(17), "readiness callback observed published ID");
            throw ReadinessFailure{};
        });
    } catch (const ReadinessFailure&) { propagated = true; }
    require(called && propagated && lists->pending_insertions() == 0 && !lists->contains(17) && !lists->location(17),
            "readiness failure leaked visibility/reservation");
    audit(*lists);
    called = false;
    lists->reserve_insert(17).commit_ready(0, code(17), [&] {
        require(!lists->contains(17), "readiness was invoked after membership publication"); called = true;
    });
    require(called && lists->contains(17), "readiness callback retry failed");
    audit(*lists);
}

void adc() {
    auto lists = MutablePostingLists::create({4, 3, 256});
    for (uint32_t id = 0; id < 96; ++id) lists->insert(id, id % 3, code(id));
    for (uint32_t id = 0; id < 96; id += 5) require(lists->erase(id), "ADC fixture delete failed");
    auto distances = table();
    const std::array<uint32_t, 3> selected{3, 2, 0};
    std::vector<PQCandidate> oracle;
    for (uint32_t id = 0; id < 96; ++id) if (id % 3 != 1 && id % 5 != 0) oracle.push_back({id, score(id, distances)});
    std::sort(oracle.begin(), oracle.end(), nearer);
    for (uint32_t budget : {1u, 10u, 300u}) {
        auto expected = oracle; expected.resize(std::min<size_t>(budget, expected.size()));
        const auto scanned = lists->scan(selected, distances, budget);
        require(scanned.candidates == expected && scanned.scanned_codes == oracle.size(), "post-mutation ADC differs from oracle");
    }
    const uint64_t workspace = 10 * 48 + selected.size() * 4 + 1024;
    rejects([&] { lists->scan(selected, distances, 10, workspace - 1); });
    require(lists->scan(selected, distances, 10, workspace).candidates.size() == 10, "exact scan workspace boundary refused");
    require(lists->scan({}, distances, 10).candidates.empty() && lists->scan(std::array<uint32_t, 1>{3}, distances, 10).scanned_codes == 0,
            "empty scan fabricated candidates");
    rejects([&] { lists->scan(selected, distances, 0); });
    rejects([&] { lists->scan(std::array<uint32_t, 2>{0, 0}, distances, 10); });
    rejects([&] { lists->scan(std::array<uint32_t, 1>{4}, distances, 10); });
    rejects([&] { lists->scan(selected, std::span<const float>(distances).first(256), 10); });
    distances[0] = -1; rejects([&] { lists->scan(selected, distances, 10); });
    distances[0] = std::numeric_limits<float>::quiet_NaN(); rejects([&] { lists->scan(selected, distances, 10); });
    std::fill(distances.begin(), distances.end(), std::numeric_limits<float>::max());
    rejects([&] { lists->scan(selected, distances, 10); });
    std::fill(distances.begin(), distances.end(), 1);
    const auto tied = lists->scan(selected, distances, 300);
    require(std::is_sorted(tied.candidates.begin(), tied.candidates.end(), nearer), "mutable ADC tie order changed");
    for (const auto hit : tied.candidates) require(hit.squared_distance == 3 && lists->contains(hit.id), "completed deletion remained in scan");
    audit(*lists);
}

template<typename T> void conversion(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    const auto vectors = (root / "vectors.bin").string(), codes = (root / "codes.bin").string();
    constexpr uint32_t rows = 6, base = 4, dim = 3;
    {
        std::ofstream out(vectors, std::ios::binary), pq(codes, std::ios::binary);
        const uint32_t shape[]{rows, dim};
        out.write(reinterpret_cast<const char*>(shape), sizeof(shape)); pq.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        for (uint32_t r = 0; r < rows; ++r) {
            const T value = T(r < 2 ? r : 20 + r);
            const T row[]{value, value, value}; out.write(reinterpret_cast<const char*>(row), sizeof(row));
            const auto bytes = code(r + 5); pq.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
    }
    const std::vector<float> centroids{0, 0, 0, 20, 20, 20, 100, 100, 100};
    auto router = CentroidRouter::build(dim, centroids);
    ListBuildOptions options{base, 5, 64, 3}; options.exact_assignment = true;
    const auto immutable = PostingLists::build<T>(vectors, codes, *router, options);
    auto mutable_lists = MutablePostingLists::from_immutable(immutable);
    require(mutable_lists->size() == base && mutable_lists->chunks() == 3 && mutable_lists->first_id() == 5 && mutable_lists->id_capacity() == 64,
            "conversion changed shape/ID policy");
    const auto allowance = mutable_lists->estimated_bytes();
    require(MutablePostingLists::from_immutable(immutable, allowance)->size() == base, "exact conversion estimate boundary refused");
    rejects([&] { MutablePostingLists::from_immutable(immutable, allowance - 1); });
    audit(*mutable_lists);
    for (uint32_t c = 0; c < immutable.nlist(); ++c) {
        const auto copy = mutable_lists->copy_list(c);
        require(copy.capacity_slots == immutable.ids(c).size() + 16, "conversion lacks exact +16 spare capacity");
        require(std::equal(copy.ids.begin(), copy.ids.end(), immutable.ids(c).begin(), immutable.ids(c).end()) &&
                std::equal(copy.codes.begin(), copy.codes.end(), immutable.codes(c).begin(), immutable.codes(c).end()), "conversion altered IDs/PQ bytes");
    }
    const auto distances = table();
    const std::array<uint32_t, 3> all{2, 0, 1};
    require(mutable_lists->scan(all, distances, 64).candidates == immutable.scan(all, distances, 64).candidates, "conversion changed ADC");
    mutable_lists->insert(9, 2, code(9)); mutable_lists->erase(5); audit(*mutable_lists);
    require(immutable.size() == base && immutable.ids(0).front() == 5, "mutable conversion modified borrowed CSR input");
}

void races() {
    auto lists = MutablePostingLists::create({4, 3, 512});
    auto distances = table();
    std::array<std::exception_ptr, 8> errors{};
    std::vector<std::thread> threads;
    std::barrier start(9);
    std::atomic<uint32_t> readers_started{0};
    for (uint32_t t = 0; t < 4; ++t) threads.emplace_back([&, t] {
        start.arrive_and_wait();
        try {
            for (uint32_t r = 0; r < 64; ++r) {
                const uint32_t id = t * 64 + r;
                lists->insert(id, 0, code(id)); // Same-list growth/swap contention.
                if (r % 3 == 0) require(lists->erase(id), "concurrent erase failed");
                if (r == 0) while (!readers_started.load(std::memory_order_acquire)) std::this_thread::yield();
                std::this_thread::yield();
            }
        } catch (...) { errors[t] = std::current_exception(); }
    });
    for (uint32_t t = 4; t < 8; ++t) threads.emplace_back([&, t] {
        start.arrive_and_wait();
        try {
            for (uint32_t r = 0; r < 200; ++r) {
                const auto scanned = lists->scan(std::array<uint32_t, 2>{0, 3}, distances, 20);
                readers_started.fetch_add(1, std::memory_order_release);
                require(std::is_sorted(scanned.candidates.begin(), scanned.candidates.end(), nearer), "racing scan order corrupt");
                for (auto hit : scanned.candidates) require(hit.squared_distance == score(hit.id, distances), "racing scan code/ID mismatch");
                const auto copy = lists->copy_list(0);
                for (size_t s = 0; s < copy.ids.size(); ++s) {
                    const auto bytes = code(copy.ids[s]);
                    require(std::equal(bytes.begin(), bytes.end(), copy.codes.begin() + s * 3), "racing growth/swap snapshot corrupt");
                }
                const auto loc = lists->location(r % 256);
                require(!loc || loc->list == 0, "locator owner changed during racing swap");
            }
        } catch (...) { readers_started.fetch_add(1, std::memory_order_release); errors[t] = std::current_exception(); }
    });
    start.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    require(lists->size() == 4 * (64 - 22), "concurrent final live count mismatch"); audit(*lists);
    for (uint32_t id = 0; id < 256; ++id) require(lists->contains(id) == (id % 64 % 3 != 0), "concurrent final live set mismatch");

    // One external ID contended across different lists: one CAS winner only.
    threads.clear(); std::barrier conflict(9); std::atomic<uint32_t> winners{0};
    for (uint32_t t = 0; t < 8; ++t) threads.emplace_back([&, t] {
        conflict.arrive_and_wait();
        try { lists->insert(300, t % 4, code(300)); winners.fetch_add(1); }
        catch (const std::invalid_argument&) {}
    });
    conflict.arrive_and_wait(); for (auto& thread : threads) thread.join();
    require(winners == 1 && lists->contains(300), "concurrent duplicate insertion had multiple/no winners"); audit(*lists);
    threads.clear(); std::barrier deletion(9); winners = 0;
    const auto before_missing = lists->missing_deletes();
    for (uint32_t t = 0; t < 8; ++t) threads.emplace_back([&] {
        deletion.arrive_and_wait(); if (lists->erase(300)) winners.fetch_add(1);
    });
    deletion.arrive_and_wait(); for (auto& thread : threads) thread.join();
    require(winners == 1 && !lists->contains(300) && lists->missing_deletes() == before_missing + 7,
            "concurrent duplicate deletion was not idempotent"); audit(*lists);
    rejects([&] { lists->insert(300, 0, code(300)); });

    // Global budget reservation covers simultaneous replacement arrays on
    // unrelated lists. Only one of these +16 growths fits, even sequentially.
    MutableListOptions options{2, 3, 64};
    const auto empty = MutablePostingLists::estimate_empty_bytes(options);
    options.memory_budget_bytes = empty + 32 * 7 + 128;
    auto limited = MutablePostingLists::create(options);
    for (uint32_t id = 0; id < 32; ++id) limited->insert(id, id / 16, code(id));
    threads.clear(); std::barrier memory(3); winners = 0;
    for (uint32_t t = 0; t < 2; ++t) threads.emplace_back([&, t] {
        memory.arrive_and_wait();
        try { limited->insert(32 + t, t, code(32 + t)); winners.fetch_add(1); }
        catch (const std::invalid_argument&) {}
    });
    memory.arrive_and_wait(); for (auto& thread : threads) thread.join();
    require(winners == 1 && limited->size() == 33 && limited->peak_estimated_bytes() == options.memory_budget_bytes,
            "concurrent growth exceeded/misaccounted budget"); audit(*limited);
    const uint32_t failed = limited->contains(32) ? 33 : 32;
    const uint32_t owner = limited->location(failed == 32 ? 33 : 32)->list;
    limited->insert(failed, owner, code(failed)); audit(*limited); // Failed ID reservation reusable.

    // Repeated cancellations can immediately race with new reservations of the
    // same never-published ID, without leaking/underflowing pending counters.
    threads.clear(); std::barrier retry(5);
    for (uint32_t t = 0; t < 4; ++t) threads.emplace_back([&] {
        retry.arrive_and_wait();
        for (uint32_t r = 0; r < 200; ++r) {
            try { auto pending = lists->reserve_insert(400); std::this_thread::yield(); }
            catch (const std::invalid_argument&) {}
        }
    });
    retry.arrive_and_wait(); for (auto& thread : threads) thread.join();
    require(lists->pending_insertions() == 0 && !lists->contains(400), "racing cancellations leaked pending state/count");
    lists->insert(400, 2, code(400)); audit(*lists);
}
void checkpoints(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    const MutableListOptions options{4, 3, 164, 100};
    auto lists = MutablePostingLists::create(options);
    for (uint32_t id = 100; id < 148; ++id) lists->insert(id, id % 2, code(id));
    for (uint32_t id : {100u, 105u, 112u, 147u}) require(lists->erase(id), "checkpoint fixture erase failed");
    require(!lists->erase(100) && !lists->erase(163), "checkpoint fixture missing delete failed");
    const auto path = (root / "mutable.lists").string();
    {
        auto pending = lists->reserve_insert(160);
        rejects([&] { lists->save_checkpoint(path); });
        require(!std::filesystem::exists(path), "pending checkpoint claimed output");
    }
    lists->save_checkpoint(path);
    const auto info = MutablePostingLists::inspect_checkpoint(path, options);
    require(info.live == 44 && info.deleted == 4 && info.missing_deletes == 2 && info.estimated_bytes == lists->estimated_bytes(),
            "checkpoint counts/capacity estimate changed");
    auto tight = options; tight.memory_budget_bytes = info.estimated_bytes - 1;
    rejects([&] { MutablePostingLists::load_checkpoint(path, tight); });
    tight.memory_budget_bytes = info.estimated_bytes;
    auto loaded = MutablePostingLists::load_checkpoint(path, tight);
    audit(*loaded);
    for (uint32_t c = 0; c < options.nlist; ++c) {
        const auto before = lists->copy_list(c), after = loaded->copy_list(c);
        require(before.ids == after.ids && before.codes == after.codes && before.capacity_slots == after.capacity_slots,
                "mutable checkpoint changed swap order/codes/capacity");
    }
    for (uint32_t id = options.first_id; id < options.id_capacity; ++id)
        require(lists->location(id) == loaded->location(id) && lists->deleted(id) == loaded->deleted(id), "checkpoint locator/tombstone reconstruction mismatch");
    rejects([&] { loaded->insert(100, 0, code(100)); });
    loaded->insert(160, 0, code(160)); audit(*loaded);
    const auto identity = artifact_digest(path);
    bool existing_refused = false;
    try { loaded->save_checkpoint(path); } catch (const std::runtime_error&) { existing_refused = true; }
    require(existing_refused, "checkpoint did not refuse existing artifact");
    require(artifact_digest(path) == identity, "checkpoint overwrote existing output");
    auto wrong = options; ++wrong.chunks;
    rejects([&] { MutablePostingLists::load_checkpoint(path, wrong); });
    for (uint32_t id = options.first_id; id < options.id_capacity; ++id) if (loaded->contains(id)) loaded->erase(id);
    const auto empty = (root / "empty.lists").string(); loaded->save_checkpoint(empty);
    auto empty_loaded = MutablePostingLists::load_checkpoint(empty, options);
    require(empty_loaded->size() == 0 && empty_loaded->deleted(160) && empty_loaded->deleted(100), "empty mutable checkpoint lost tombstones");
    audit(*empty_loaded);
    // Duplicate IDs survive a recomputed file checksum but must not reconstruct
    // two locators pointing to one record. This mutation touches owned fixtures only.
    const auto duplicate = (root / "duplicate.lists").string(); std::filesystem::copy_file(path, duplicate);
    const uint64_t first_list_ids = 44 + options.id_capacity - options.first_id + 12;
    const auto copied = lists->copy_list(0); require(copied.ids.size() > 1, "duplicate fixture needs two entries");
    {
        std::fstream out(duplicate, std::ios::binary | std::ios::in | std::ios::out); out.seekp(first_list_ids + 4);
        out.write(reinterpret_cast<const char*>(&copied.ids[0]), 4);
    }
    const auto bytes = std::filesystem::file_size(duplicate), hash = artifact_hash_prefix(duplicate, bytes - 8);
    {
        std::fstream out(duplicate, std::ios::binary | std::ios::in | std::ios::out); out.seekp(bytes - 8);
        out.write(reinterpret_cast<const char*>(&hash), 8);
    }
    rejects([&] { MutablePostingLists::load_checkpoint(duplicate, options); });
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected fixture directory");
        const auto root = std::filesystem::path(argv[1]) /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        basic(); budget_failure(); reservations(); adc(); races(); checkpoints(root / "checkpoints");
        conversion<float>(root / "float"); conversion<int8_t>(root / "int8"); conversion<uint8_t>(root / "uint8");
        std::cout << "Mutable IVF list/locator/growth/ADC tests passed (component only)\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
