#include "bufann/ivf_index.h"
#include "../ivf_mixed_workload.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <barrier>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <sys/resource.h>

namespace {
using namespace diskann::ivf;
constexpr uint32_t dim = 13, chunks = 3, rows = 64, base = 24, first = 5, capacity = 96;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
template<typename F> void rejects(F&& operation) {
    try { operation(); } catch (const std::exception&) { return; }
    catch (const diskann::ANNException&) { return; }
    throw std::runtime_error("expected refusal");
}
template<typename T> void write(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void pivots(const std::string& path) {
    std::array<uint64_t, 5> offsets{4096, 0, 0, 0, 0};
    offsets[1] = offsets[0] + 8 + 256 * dim * 4;
    offsets[2] = offsets[1] + 8 + dim * 4;
    offsets[3] = offsets[2] + 8 + dim * 4;
    offsets[4] = offsets[3] + 8 + (chunks + 1) * 4;
    std::ofstream out(path, std::ios::binary);
    write(out, uint32_t(5)); write(out, uint32_t(1));
    for (auto offset : offsets) write(out, offset);
    out.seekp(offsets[0]); write(out, uint32_t(256)); write(out, dim);
    for (uint32_t c = 0; c < 256; ++c)
        for (uint32_t d = 0; d < dim; ++d) write(out, float(c) - 100 + d * .25f);
    write(out, dim); write(out, uint32_t(1));
    for (uint32_t d = 0; d < dim; ++d) write(out, float(d) * .25f);
    write(out, dim); write(out, uint32_t(1));
    for (uint32_t d = 0; d < dim; ++d) write(out, dim - 1 - d);
    write(out, chunks + 1); write(out, uint32_t(1));
    for (uint32_t offset : {0u, 4u, 8u, 13u}) write(out, offset);
    require(bool(out), "pivot fixture write failed");
}
bool nearer(const ExactHit& a, const ExactHit& b) {
    return a.squared_distance < b.squared_distance || (a.squared_distance == b.squared_distance && a.id < b.id);
}
template<typename T> void replace_at(const std::string& path, uint64_t offset, T value) {
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out); out.seekp(offset);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value)); require(bool(out), "corruption fixture write failed");
}
void rehash_trailer(const std::string& path) {
    const auto bytes = std::filesystem::file_size(path);
    replace_at(path, bytes - 8, artifact_hash_prefix(path, bytes - 8));
}
void rehash_manifest(const std::string& prefix) {
    std::ifstream in(prefix + ".manifest"); std::string header; std::getline(in, header);
    std::ostringstream text; text << header << '\n';
    for (auto suffix : mutable_component_suffixes) {
        const auto digest = artifact_digest(prefix + suffix); text << suffix << ' ' << digest.bytes << ' ' << digest.hash << '\n';
    }
    in.close(); std::ofstream out(prefix + ".manifest"); out << text.str(); require(bool(out), "manifest fixture write failed");
}
void copy_checkpoint(const std::string& from, const std::string& to) {
    for (auto suffix : mutable_component_suffixes) std::filesystem::copy_file(from + suffix, to + suffix);
    std::filesystem::copy_file(from + ".manifest", to + ".manifest");
}

// Restrict only this fixture process, and restore even when a check throws.
// Growth fixtures have no dirty pages; output-copy fixtures have a clean source.
class RefuseFileGrowth {
  public:
    explicit RefuseFileGrowth(rlim_t bytes = 0) {
        require(getrlimit(RLIMIT_FSIZE, &_saved) == 0, "getrlimit failed");
        struct sigaction ignored{}; ignored.sa_handler = SIG_IGN; sigemptyset(&ignored.sa_mask);
        require(sigaction(SIGXFSZ, &ignored, &_signal) == 0, "sigaction failed");
        auto limit = _saved; limit.rlim_cur = std::min(_saved.rlim_cur, bytes);
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) {
            sigaction(SIGXFSZ, &_signal, nullptr); throw std::runtime_error("setrlimit failed");
        }
    }
    ~RefuseFileGrowth() { setrlimit(RLIMIT_FSIZE, &_saved); sigaction(SIGXFSZ, &_signal, nullptr); }
    RefuseFileGrowth(const RefuseFileGrowth&) = delete;
    RefuseFileGrowth& operator=(const RefuseFileGrowth&) = delete;
  private:
    struct rlimit _saved{};
    struct sigaction _signal{};
};

template<typename T> void heap_lifecycle(const std::filesystem::path& root) {
    const HeapOptions options{dim, capacity, 4096, 2};
    auto heap = HeapStorage<T>::create_mutable(options, (root / "standalone mutable.heap").string());
    std::array<T, dim> vector; vector.fill(T(7));
    alignas(64) std::array<T, 16> output; output.fill(T(42));
    heap->prepare_insert(1, vector);
    require(heap->size() == 0 && heap->retained_size() == 1 && !heap->contains(1) &&
            !heap->copy_coordinates(1, output) && output[0] == T(42), "prepared heap entry was visible");
    rejects([&] { heap->prepare_insert(1, vector); });
    heap->cancel_insert(1); heap->cancel_insert(1);
    vector.fill(T(8)); heap->prepare_insert(1, vector);
    require(heap->retained_size() == 1, "aborted retry allocated/reused another physical slot");
    heap->publish_insert(1);
    require(heap->size() == 1 && heap->copy_coordinates(1, output) && output[0] == T(8) && output[13] == T{}, "publication/copy mismatch");
    rejects([&] { heap->cancel_insert(1); });
    require(heap->retire(1) && !heap->retire(1) && heap->size() == 0 && heap->retained_size() == 1, "retirement reclaimed or remained live");
    output.fill(T(42)); require(!heap->copy_coordinates(1, output) && output[0] == T(42), "retired copy changed missing output");
    rejects([&] { heap->prepare_insert(1, vector); });
    rejects([&] { heap->append(2, vector); });
    rejects([&] { heap->publish_insert(2); });
    rejects([&] { heap->prepare_insert(capacity, vector); });
    rejects([&] { heap->prepare_insert(2, std::span<const T>(vector).first(dim - 1)); });
    const auto checkpoint = (root / "unsupported mutable.meta").string();
    rejects([&] { heap->checkpoint(checkpoint); });
    require(!heap->failed() && !std::filesystem::exists(checkpoint) && !std::filesystem::exists(checkpoint + ".ivf"),
            "unsupported mutable checkpoint published/poisoned session");
    if constexpr (std::is_same_v<T, float>) {
        vector[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { heap->prepare_insert(2, vector); });
        require(!heap->failed() && heap->retained_size() == 1, "coordinate validation changed physical state");
    }
    auto failed = HeapStorage<T>::create_mutable(options, (root / "failed growth.heap").string());
    vector.fill(T(7));
    bool growth_failed = false;
    {
        RefuseFileGrowth limit;
        try { failed->prepare_insert(2, vector); }
        catch (const diskann::ANNException& error) {
            growth_failed = error.message().find("grow_heap ftruncate failed") != std::string::npos;
        }
    }
    require(growth_failed && failed->failed() && failed->size() == 0 && !failed->contains(2),
            "real backend file-growth failure did not poison/hide the entry");
    rejects([&] { failed->prepare_insert(2, vector); });
    rejects([&] { failed->publish_insert(2); });
    rejects([&] { failed->checkpoint((root / "failed.meta").string()); });
    require(!std::filesystem::exists(root / "failed.meta"), "failed heap published a checkpoint");
}

template<typename T> void exercise(const std::filesystem::path& root, DataType type) {
    std::filesystem::create_directories(root);
    heap_lifecycle<T>(root);
    const auto vectors = (root / "vectors.bin").string(), pq_path = (root / "pivots.bin").string();
    const auto codes_path = (root / "codes.bin").string(); pivots(pq_path);
    PQCodebook<T> pq(pq_path, dim, chunks);
    std::vector<T> data(rows * dim);
    {
        std::ofstream out(vectors, std::ios::binary), codes(codes_path, std::ios::binary);
        write(out, rows); write(out, dim); write(codes, rows); write(codes, chunks);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t d = 0; d < dim; ++d) {
                const int coordinate = int(r % 4) * 30 + int((r / 4 + d) % 9) - (std::is_same_v<T, uint8_t> ? -140 : 50);
                data[r * dim + d] = T(coordinate); write(out, data[r * dim + d]);
            }
            std::array<uint8_t, chunks> code{}; pq.encode(data.data() + r * dim, code.data());
            codes.write(reinterpret_cast<const char*>(code.data()), code.size());
        }
        require(bool(out) && bool(codes), "source fixture write failed");
    }
    const auto prefix = (root / "immutable source").string();
    BuildConfig training; training.data_type = type; training.dim = dim; training.base_count = base;
    training.nlist = 4; training.max_reps = 2; training.build_threads = 2;
    NativeIndex<T>::build(prefix, vectors, pq_path, codes_path, training, {dim, capacity, 4096, 2}, {}, first);
    std::array<ArtifactDigest, 9> source_identity;
    for (size_t i = 0; i < index_component_suffixes.size(); ++i) source_identity[i] = artifact_digest(prefix + index_component_suffixes[i]);
    source_identity.back() = artifact_digest(prefix + ".manifest");
    const auto private_heap = (root / "private mutable.heap").string();
    auto index = MutableIndex<T>::fork(prefix, private_heap, {2});
    require(index->size() == base && index->retained_heap_records() == base && !index->contains(first + base), "fork indexed reserved tail");
    const auto initial_bytes = index->retained_heap_bytes();
    const auto row = [&](uint32_t r) { return std::span<const T>(data.data() + size_t(r) * dim, dim); };
    const auto query = row(60);
    const SearchConfig config{4, 5, 32, 2};
    // S06: real frozen IVF composition under both schedulers, three matching
    // round barriers/checkpoint reopens. All inserts target the SAME centroid,
    // overlapping its +16 growth/swap deletion and rerank with eight queries.
    for (bool fixed : {false, true}) {
        const auto label = fixed ? "fixed" : "weighted";
        auto mixed = MutableIndex<T>::fork(prefix, (root / (std::string(label) + " mixed.heap")).string(), {2});
        for (uint32_t round = 0; round < 3; ++round) {
            ivf_cli::MixedPlan plan{fixed, 16, 8, 8, 64};
            std::barrier overlap_gate(16); std::atomic<uint32_t> first_mutations{0};
            auto mutation_gate = [&] { if (fixed && first_mutations.fetch_add(1) < 8) overlap_gate.arrive_and_wait(); };
            const auto stats = ivf_cli::run_mixed_round(plan, {
                [&](uint64_t i) { mutation_gate(); mixed->insert(first + base + round * 8 + uint32_t(i), row(0)); },
                [&](uint64_t i) { mutation_gate(); require(mixed->erase(first + round * 8 + uint32_t(i)), "mixed delete lost a base ID"); },
                [&](uint64_t i) {
                    if (fixed && i < 8) overlap_gate.arrive_and_wait();
                    for (auto hit : mixed->query(query, config).hits) {
                        double squared = 0; const auto coordinates = row(hit.id >= first + base ? 0 : hit.id - first);
                        for (uint32_t d = 0; d < dim; ++d) { const double delta = double(query[d]) - double(coordinates[d]); squared += delta * delta; }
                        require(hit.squared_distance == squared, "mixed query returned mismatched/stale coordinates");
                    }
                }});
            if (fixed) require(stats.overlapping_queries >= 8 && stats.peak_queries == 8 && stats.peak_mutations == 8,
                               "fixed native fixture failed dedicated 8+8 overlap");
            std::vector<uint8_t> seen(capacity, 0); uint32_t entries = 0;
            for (uint32_t c = 0; c < training.nlist; ++c) {
                const auto copied = mixed->copy_list(c);
                for (auto id : copied.ids) { require(id < capacity && !seen[id] && mixed->contains(id), "mixed round list/locator membership mismatch"); seen[id] = 1; ++entries; }
            }
            std::vector<ExactHit> expected;
            for (uint32_t id = first; id < capacity; ++id) {
                const bool live = (id >= first + (round + 1) * 8 && id < first + base) ||
                                  (id >= first + base && id < first + base + (round + 1) * 8);
                require(bool(seen[id]) == live && mixed->contains(id) == live, "mixed round live set differs from schedule");
                if (live) {
                    double squared = 0; const auto coordinates = row(id >= first + base ? 0 : id - first);
                    for (uint32_t d = 0; d < dim; ++d) { const double delta = double(query[d]) - double(coordinates[d]); squared += delta * delta; }
                    expected.push_back({id, squared});
                }
            }
            std::sort(expected.begin(), expected.end(), nearer); expected.resize(config.recall_at);
            require(entries == base && mixed->size() == base && mixed->query(query, config).hits == expected, "mixed barrier exact oracle/count failure");
            const auto saved = (root / (std::string(label) + " round " + std::to_string(round))).string();
            mixed->checkpoint(saved);
            mixed = MutableIndex<T>::reopen(saved, saved + ".private.heap", {2});
            require(mixed->query(query, config).hits == expected, "mixed round checkpoint reopen changed results");
        }
        rejects([&] { mixed->insert(first, row(0)); });
    }
    std::vector<uint8_t> live(rows, 0); std::fill_n(live.begin(), base, 1);
    auto distance = [&](uint32_t id) {
        double total = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            const double delta = double(query[d]) - double(data[(id - first) * dim + d]); total += delta * delta;
        }
        return ExactHit{id, total};
    };
    auto oracle = [&] {
        std::vector<ExactHit> hits;
        for (uint32_t r = 0; r < rows; ++r) if (live[r]) hits.push_back(distance(first + r));
        std::sort(hits.begin(), hits.end(), nearer); if (hits.size() > config.recall_at) hits.resize(config.recall_at);
        return hits;
    };
    require(index->query(query, config).hits == oracle(), "fork query differs from base exact oracle");
    const auto mutation_workspace = index->estimated_mutation_bytes();
    rejects([&] { index->insert(first + base, row(base), {mutation_workspace - 1}); });
    require(index->retained_heap_records() == base && !index->contains(first + base), "mutation workspace refusal wrote heap/list");
    index->insert(first + base, row(base), {mutation_workspace}); live[base] = 1;
    require(index->query(query, config).hits == oracle(), "insert was not immediately visible");
    std::array<uint8_t, chunks> expected_code{}; pq.encode(row(base).data(), expected_code.data());
    uint32_t found = 0;
    for (uint32_t c = 0; c < 4; ++c) {
        auto copy = index->copy_list(c);
        for (size_t s = 0; s < copy.ids.size(); ++s) if (copy.ids[s] == first + base) {
            ++found; require(std::equal(expected_code.begin(), expected_code.end(), copy.codes.begin() + s * chunks), "insert changed existing PQ encoding");
        }
    }
    require(found == 1, "inserted ID duplicated/lost across lists");
    rejects([&] { index->insert(first + base, row(base)); });
    const std::array<PQCandidate, 2> stale{{{first, 0}, {first + base, 0}}};
    require(index->erase(first), "base deletion failed"); live[0] = 0;
    const auto filtered = index->rerank_candidates(query, stale, config);
    require(filtered.hits == std::vector<ExactHit>{distance(first + base)} && filtered.stale_candidates == 1,
            "stale scan candidate was not dropped before fetch/rerank");
    require(!index->erase(first) && !index->erase(capacity - 1) && index->missing_deletes() == 2, "missing deletion was not counted/idempotent");
    rejects([&] { index->insert(first, row(0)); });
    // Concurrent same-ID reservations have exactly one publisher. Successful
    // duplicate deletions also have exactly one winner and never permit reuse.
    {
        const uint32_t id = first + base + 1;
        std::barrier gate(3);
        std::atomic<uint32_t> published{0}, refused{0};
        std::array<std::exception_ptr, 2> errors{};
        auto operation = [&](uint32_t t) {
            gate.arrive_and_wait();
            try { index->insert(id, row(base + 1)); ++published; }
            catch (const std::exception&) { ++refused; }
            catch (...) { errors[t] = std::current_exception(); }
        };
        std::thread a(operation, 0), b(operation, 1);
        gate.arrive_and_wait(); a.join(); b.join();
        for (auto error : errors) if (error) std::rethrow_exception(error);
        require(published == 1 && refused == 1 && index->contains(id), "same-ID insert had multiple/no publishers");
        live[base + 1] = 1;
    }
    rejects([&] { index->insert(capacity, row(0)); });
    rejects([&] { index->insert(first + base + 1, row(base + 1).first(dim - 1)); });
    if constexpr (std::is_same_v<T, float>) {
        auto bad = std::vector<T>(row(base + 1).begin(), row(base + 1).end()); bad[0] = std::numeric_limits<float>::infinity();
        rejects([&] { index->insert(first + base + 1, std::span<const T>(bad)); });
    }
    index->insert(first + base + 2, row(base + 2), {index->estimated_mutation_bytes(true), true}); live[base + 2] = 1;
    const auto workspace = index->estimated_query_bytes(config);
    rejects([&] { index->query(query, config, {workspace - 1}); });
    require(index->query(query, config, {workspace}).hits == oracle(), "exact query workspace boundary refused");
    const std::array<PQCandidate, 2> duplicate{{{first + base, 0}, {first + base, 1}}};
    rejects([&] { index->rerank_candidates(query, duplicate, config); });
    const std::array<PQCandidate, 1> invalid{{{capacity, 0}}};
    rejects([&] { index->rerank_candidates(query, invalid, config); });
    // Actual insert/delete activity overlaps queries; this is not the S06 8+8 scheduler.
    std::array<std::exception_ptr, 4> errors{};
    std::barrier start(5);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < 2; ++t) threads.emplace_back([&, t] {
        start.arrive_and_wait();
        try {
            for (uint32_t r = base + 3 + t; r < rows; r += 2) {
                index->insert(first + r, row(r));
                if (r % 3 == 0) require(index->erase(first + r), "concurrent inserted-ID deletion failed");
                std::this_thread::yield();
            }
        } catch (...) { errors[t] = std::current_exception(); }
    });
    for (uint32_t t = 2; t < 4; ++t) threads.emplace_back([&, t] {
        start.arrive_and_wait();
        try {
            for (uint32_t r = 0; r < 25; ++r) {
                const auto result = index->query(query, config);
                require(result.hits.size() <= config.recall_at && std::is_sorted(result.hits.begin(), result.hits.end(), nearer), "concurrent query order/count corrupt");
                for (auto hit : result.hits) require(hit.id >= first && hit.id < first + rows && hit == distance(hit.id), "query copied wrong-ID coordinates");
            }
        } catch (...) { errors[t] = std::current_exception(); }
    });
    start.arrive_and_wait(); for (auto& thread : threads) thread.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    for (uint32_t r = base + 3; r < rows; ++r) live[r] = r % 3 != 0;
    require(index->query(query, config).hits == oracle(), "post-round live set differs from exact oracle");
    {
        const SearchConfig limited_config{2, 5, 4, 2};
        auto frozen_router = CentroidRouter::load(prefix);
        const auto routed = frozen_router->search(query, limited_config.nprobe);
        std::vector<uint32_t> selected;
        std::vector<float> adc_table; pq.populate_distances(query.data(), adc_table);
        std::vector<PQCandidate> candidates;
        for (auto hit : routed) {
            selected.push_back(hit.list_id);
            const auto list = index->copy_list(hit.list_id);
            for (size_t s = 0; s < list.ids.size(); ++s) {
                float distance = 0;
                for (uint32_t c = 0; c < chunks; ++c) distance += adc_table[c * 256 + list.codes[s * chunks + c]];
                candidates.push_back({list.ids[s], distance});
            }
        }
        const auto scanned = candidates.size();
        std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
            return a.squared_distance < b.squared_distance || (a.squared_distance == b.squared_distance && a.id < b.id);
        });
        if (candidates.size() > limited_config.candidate_budget(capacity - first))
            candidates.resize(limited_config.candidate_budget(capacity - first));
        std::vector<ExactHit> exact;
        for (auto candidate : candidates) exact.push_back(distance(candidate.id));
        std::sort(exact.begin(), exact.end(), nearer);
        if (exact.size() > limited_config.recall_at) exact.resize(limited_config.recall_at);
        const auto result = index->query(query, limited_config);
        require(result.selected_lists == selected && result.hits == exact && result.scanned_codes == scanned &&
                result.reranked_vectors == candidates.size() && result.stale_candidates == 0,
                "limited mutable route/ADC/exact rerank differs from independent oracle");
    }
    require(index->retained_heap_records() == rows && index->retained_heap_bytes() >= initial_bytes && !index->failed(), "deleted/aborted records reclaimed or session failed");
    for (uint32_t r = 0; r < rows; ++r) require(index->contains(first + r) == bool(live[r]), "post-round membership mismatch");
    // Independent tight list budget: growth failure occurs AFTER heap preparation,
    // before readiness/list publication. Retry uses the same unpublished physical slot.
    const auto initial_list_bytes = MutablePostingLists::estimate_empty_bytes({4, chunks, capacity, first}) + base * (chunks + 4);
    auto limited = MutableIndex<T>::fork(prefix, (root / "limited mutable.heap").string(), {2, 1024ull * 1024 * 1024, initial_list_bytes});
    for (uint32_t id = first + rows; id < first + rows + 16; ++id) limited->insert(id, row(0));
    const uint32_t failed_id = first + rows + 16;
    rejects([&] { limited->insert(failed_id, row(0)); });
    const auto retained = limited->retained_heap_records();
    require(!limited->contains(failed_id) && limited->size() == base + 16 && retained == base + 17 && !limited->failed(), "failed list growth published/poisoned ID");
    rejects([&] { limited->insert(failed_id, row(0)); });
    require(limited->retained_heap_records() == retained, "retry allocated a second retained slot");
    const auto aborted_checkpoint = (root / "aborted checkpoint").string(); limited->checkpoint(aborted_checkpoint);
    auto aborted = MutableIndex<T>::reopen(aborted_checkpoint, (root / "aborted reopened.heap").string(),
        {2, 1024ull * 1024 * 1024, initial_list_bytes});
    require(aborted->retained_heap_records() == retained && !aborted->contains(failed_id), "checkpoint lost/activated aborted physical slot");
    rejects([&] { aborted->insert(failed_id, row(0)); });
    require(aborted->retained_heap_records() == retained, "aborted reopen retry allocated another slot");
    require(aborted->erase(first), "aborted reopen could not delete base ID"); aborted->insert(failed_id, row(0));
    require(aborted->retained_heap_records() == retained && aborted->contains(failed_id), "aborted reopen retry did not reuse its unpublished slot");
    require(limited->erase(first), "growth retry could not free a list slot");
    limited->insert(failed_id, row(0));
    require(limited->contains(failed_id) && limited->retained_heap_records() == retained && limited->size() == base + 16, "failed-insert retry did not publish correct live set");
    const auto checkpoint = (root / "live checkpoint").string();
    const auto checkpoint_workspace = index->estimated_checkpoint_bytes();
    rejects([&] { index->checkpoint(checkpoint, checkpoint_workspace - 1); });
    require(!std::filesystem::exists(checkpoint + ".manifest"), "checkpoint workspace refusal claimed a prefix");
    // Checkpoint lifecycle admission overlaps real C++ query/delete workers.
    {
        std::barrier gate(3); std::array<std::exception_ptr, 2> errors{};
        std::thread writer([&] {
            gate.arrive_and_wait();
            try { require(index->erase(first + base + 1) && index->erase(first + base + 2), "checkpoint-overlap deletion failed"); }
            catch (...) { errors[0] = std::current_exception(); }
        });
        std::thread reader([&] {
            gate.arrive_and_wait();
            try {
                for (uint32_t i = 0; i < 10; ++i) for (auto hit : index->query(query, config).hits)
                    require(hit == distance(hit.id), "checkpoint-overlap query copied mismatched coordinates");
            } catch (...) { errors[1] = std::current_exception(); }
        });
        gate.arrive_and_wait();
        std::exception_ptr failure;
        try { index->checkpoint(checkpoint, checkpoint_workspace); } catch (...) { failure = std::current_exception(); }
        writer.join(); reader.join();
        if (failure) std::rethrow_exception(failure);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        live[base + 1] = live[base + 2] = 0;
    }
    std::array<ArtifactDigest, 10> checkpoint_identity;
    for (size_t i = 0; i < checkpoint_identity.size(); ++i) checkpoint_identity[i] = artifact_digest(checkpoint + mutable_component_suffixes[i]);
    auto reopened = MutableIndex<T>::reopen(checkpoint, (root / "reopened live.heap").string(), {2});
    std::vector<ExactHit> checkpoint_oracle;
    for (uint32_t r = 0; r < rows; ++r) if (reopened->contains(first + r)) checkpoint_oracle.push_back(distance(first + r));
    std::sort(checkpoint_oracle.begin(), checkpoint_oracle.end(), nearer);
    if (checkpoint_oracle.size() > config.recall_at) checkpoint_oracle.resize(config.recall_at);
    require(reopened->query(query, config).hits == checkpoint_oracle && reopened->retained_heap_records() == rows &&
            reopened->missing_deletes() >= 2, "mutable checkpoint exact results/counts/counters changed");
    rejects([&] { reopened->insert(first, row(0)); });
    rejects([&] { NativeIndex<T>::open(checkpoint); });
    rejects([&] { HeapStorage<T>::reopen({dim, capacity, 4096, 2}, checkpoint + ".heap", checkpoint + ".heap.meta"); });
    rejects([&] { MutableIndex<T>::reopen(prefix, (root / "wrong format.heap").string()); });
    rejects([&] { index->checkpoint(checkpoint); });
    const auto no_budget = (root / "reopen budget refusal.heap").string();
    rejects([&] { MutableIndex<T>::reopen(checkpoint, no_budget, {2, reopened->estimated_retained_bytes() - 1}); });
    require(!std::filesystem::exists(no_budget), "reopen memory refusal copied a heap");
    const auto sidecar_collision = (root / "sidecar collision.heap").string();
    { std::ofstream out(sidecar_collision + ".pq.bin"); out << "owned existing sidecar"; }
    rejects([&] { MutableIndex<T>::reopen(checkpoint, sidecar_collision); });
    require(!std::filesystem::exists(sidecar_collision), "reopen sidecar refusal claimed a heap");
    const auto checkpoint_alias = (root / "checkpoint alias").string();
    std::filesystem::create_symlink(checkpoint + ".heap", checkpoint_alias + ".heap");
    rejects([&] { index->checkpoint(checkpoint_alias); });
    require(!std::filesystem::exists(checkpoint_alias + ".manifest"), "checkpoint symlink refusal claimed manifest");
    const auto relocated = (root / "relocated checkpoint").string(); copy_checkpoint(checkpoint, relocated);
    auto relocated_index = MutableIndex<T>::reopen(relocated, (root / "relocated reopened.heap").string(), {2});
    require(relocated_index->query(query, config).hits == checkpoint_oracle, "relocated checkpoint used original prefix paths");
    // Both sessions can keep mutating after checkpointing/reopen; saved family stays immutable.
    reopened->insert(first + rows, row(0));
    const auto second_checkpoint = (root / "second checkpoint").string(); reopened->checkpoint(second_checkpoint);
    auto second = MutableIndex<T>::reopen(second_checkpoint, (root / "second reopened.heap").string(), {2});
    require(second->contains(first + rows) && second->retained_heap_records() == rows + 1, "second generation lost post-reopen insert");
    for (size_t i = 0; i < checkpoint_identity.size(); ++i)
        require(artifact_digest(checkpoint + mutable_component_suffixes[i]) == checkpoint_identity[i], "continued mutation modified checkpoint family");
    if constexpr (std::is_same_v<T, float>) {
        uint32_t corruption = 0;
        auto bad_case = [&](auto edit) {
            const auto bad = (root / ("corrupt checkpoint " + std::to_string(corruption))).string(); copy_checkpoint(checkpoint, bad);
            edit(bad); rehash_manifest(bad);
            rejects([&] { MutableIndex<T>::reopen(bad, (root / ("corrupt reopen " + std::to_string(corruption) + ".heap")).string(), {2}); });
            ++corruption;
        };
        bad_case([&](const std::string& bad) { replace_at(bad + ".mlists", 44, uint8_t(1)); rehash_trailer(bad + ".mlists"); }); // Reserved state.
        bad_case([&](const std::string& bad) { replace_at(bad + ".mlists", 44 + capacity - first + 4, UINT64_MAX); rehash_trailer(bad + ".mlists"); });
        bad_case([&](const std::string& bad) { replace_at(bad + ".heap.state", 36 + first, uint8_t(1)); rehash_trailer(bad + ".heap.state"); }); // Prepared state.
        bad_case([&](const std::string& bad) { replace_at(bad + ".heap.state", 36 + first, uint8_t(4)); rehash_trailer(bad + ".heap.state"); }); // Deleted -> Aborted mismatch.
        bad_case([&](const std::string& bad) {
            replace_at(bad + ".heap.meta", 112, capacity); // ID outside metadata arrays.
            std::ofstream out(bad + ".heap.meta.ivf"); const auto meta = artifact_digest(bad + ".heap.meta");
            out << "IVFHEAPM1 float " << dim << ' ' << capacity << " 4096 " << rows << ' ' << meta.bytes << ' '
                << std::filesystem::file_size(bad + ".heap") << ' ' << meta.hash << '\n';
        });
        auto repair_meta_descriptor = [&](const std::string& bad) {
            std::ofstream out(bad + ".heap.meta.ivf"); const auto meta = artifact_digest(bad + ".heap.meta");
            out << "IVFHEAPM1 float " << dim << ' ' << capacity << " 4096 " << rows << ' ' << meta.bytes << ' '
                << std::filesystem::file_size(bad + ".heap") << ' ' << meta.hash << '\n';
        };
        bad_case([&](const std::string& bad) {
            std::ifstream in(bad + ".heap.meta", std::ios::binary); in.seekg(116);
            uint64_t locator = 0; in.read(reinterpret_cast<char*>(&locator), 8);
            replace_at(bad + ".heap.meta", 128, locator); repair_meta_descriptor(bad); // Two IDs sharing a slot.
        });
        bad_case([&](const std::string& bad) {
            std::ifstream in(bad + ".heap.meta", std::ios::binary); in.seekg(36);
            uint32_t slots = 0; in.read(reinterpret_cast<char*>(&slots), 4); in.seekg(116);
            uint32_t page = 0; uint16_t slot = 0;
            in.read(reinterpret_cast<char*>(&page), 4); in.read(reinterpret_cast<char*>(&slot), 2);
            const uint64_t coordinates = uint64_t(page) * 4096 + 8 + (slots + 7) / 8 + uint64_t(slot) * (8 + 16 * 4) + 8;
            replace_at(bad + ".heap", coordinates, std::numeric_limits<float>::quiet_NaN());
        });
        bad_case([&](const std::string& bad) {
            const auto bytes = std::filesystem::file_size(bad + ".heap");
            replace_at(bad + ".heap", bytes - 4096 + 8, uint8_t(1)); // Ghost bitmap slot on a spare page.
        });
        bad_case([&](const std::string& bad) {
            std::ofstream out(bad + ".heap.state", std::ios::binary | std::ios::app); out.put('\0');
        });
        bad_case([&](const std::string& bad) { replace_at(bad + ".pq.bin", 4096 + 8, float(999)); }); // Frozen PQ identity.
        const auto damaged = (root / "hash-damaged checkpoint").string(); copy_checkpoint(checkpoint, damaged);
        replace_at(damaged + ".heap", 0, uint8_t(123));
        const auto damaged_output = (root / "hash-damaged reopen.heap").string();
        rejects([&] { MutableIndex<T>::reopen(damaged, damaged_output); });
        require(!std::filesystem::exists(damaged_output), "digest refusal wrote a private heap");
        const auto partial = (root / "partial checkpoint").string();
        { std::ofstream out(partial + ".manifest"); }
        rejects([&] { MutableIndex<T>::reopen(partial, (root / "partial reopen.heap").string()); });
        const auto failed_output = (root / "failed checkpoint output").string();
        {
            RefuseFileGrowth refuse(4096); // Router/lists succeed; the new heap copy fails.
            rejects([&] { index->checkpoint(failed_output); });
        }
        require(!index->failed() && index->query(query, config).hits == oracle(), "checkpoint output failure poisoned/changed healthy session");
        require(std::filesystem::exists(failed_output + ".manifest"), "checkpoint failure did not preserve owned prefix reservation");
        require(std::filesystem::exists(failed_output + ".heap") && std::filesystem::file_size(failed_output + ".heap") <= 4096,
                "checkpoint failure did not exercise/preserve partial heap output");
        rejects([&] { MutableIndex<T>::reopen(failed_output, (root / "failed output reopen.heap").string()); });
    }
    // Preflight and existing private-file refusals do not alter owned/source files.
    const auto refused = (root / "refused mutable.heap").string();
    rejects([&] { MutableIndex<T>::fork(prefix, refused, {2, index->estimated_fork_bytes() - 1}); });
    require(!std::filesystem::exists(refused), "fork memory refusal claimed an output heap");
    rejects([&] { MutableIndex<T>::fork(prefix, private_heap); });
    rejects([&] { MutableIndex<T>::fork(prefix, prefix + ".heap"); });
    const auto alias = root / "mutable heap alias";
    std::filesystem::create_symlink(private_heap, alias);
    rejects([&] { MutableIndex<T>::fork(prefix, alias.string()); });
    require(std::filesystem::is_symlink(alias), "fork replaced an existing private-heap symlink");
    {
        // A missing LIVE heap ID is a composition error, not a stale-delete
        // candidate. The list-removal/heap-retirement gap, however, is safe.
        auto router = CentroidRouter::load(prefix);
        auto lists = MutablePostingLists::create({4, chunks, capacity, first});
        auto heap = HeapStorage<T>::create_mutable({dim, capacity, 4096, 2}, (root / "broken composition.heap").string());
        lists->insert(first, 0, expected_code);
        const std::array<PQCandidate, 1> candidate{{{first, 0}}};
        rejects([&] { rerank_mutable_candidates<T>(*router, *lists, pq, *heap, query, candidate, config); });
        heap->prepare_insert(first, row(0)); heap->publish_insert(first);
        require(rerank_mutable_candidates<T>(*router, *lists, pq, *heap, query, candidate, config).hits ==
                std::vector<ExactHit>{distance(first)}, "healthy live heap ID was dropped");
        require(lists->erase(first) && heap->contains(first), "failed to create delete/retirement gap");
        const auto result = rerank_mutable_candidates<T>(*router, *lists, pq, *heap, query, candidate, config);
        require(result.hits.empty() && result.stale_candidates == 1 && result.backend_batch_calls == 0,
                "logically removed but retained heap record was reranked");
        require(heap->retire(first), "heap retirement after list removal failed");
    }
    {
        std::barrier gate(3);
        std::atomic<uint32_t> deleted{0};
        std::array<std::exception_ptr, 2> errors{};
        auto operation = [&](uint32_t t) {
            gate.arrive_and_wait();
            try { if (index->erase(first + base)) ++deleted; }
            catch (...) { errors[t] = std::current_exception(); }
        };
        std::thread a(operation, 0), b(operation, 1);
        gate.arrive_and_wait(); a.join(); b.join();
        for (auto error : errors) if (error) std::rethrow_exception(error);
        require(deleted == 1 && !index->contains(first + base), "same-ID delete had multiple/no winners");
        rejects([&] { index->insert(first + base, row(base)); }); live[base] = 0;
    }
    for (uint32_t r = 0; r < rows; ++r) if (live[r]) require(index->erase(first + r), "empty-set delete failed");
    const auto empty = index->query(query, config);
    require(index->size() == 0 && empty.hits.empty() && empty.logical_fetch_stages == 0 && empty.backend_batch_calls == 0,
            "empty mutable live set fabricated results/I/O");
    const auto empty_checkpoint = (root / "empty checkpoint").string(); index->checkpoint(empty_checkpoint);
    auto empty_reopened = MutableIndex<T>::reopen(empty_checkpoint, (root / "empty reopened.heap").string(), {2});
    require(empty_reopened->size() == 0 && empty_reopened->retained_heap_records() == rows &&
            empty_reopened->query(query, config).hits.empty(), "empty checkpoint lost tombstones/retained records");
    rejects([&] { empty_reopened->insert(first + base, row(base)); });
    index.reset(); limited.reset();
    for (size_t i = 0; i < index_component_suffixes.size(); ++i)
        require(artifact_digest(prefix + index_component_suffixes[i]) == source_identity[i], "session changed immutable source component");
    require(artifact_digest(prefix + ".manifest") == source_identity.back(), "session changed immutable manifest");
    auto original = NativeIndex<T>::open(prefix, {2});
    std::fill(live.begin(), live.end(), 0); std::fill_n(live.begin(), base, 1);
    require(original->query(query, config).hits == oracle(), "source query changed after mutation session destruction");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected owned fixture directory");
        const auto root = std::filesystem::path(argv[1]) / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        exercise<float>(root / "float", DataType::Float);
        exercise<int8_t>(root / "int8", DataType::Int8);
        exercise<uint8_t>(root / "uint8", DataType::UInt8);
        std::cout << "Mutable native IVF heap/insert/delete/query/checkpoint fixtures passed (no CLI/production qualification)\n";
        return 0;
    } catch (const diskann::ANNException& error) { std::cerr << error.message() << '\n'; return 1;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
