#include "ivf_cli_execution.h"
#include "bufann/ivf_index.h"
#include "ivf_mixed_workload.h"
#include <omp.h>
#include <mkl_service.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>
#include <sys/resource.h>

namespace ivf_cli {
namespace {
using namespace diskann::ivf;
using Clock = std::chrono::steady_clock;
void check(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
uint64_t budget_bytes(double gb) {
    const long double bytes = static_cast<long double>(gb) * (1ull << 30);
    check(gb > 0 && bytes < static_cast<long double>(UINT64_MAX), "Invalid memory budget");
    return static_cast<uint64_t>(bytes);
}
void add(uint64_t& bytes, uint64_t count, uint64_t stride = 1) {
    check(!stride || count <= (UINT64_MAX - bytes) / stride, "CLI workspace overflow");
    bytes += count * stride;
    check(bytes <= SIZE_MAX, "CLI workspace exceeds size_t");
}
template<typename T>
void read_values(std::ifstream& in, std::vector<T>& values, uint64_t count) {
    check(count <= SIZE_MAX / sizeof(T) && count <= uint64_t(INT64_MAX) / sizeof(T), "Input payload too large");
    values.resize(static_cast<size_t>(count));
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    check(bool(in), "Input payload read failed");
}
double peak_process_rss_mb() {
    rusage usage{};
    check(getrusage(RUSAGE_SELF, &usage) == 0, "Cannot read process RSS");
    return double(usage.ru_maxrss) / 1024.0; // Linux KiB; lifetime high-water mark.
}
void begin_record(std::ostream& out, const char* phase, DataType type) {
    out << std::setprecision(12) << "{\"baseline\":\"IVF\",\"phase\":\"" << phase
              << "\",\"data_type\":\"" << data_type_name(type) << "\",\"metric\":\"l2\"";
}
void validate_output_paths(const Arguments& args) {
    for (const auto* key : {"--result_file", "--search_result_file"}) if (args.has(key))
        check(!std::filesystem::exists(std::filesystem::symlink_status(args.require(key))), "Output file already exists");
    if (args.has("--result_file") && args.has("--search_result_file"))
        check(std::filesystem::absolute(args.require("--result_file")).lexically_normal() !=
              std::filesystem::absolute(args.require("--search_result_file")).lexically_normal(), "Output paths must differ");
}
void publish_record(const Arguments& args, const std::string& record) {
    if (args.has("--result_file")) {
        claim_artifact(args.require("--result_file"));
        std::ofstream out(args.require("--result_file"), std::ios::binary | std::ios::trunc);
        out << record << '\n'; out.close(); check(bool(out), "Metrics file write failed");
    }
    std::cout << record << std::endl;
}
template<typename T>
void build_typed(const Arguments& args, const BuildConfig& config, const BinMetadata& vectors, const PQMetadata& pq) {
    validate_output_paths(args);
    const auto budget = budget_bytes(config.memory_budget_gb);
    HeapOptions heap{config.dim, args.u32("--id_capacity", vectors.rows),
                     args.u32("--page_size", 4096), args.u32("--buffer_frames", 64)};
    RouterOptions routing;
    routing.degree = args.u32("--router_degree", 16);
    routing.construction_breadth = args.u32("--construction_breadth", 200);
    routing.query_breadth = args.u32("--query_breadth", 128);
    routing.insert_breadth = args.u32("--insert_breadth", 64);
    routing.seed = args.u32("--router_seed", 100);
    routing.memory_budget_bytes = budget;
    const auto start = Clock::now();
    NativeIndex<T>::build(args.require("--index_prefix"), args.require("--data_file"),
        args.require("--pq_prefix") + "_pq_pivots.bin", args.require("--pq_prefix") + "_pq_compressed.bin",
        config, heap, routing);
    const auto elapsed = seconds(start);
    const auto peak_rss = peak_process_rss_mb();
    std::ostringstream record;
    begin_record(record, "build", config.data_type);
    bool first = false;
    auto emit = [&](const char* key, const auto& value) { bufann_bench::json_kv(record, first, key, value); };
    emit("workload", "build"); emit("index_prefix", args.require("--index_prefix"));
    emit("dim", config.dim); emit("source_rows", vectors.rows); emit("base_count", config.base_count);
    emit("nlist", config.nlist); emit("pq_chunks", pq.chunks); emit("build_threads", config.build_threads);
    emit("sample_rate", config.sample_rate); emit("max_reps", config.max_reps); emit("seed", config.seed);
    emit("id_capacity", heap.id_capacity); emit("page_size", heap.page_size); emit("buffer_frames", heap.buffer_frames);
    emit("router_degree", routing.degree); emit("construction_breadth", routing.construction_breadth);
    emit("query_breadth", routing.query_breadth); emit("insert_breadth", routing.insert_breadth); emit("router_seed", routing.seed);
    emit("build_time_s", elapsed); emit("peak_process_rss_mb", peak_rss);
    emit("memory_budget_bytes", budget); emit("training_assignment", "hnsw_rebuilt_each_pass");
    emit("pq_policy", "global_nonresidual_pq8_external_unverified");
    emit("publication", "manifest_last_nonatomic_no_fsync");
    record << "}"; publish_record(args, record.str());
}

struct WorkerStats {
    uint64_t scanned = 0, reranked = 0, stages = 0, batches = 0, pages = 0, short_rows = 0;
    double routing = 0, adc = 0, scan = 0, fetch = 0, rerank = 0;
    std::exception_ptr failure;
};

// S05 sequential foreground updates, not S06 scheduling. Full-source identity
// prevents changing row-ID meaning between build and insertion. Partial failures
// keep owned files but publish neither a checkpoint nor a success record.
struct UpdateStats {
    uint64_t inserted = 0, deleted = 0, missing = 0, bytes = 0;
    double insert_s = 0, delete_s = 0, checkpoint_s = 0;
};
template<typename T>
UpdateStats mixed_index(const Arguments& args, MutableIndex<T>& index, uint64_t budget, uint64_t resident_bytes,
                        std::span<const T> queries, const SearchConfig& config, MixedStats& mixed) {
    const auto& m = index.origin_manifest();
    args.reject_present({"--pq_prefix"}, "mixed updates use frozen bundle PQ");
    const auto source = args.require("--full_data_file");
    check(inspect_vectors(source, m.training.data_type, m.training.dim).rows == m.source_rows &&
          artifact_digest(source) == m.sources[0], "Mixed full-source identity mismatch");
    const auto scheduler = args.get("--scheduler", "weighted");
    check(scheduler == "weighted" || scheduler == "fixed_8_8", "Unknown mixed scheduler");
    const auto ratio = args.u32("--query_ratio", 80);
    check(ratio > 0 && ratio < 100, "Mixed query_ratio must be in [1,99]");
    MixedPlan plan; plan.fixed_8_8 = scheduler == "fixed_8_8";
    plan.pool_threads = args.u32("--pool_threads", 16); plan.seed = args.integer("--scheduler_seed", plan.seed);
    UpdateStats s; s.bytes = resident_bytes;
    for (auto key : {"--insert_ids_file", "--delete_ids_file"}) add(s.bytes, std::filesystem::file_size(args.require(key)), 16);
    check(s.bytes <= budget, "CLI mixed ID parsing exceeds allowance");
    const auto inserts = load_counted_ids(args.require("--insert_ids_file"), m.source_rows);
    const auto deletes = load_counted_ids(args.require("--delete_ids_file"), m.source_rows);
    plan.inserts = inserts.size(); plan.deletes = deletes.size();
    // Validate bounds BEFORE calling the legacy count helper's multiplication.
    check(plan.inserts + plan.deletes <= UINT64_MAX / 99, "Mixed query-count overflow");
    plan.queries = args.integer("--query_count", ann_bench::mixed_update_query_count(inserts.size(), deletes.size(), ratio));
    plan.validate();
    auto sorted = deletes; std::sort(sorted.begin(), sorted.end());
    for (auto id : inserts) check(id >= m.first_id && id < m.heap.id_capacity && !index.contains(id) &&
        !std::binary_search(sorted.begin(), sorted.end(), id), "Invalid/live/overlapping mixed insert ID");
    for (auto id : deletes) check(id >= m.first_id && id < m.heap.id_capacity && index.contains(id), "Mixed deletion must refer to a live ID");
    const auto initial_live = index.size(); const auto retained_before = index.estimated_retained_bytes();
    const auto query_bytes = index.estimated_query_bytes(config), mutation_bytes = index.estimated_mutation_bytes();
    add(s.bytes, plan.inserts * m.training.dim, sizeof(T)); add(s.bytes, plan.queries, 2 * sizeof(double));
    uint64_t worker_bytes = 8ull * 1024 * 1024; add(worker_bytes, query_bytes); add(worker_bytes, mutation_bytes);
    add(s.bytes, plan.pool_threads, worker_bytes);
    add(s.bytes, index.estimated_checkpoint_bytes()); add(s.bytes, 1024 * 1024);
    check(s.bytes <= budget, "CLI composed mixed workload exceeds allowance");
    std::vector<T> coordinates(size_t(plan.inserts) * m.training.dim);
    std::ifstream input(source, std::ios::binary);
    for (size_t i = 0; i < inserts.size(); ++i) {
        input.seekg(8 + uint64_t(inserts[i]) * m.training.dim * sizeof(T));
        input.read(reinterpret_cast<char*>(coordinates.data() + i * m.training.dim), size_t(m.training.dim) * sizeof(T));
        check(bool(input), "Mixed source row read failed");
    }
    for (auto v : coordinates) check(std::isfinite(double(v)), "Nonfinite mixed insert coordinate");
    const auto query_rows = queries.size() / m.training.dim;
    check(query_rows != 0, "Mixed queries require a nonempty source");
    std::atomic<uint64_t> missing{0};
    const auto runtime_threads = uint32_t(std::distance(std::filesystem::directory_iterator("/proc/self/task"), std::filesystem::directory_iterator{}));
    mixed = run_mixed_round(plan, {
        [&](uint64_t i) { index.insert(inserts[size_t(i)], {coordinates.data() + size_t(i) * m.training.dim, m.training.dim}, {mutation_bytes, false}); },
        [&](uint64_t i) { if (!index.erase(deletes[size_t(i)])) ++missing; },
        [&](uint64_t i) { (void)index.query(queries.subspan(size_t(i % query_rows) * m.training.dim, m.training.dim), config, {query_bytes, false}); }});
    mixed.runtime_threads_before_launch = runtime_threads;
    check(missing == 0 && index.size() == uint64_t(initial_live) + plan.inserts - plan.deletes, "Mixed round count/live set mismatch");
    for (auto id : inserts) check(index.contains(id), "Completed mixed insert missing at barrier");
    for (auto id : deletes) check(!index.contains(id), "Completed mixed delete live at barrier");
    check(artifact_digest(source) == m.sources[0], "Mixed source changed during execution");
    s.inserted = mixed.completed[0]; s.deleted = mixed.completed[1];
    s.insert_s = s.delete_s = mixed.wall_s; // Rates use full mixed foreground wall, not summed callback time.
    add(s.bytes, index.estimated_retained_bytes() - retained_before);
    check(s.bytes <= budget, "CLI post-growth mixed workload exceeds allowance");
    const auto start = Clock::now();
    index.checkpoint(args.require("--checkpoint_prefix"), index.estimated_checkpoint_bytes());
    s.checkpoint_s = seconds(start); return s;
}
void validate_driver_paths(const Arguments& args, bool updating) {
    validate_output_paths(args);
    std::vector<std::string> paths;
    for (auto key : {"--result_file", "--search_result_file"}) if (args.has(key)) paths.push_back(args.require(key));
    if (args.has("--private_heap_path")) {
        paths.push_back(args.require("--private_heap_path")); paths.push_back(args.require("--private_heap_path") + ".pq.bin");
    }
    if (updating) {
        const auto prefix = args.require("--checkpoint_prefix");
        paths.push_back(prefix + ".manifest");
        for (auto suffix : mutable_component_suffixes) paths.push_back(prefix + suffix);
        args.require("--private_heap_path");
    } else args.reject_present({"--checkpoint_prefix"}, "not used by search-only execution");
    std::vector<std::filesystem::path> normalized;
    for (const auto& path : paths) {
        check(!std::filesystem::exists(std::filesystem::symlink_status(path)), "Driver output artifact exists");
        const auto resolved = std::filesystem::weakly_canonical(std::filesystem::absolute(path));
        check(std::find(normalized.begin(), normalized.end(), resolved) == normalized.end(), "Driver output paths overlap");
        normalized.push_back(resolved);
    }
}
template<typename T> std::unique_ptr<MutableIndex<T>> open_mutable(const Arguments& args, uint64_t budget) {
    const MutableIndexOptions options{args.u32("--buffer_frames", 64), budget};
    const auto format = args.get("--index_format", "immutable");
    check(format == "immutable" || format == "mutable", "Unknown index_format");
    return format == "mutable" ? MutableIndex<T>::reopen(args.require("--index_prefix"), args.require("--private_heap_path"), options) :
        MutableIndex<T>::fork(args.require("--index_prefix"), args.require("--private_heap_path"), options);
}
template<typename T> const IndexManifest& cli_manifest(const NativeIndex<T>& index) { return index.manifest(); }
template<typename T> const IndexManifest& cli_manifest(const MutableIndex<T>& index) { return index.origin_manifest(); }
template<typename T> uint64_t cli_retained(const NativeIndex<T>& index) { return index.estimated_open_bytes(); }
template<typename T> uint64_t cli_retained(const MutableIndex<T>& index) { return index.estimated_retained_bytes(); }
template<typename T>
UpdateStats update_index(const Arguments& args, MutableIndex<T>& index, uint64_t budget, uint64_t resident_bytes) {
    const auto& m = index.origin_manifest();
    const auto workload = args.require("--workload");
    const bool insert = workload != "delete_only", erase = workload != "insert_only";
    args.reject_present({"--pq_prefix"}, "frozen bundle PQ is used for updates");
    if (!insert) args.reject_present({"--insert_ids_file"}, "not used by delete_only");
    if (!erase) args.reject_present({"--delete_ids_file"}, "not used by insert_only");
    const auto source = args.require("--full_data_file");
    const auto shape = inspect_vectors(source, m.training.data_type, m.training.dim);
    check(shape.rows == m.source_rows && artifact_digest(source) == m.sources[0], "Update full-source identity mismatch");
    // Counted-ID parser temporarily uses an unordered_set as well as the vector.
    // Conservatively budget both ID files before allocating either parser.
    UpdateStats s; s.bytes = resident_bytes;
    const auto retained_before = index.estimated_retained_bytes();
    for (auto key : {"--insert_ids_file", "--delete_ids_file"}) if (args.has(key))
        add(s.bytes, std::filesystem::file_size(args.require(key)), 16);
    add(s.bytes, m.training.dim, sizeof(T)); add(s.bytes, index.estimated_mutation_bytes());
    add(s.bytes, index.estimated_checkpoint_bytes()); add(s.bytes, 1024 * 1024);
    check(s.bytes <= budget, "CLI composed update exceeds workspace allowance");
    auto inserts = insert ? load_counted_ids(args.require("--insert_ids_file"), m.source_rows) : std::vector<uint32_t>{};
    auto deletes = erase ? load_counted_ids(args.require("--delete_ids_file"), m.source_rows) : std::vector<uint32_t>{};
    for (auto id : inserts) check(id >= m.first_id && id < m.heap.id_capacity &&
        !index.contains(id), "Insert ID is outside allowed range or already live");
    for (auto id : deletes) check(id >= m.first_id && id < m.heap.id_capacity, "Delete ID outside allowed range");
    auto sorted_deletes = deletes; std::sort(sorted_deletes.begin(), sorted_deletes.end());
    for (auto id : inserts) check(!std::binary_search(sorted_deletes.begin(), sorted_deletes.end(), id), "Insert/delete ID sets overlap");
    const auto identity = m.sources[0];
    std::ifstream input(source, std::ios::binary); std::vector<T> row(m.training.dim);
    auto read_row = [&](uint32_t id) {
        input.seekg(8 + uint64_t(id) * m.training.dim * sizeof(T));
        input.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
        check(bool(input), "Insert source row read failed");
        for (auto value : row) check(std::isfinite(double(value)), "Nonfinite insert coordinate");
    };
    // Validate all requested coordinates before starting mutations, using bounded scratch.
    for (auto id : inserts) read_row(id);
    auto start = Clock::now();
    for (auto id : deletes) { if (index.erase(id)) ++s.deleted; else ++s.missing; }
    s.delete_s = seconds(start); start = Clock::now();
    for (auto id : inserts) { read_row(id); index.insert(id, row, {index.estimated_mutation_bytes(), false}); ++s.inserted; }
    s.insert_s = seconds(start);
    check(artifact_digest(source) == identity, "Update source changed during execution");
    // Growth may consume retained list allowance; verify final composed budget too.
    add(s.bytes, index.estimated_retained_bytes() - retained_before);
    check(s.bytes <= budget, "CLI post-growth update exceeds workspace allowance");
    start = Clock::now(); index.checkpoint(args.require("--checkpoint_prefix"), index.estimated_checkpoint_bytes());
    s.checkpoint_s = seconds(start); return s;
}
template<typename T> void emit_update(std::ostream& record, bool& first, const Arguments& args,
                                    const MutableIndex<T>& index, const UpdateStats& s, const MixedStats* mixed = nullptr) {
    auto emit = [&](const char* key, const auto& value) { bufann_bench::json_kv(record, first, key, value); };
    emit("insert_count", s.inserted); emit("delete_count", s.deleted); emit("missing_delete_count", s.missing);
    emit("insert_time_s", s.insert_s); emit("delete_time_s", s.delete_s);
    emit("insert_qps", bufann_bench::safe_div(s.inserted, s.insert_s)); emit("delete_qps", bufann_bench::safe_div(s.deleted + s.missing, s.delete_s));
    emit("checkpoint_time_s", s.checkpoint_s); emit("checkpoint_prefix", args.require("--checkpoint_prefix"));
    emit("live_count", index.size()); emit("retained_heap_records", index.retained_heap_records());
    emit("estimated_update_workspace_bytes", s.bytes);
    if (!mixed) {
        emit("mutation_workers", 1); emit("scheduler", "sequential_delete_then_insert_no_query_overlap");
        emit("mutation_timing_scope", "foreground_loops_excludes_open_validation_hash_checkpoint_and_post_update_query");
    } else {
        const auto fixed = args.get("--scheduler", "weighted") == "fixed_8_8";
        const auto threads = args.u32("--pool_threads", 16);
        emit("scheduler", fixed ? "fixed_8_8" : "weighted_remaining_ops"); emit("foreground_workers", threads);
        emit("mutation_workers", fixed ? 8u : threads); emit("concurrent_query_workers", fixed ? 8u : threads);
        emit("worker_allocation", fixed ? "dedicated_8_mutation_8_query" : "shared_pool_not_additive");
        emit("scheduler_seed", args.integer("--scheduler_seed", ann_bench::WeightedUpdatePool::kDefaultSeed));
        emit("query_ratio", args.u32("--query_ratio", 80)); emit("mixed_wall_s", mixed->wall_s);
        emit("concurrent_query_count_policy", args.has("--query_count") ? "explicit_finite_count" : "floor_ratio_times_updates_over_100_minus_ratio");
        emit("concurrent_query_count", mixed->completed[2]); emit("concurrent_query_qps", bufann_bench::safe_div(mixed->completed[2], mixed->wall_s));
        emit("overlapping_query_count", mixed->overlapping_queries); emit("peak_active_mutations", mixed->peak_mutations); emit("peak_active_queries", mixed->peak_queries);
        emit("insert_callback_us_total", mixed->callback_us[0]); emit("delete_callback_us_total", mixed->callback_us[1]);
        emit("concurrent_query_callback_us_total", mixed->callback_us[2]);
        auto latencies = mixed->query_latency_us; std::sort(latencies.begin(), latencies.end());
        emit("concurrent_query_lat_avg_us", std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size());
        emit("concurrent_query_lat_p99_us", latencies[size_t(0.99 * (latencies.size() - 1))]);
        emit("mutation_timing_scope", "full_mixed_foreground_wall_includes_launch_join_excludes_setup_checkpoint_post_round_query");
        emit("concurrent_recall_policy", "not_scored_weakly_consistent_queries");
        emit("overlap_scope", "admitted_operation_intervals_including_monitor_native_lock_io_and_retirement_not_cpu_simultaneity");
        emit("post_round_barrier", "all_foreground_workers_joined_checkpoint_validated_before_scoring");
        emit("backend_bg_flush_workers", 0); emit("openmp_threads", 1); emit("mkl_threads", 1);
        emit("runtime_threads_before_mixed_launch", mixed->runtime_threads_before_launch);
    }
    emit("publication", "manifest_last_nonatomic_no_fsync");
    const auto& m = index.origin_manifest();
    emit("base_count", m.training.base_count); emit("source_rows", m.source_rows);
    emit("router_degree", m.routing.degree); emit("construction_breadth", m.routing.construction_breadth);
    emit("query_breadth", m.routing.query_breadth); emit("insert_breadth", m.routing.insert_breadth); emit("router_seed", m.routing.seed);
    emit("input_index_format", args.get("--index_format", "immutable"));
    const auto manifest = artifact_digest(args.require("--index_prefix") + ".manifest");
    emit("input_manifest_bytes", manifest.bytes); emit("input_manifest_hash_fnv1a", manifest.hash);
    emit("source_bytes", m.sources[0].bytes); emit("source_hash_fnv1a", m.sources[0].hash);
    emit("buffer_pool_frames", args.u32("--buffer_frames", 64));
    emit("cache_policy", "uncontrolled_after_full_source_and_bundle_hash_and_private_copy");
}
template<typename T> void mutate_typed(const Arguments& args, DataType type) {
    args.reject_present({"--query_file", "--gt_file", "--search_result_file", "--query_threads", "--recall_at", "--nprobe", "--rerank_factor"},
                        "not used by mutation-only execution; query the saved checkpoint separately");
    validate_driver_paths(args, true);
    const auto budget = budget_bytes(args.real("--memory_budget_gb", 1)); const auto start = Clock::now();
    auto index = open_mutable<T>(args, budget); const auto open_time = seconds(start);
    const auto& m = index->origin_manifest();
    check(!args.has("--dim") || args.u32("--dim") == m.training.dim, "CLI/manifest dimension mismatch");
    check(!args.has("--nlist") || args.u32("--nlist") == m.training.nlist, "CLI/manifest nlist mismatch");
    const auto s = update_index(args, *index, budget, index->estimated_retained_bytes());
    std::ostringstream record; const auto phase = args.require("--workload") == "insert_only" ? "insert" : "delete";
    begin_record(record, phase, type); bool first = false;
    bufann_bench::json_kv(record, first, "workload", phase);
    bufann_bench::json_kv(record, first, "dim", m.training.dim); bufann_bench::json_kv(record, first, "nlist", m.training.nlist);
    bufann_bench::json_kv(record, first, "pq_chunks", m.chunks); bufann_bench::json_kv(record, first, "open_time_s", open_time);
    bufann_bench::json_kv(record, first, "index_prefix", args.require("--index_prefix"));
    bufann_bench::json_kv(record, first, "memory_budget_bytes", budget);
    bufann_bench::json_kv(record, first, "peak_process_rss_mb", peak_process_rss_mb());
    emit_update(record, first, args, *index, s); record << "}"; publish_record(args, record.str());
}

template<typename T, typename Index>
void search_typed(const Arguments& args, DataType type, std::unique_ptr<Index> index, double open_time) {
    const bool is_mixed = args.require("--workload") == "mixed";
    const bool updating = args.require("--workload") == "search_after_update" || is_mixed;
    const auto budget = budget_bytes(args.real("--memory_budget_gb", 1.0));
    const auto& manifest = cli_manifest(*index);
    const auto dim = manifest.training.dim;
    check(!args.has("--dim") || args.u32("--dim") == dim, "CLI/manifest dimension mismatch");
    check(!args.has("--nlist") || args.u32("--nlist") == manifest.training.nlist, "CLI/manifest nlist mismatch");
    SearchConfig config{args.u32("--nprobe", 1), args.u32("--recall_at", 10),
                        args.real("--rerank_factor", 2.0), args.u32("--query_threads", 1)};
    config.validate(manifest.training.nlist);
    const auto queries = inspect_vectors(args.require("--query_file"), type, dim);
    const bool has_truth = args.has("--gt_file");
    BinMetadata truth{};
    if (has_truth) truth = inspect_truth(args.require("--gt_file"), queries.rows, config.recall_at);
    const uint32_t workers = std::min(config.query_threads, queries.rows);
    const uint64_t result_count = uint64_t(queries.rows) * config.recall_at;
    auto query_bytes = index->estimated_query_bytes(config);
    uint64_t bytes = cli_retained(*index);
    add(bytes, uint64_t(queries.rows) * dim, sizeof(T));
    add(bytes, result_count, sizeof(uint32_t) + sizeof(float));
    add(bytes, queries.rows, sizeof(double));
    add(bytes, uint64_t(truth.rows) * truth.cols, sizeof(uint32_t) + sizeof(float));
    add(bytes, truth.cols, sizeof(uint32_t)); // sorted GT row scratch
    add(bytes, workers, query_bytes); add(bytes, workers, 8ull * 1024 * 1024 + sizeof(WorkerStats));
    add(bytes, 1024 * 1024); // I/O streams, worker handles and control overhead
    check(bytes <= budget, "CLI composed search exceeds workspace allowance");

    std::ifstream input(args.require("--query_file"), std::ios::binary); input.seekg(8);
    std::vector<T> coordinates;
    read_values(input, coordinates, uint64_t(queries.rows) * dim);
    for (T value : coordinates) check(std::isfinite(double(value)), "Nonfinite query coordinate");
    std::vector<uint32_t> gold_ids;
    std::vector<float> gold_distances;
    std::vector<uint32_t> gold_row;
    if (has_truth) {
        std::ifstream gt(args.require("--gt_file"), std::ios::binary); gt.seekg(8);
        read_values(gt, gold_ids, uint64_t(truth.rows) * truth.cols);
        read_values(gt, gold_distances, uint64_t(truth.rows) * truth.cols);
        const uint64_t entries = uint64_t(truth.rows) * truth.cols;
        if (std::filesystem::file_size(args.require("--gt_file")) == 8 + entries * 12) {
            std::array<uint32_t, 4096> tags;
            for (uint64_t offset = 0; offset < entries; offset += tags.size()) {
                const auto count = static_cast<size_t>(std::min<uint64_t>(tags.size(), entries - offset));
                gt.read(reinterpret_cast<char*>(tags.data()), count * sizeof(uint32_t));
                check(bool(gt), "GT tags read failed");
                check(std::equal(tags.begin(), tags.begin() + count, gold_ids.begin() + offset),
                      "IVF GT tags must equal source row IDs");
            }
        }
        gold_row.reserve(truth.cols);
        for (uint32_t q = 0; q < truth.rows; ++q) {
            const auto offset = size_t(q) * truth.cols;
            for (uint32_t j = 0; j < truth.cols; ++j) {
                const auto distance = gold_distances[offset + j];
                check(std::isfinite(distance) && distance >= 0 &&
                    (!j || distance >= gold_distances[offset + j - 1]), "GT distances must be finite, nonnegative and sorted");
            }
            gold_row.assign(gold_ids.begin() + offset, gold_ids.begin() + offset + truth.cols);
            std::sort(gold_row.begin(), gold_row.end());
            check(std::adjacent_find(gold_row.begin(), gold_row.end()) == gold_row.end(), "Duplicate GT ID in row");
        }
    }
    UpdateStats updates; MixedStats mixed;
    if constexpr (std::is_same_v<Index, MutableIndex<T>>) {
        if (updating) {
            updates = is_mixed ? mixed_index(args, *index, budget, bytes, std::span<const T>(coordinates), config, mixed) : update_index(args, *index, budget, bytes);
            // Account for additional retained list capacity after insertion.
            bytes = std::max(bytes, updates.bytes);
            const auto post_query_bytes = index->estimated_query_bytes(config);
            if (post_query_bytes > query_bytes) add(bytes, workers, post_query_bytes - query_bytes);
            query_bytes = post_query_bytes;
            check(bytes <= budget, "CLI post-update search exceeds workspace allowance");
        }
    }
    std::vector<uint32_t> ids(result_count, UINT32_MAX);
    std::vector<float> distances(result_count, std::numeric_limits<float>::infinity());
    std::vector<double> latency(queries.rows);
    std::vector<WorkerStats> stats(workers);
    std::atomic<uint32_t> next{0};
    std::atomic_bool failed{false};
    const auto start = Clock::now();
    {
        // jthread joins even if later thread construction fails. No shared ef setter.
        std::vector<std::jthread> threads;
        threads.reserve(workers);
        for (uint32_t worker = 0; worker < workers; ++worker) threads.emplace_back([&, worker] {
            auto& s = stats[worker];
            try {
                while (!failed.load(std::memory_order_relaxed)) {
                    const auto q = next.fetch_add(1, std::memory_order_relaxed);
                    if (q >= queries.rows) break;
                    const auto query_start = Clock::now();
                    auto result = index->query({coordinates.data() + size_t(q) * dim, dim}, config, {query_bytes, false});
                    latency[q] = seconds(query_start) * 1e6;
                    s.scanned += result.scanned_codes; s.reranked += result.reranked_vectors;
                    s.stages += result.logical_fetch_stages; s.batches += result.backend_batch_calls;
                    s.pages += result.heap_pages_read; s.short_rows += result.hits.size() < config.recall_at;
                    s.routing += result.routing_us; s.adc += result.adc_us; s.scan += result.scan_select_us;
                    s.fetch += result.fetch_us; s.rerank += result.rerank_us;
                    for (size_t j = 0; j < result.hits.size(); ++j) {
                        ids[size_t(q) * config.recall_at + j] = result.hits[j].id;
                        const double distance = result.hits[j].squared_distance;
                        distances[size_t(q) * config.recall_at + j] = distance > std::numeric_limits<float>::max() ?
                            std::numeric_limits<float>::infinity() : static_cast<float>(distance);
                    }
                }
            } catch (...) { s.failure = std::current_exception(); failed.store(true, std::memory_order_relaxed); }
        });
    }
    const auto query_time = seconds(start);
    WorkerStats total;
    for (const auto& s : stats) {
        if (s.failure) std::rethrow_exception(s.failure);
        total.scanned += s.scanned; total.reranked += s.reranked; total.stages += s.stages;
        total.batches += s.batches; total.pages += s.pages; total.short_rows += s.short_rows;
        total.routing += s.routing; total.adc += s.adc; total.scan += s.scan;
        total.fetch += s.fetch; total.rerank += s.rerank;
    }
    double recall = 0;
    if (has_truth) {
        uint64_t matches = 0;
        for (uint32_t q = 0; q < queries.rows; ++q) {
            const auto offset = size_t(q) * truth.cols;
            uint32_t end = config.recall_at;
            while (end < truth.cols && gold_distances[offset + end] == gold_distances[offset + config.recall_at - 1]) ++end;
            gold_row.assign(gold_ids.begin() + offset, gold_ids.begin() + offset + end);
            std::sort(gold_row.begin(), gold_row.end());
            for (uint32_t j = 0; j < config.recall_at; ++j) {
                const auto id = ids[size_t(q) * config.recall_at + j];
                if (id != UINT32_MAX && std::binary_search(gold_row.begin(), gold_row.end(), id)) ++matches;
            }
        }
        recall = 100.0 * double(matches) / double(result_count);
    }
    if (args.has("--search_result_file")) {
        const auto path = args.require("--search_result_file");
        claim_artifact(path);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&queries.rows), 4);
        out.write(reinterpret_cast<const char*>(&config.recall_at), 4);
        // SPFresh results are row-major interleaved ID/float-distance pairs;
        // unlike GT, which stores separate ID and distance planes.
        for (size_t i = 0; i < ids.size(); ++i) {
            out.write(reinterpret_cast<const char*>(&ids[i]), 4);
            out.write(reinterpret_cast<const char*>(&distances[i]), 4);
        }
        out.close(); check(bool(out), "Result file write failed");
    }
    std::sort(latency.begin(), latency.end());
    const double avg_latency = std::accumulate(latency.begin(), latency.end(), 0.0) / latency.size();
    const auto peak_rss = peak_process_rss_mb();
    auto percentile = [&](double p) { return latency[static_cast<size_t>(p * double(latency.size() - 1))]; };
    std::ostringstream record;
    begin_record(record, "query", type);
    bool first = false;
    auto emit = [&](const char* key, const auto& value) { bufann_bench::json_kv(record, first, key, value); };
    emit("workload", is_mixed ? "mixed" : (updating ? "search_after_update" : "query")); emit("dim", dim); emit("nlist", manifest.training.nlist);
    emit("index_prefix", args.require("--index_prefix")); emit("query_file", args.require("--query_file"));
    if (!updating) emit("base_count", manifest.training.base_count);
    emit("pq_chunks", manifest.chunks);
    emit("nprobe", config.nprobe); emit("recall_at", config.recall_at); emit("rerank_factor", config.rerank_factor);
    emit("query_threads", config.query_threads); emit("active_query_threads", workers); emit("query_count", queries.rows);
    if (!updating) {
        emit("query_breadth", manifest.routing.query_breadth); emit("buffer_pool_frames", args.u32("--buffer_frames", 64));
    }
    emit("open_time_s", open_time); emit("query_time_s", query_time);
    emit("query_qps", bufann_bench::safe_div(queries.rows, query_time));
    emit("query_lat_avg_us", avg_latency); emit("query_lat_p50_us", percentile(0.5)); emit("query_lat_p90_us", percentile(0.9));
    emit("query_lat_p95_us", percentile(0.95)); emit("query_lat_p99_us", percentile(0.99)); emit("query_lat_p999_us", percentile(0.999));
    emit("scanned_codes", total.scanned); emit("reranked_vectors", total.reranked);
    emit("logical_fetch_stages", total.stages); emit("backend_batch_calls", total.batches); emit("heap_pages_read", total.pages);
    emit("queries_with_short_results", total.short_rows);
    emit("routing_us_total", total.routing); emit("adc_us_total", total.adc); emit("scan_select_us_total", total.scan);
    emit("fetch_us_total", total.fetch); emit("rerank_us_total", total.rerank);
    emit("estimated_workspace_bytes", bytes); emit("memory_budget_bytes", budget);
    emit("peak_process_rss_mb", peak_rss);
    if (!updating) emit("cache_policy", "uncontrolled_after_full_bundle_hash");
    emit("query_timing_scope", "workers_start_join_and_result_collection_excludes_gt_and_export");
    emit("stage_timing_scope", "summed_per_query_wall_us_not_run_wall_or_cpu");
    emit("latency_percentile_policy", "floor_p_times_n_minus_one");
    emit("heap_pages_read_scope", "query_backend_page_reads_not_device_requests");
    emit("result_format", "spfresh_interleaved_uint32_float32_squared_l2");
    if (has_truth) {
        emit("recall", recall); emit("recall_policy", "percent_tie_aware_gt_expansion");
        emit("gt_id_policy", "source_row_ids_optional_identity_tags");
    }
    if (args.has("--search_result_file")) emit("search_result_file", args.require("--search_result_file"));
    if constexpr (std::is_same_v<Index, MutableIndex<T>>) {
        if (!updating) emit("live_count", index->size());
        emit("index_format", "mutable");
        if (updating) emit_update(record, first, args, *index, updates, is_mixed ? &mixed : nullptr);
    }
    record << "}"; publish_record(args, record.str());
}
template<typename T> void driver_typed(const Arguments& args, DataType type) {
    const auto workload = args.require("--workload");
    const bool mixed = workload == "mixed";
    check(workload == "search_only" || workload == "insert_only" || workload == "delete_only" || workload == "search_after_update" || mixed, "Unsupported native workload");
    if (!mixed) args.reject_present({"--scheduler", "--pool_threads", "--query_ratio", "--query_count", "--scheduler_seed"}, "mixed-only setting");
    else { omp_set_num_threads(1); mkl_set_num_threads(1); }
    if (workload == "insert_only" || workload == "delete_only") return mutate_typed<T>(args, type);
    const bool updating = workload == "search_after_update" || mixed;
    validate_driver_paths(args, updating);
    if (!updating) args.reject_present({"--pq_prefix", "--full_data_file", "--insert_ids_file", "--delete_ids_file"}, "not used by search-only execution");
    const auto format = args.get("--index_format", "immutable");
    check(format == "immutable" || format == "mutable", "Unknown index_format");
    const auto budget = budget_bytes(args.real("--memory_budget_gb", 1)); const auto start = Clock::now();
    if (updating || format == "mutable") {
        auto index = open_mutable<T>(args, budget);
        return search_typed<T>(args, type, std::move(index), seconds(start));
    }
    args.reject_present({"--private_heap_path"}, "not used by immutable search");
    auto index = NativeIndex<T>::open(args.require("--index_prefix"), {args.u32("--buffer_frames", 64), budget});
    return search_typed<T>(args, type, std::move(index), seconds(start));
}
} // namespace

void execute_build(const Arguments& args, const diskann::ivf::BuildConfig& config,
                   const diskann::ivf::BinMetadata& vectors, const diskann::ivf::PQMetadata& pq) {
    switch (config.data_type) {
        case DataType::Float: return build_typed<float>(args, config, vectors, pq);
        case DataType::Int8: return build_typed<int8_t>(args, config, vectors, pq);
        case DataType::UInt8: return build_typed<uint8_t>(args, config, vectors, pq);
    }
    throw std::invalid_argument("Unsupported dtype");
}
void execute_driver(const Arguments& args) {
    switch (parse_data_type(args.get("--data_type", "float"))) {
        case DataType::Float: return driver_typed<float>(args, DataType::Float);
        case DataType::Int8: return driver_typed<int8_t>(args, DataType::Int8);
        case DataType::UInt8: return driver_typed<uint8_t>(args, DataType::UInt8);
    }
    throw std::invalid_argument("Unsupported dtype");
}
} // namespace ivf_cli
