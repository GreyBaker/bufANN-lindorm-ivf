#include "ivf_cli_execution.h"
#include "bufann/ivf_index.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
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
              << "\",\"data_type\":\"" << data_type_name(type) << "\"";
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

template<typename T>
void search_typed(const Arguments& args, DataType type) {
    validate_output_paths(args);
    args.reject_present({"--pq_prefix", "--full_data_file", "--insert_ids_file", "--delete_ids_file"},
                        "not used by native search-only execution");
    const auto budget = budget_bytes(args.real("--memory_budget_gb", 1.0));
    const auto start_open = Clock::now();
    auto index = NativeIndex<T>::open(args.require("--index_prefix"), {args.u32("--buffer_frames", 64), budget});
    const auto open_time = seconds(start_open);
    const auto& manifest = index->manifest();
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
    const auto query_bytes = index->estimated_query_bytes(config);
    uint64_t bytes = index->estimated_open_bytes();
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
    emit("workload", "query"); emit("dim", dim); emit("nlist", manifest.training.nlist);
    emit("index_prefix", args.require("--index_prefix")); emit("query_file", args.require("--query_file"));
    emit("base_count", manifest.training.base_count); emit("pq_chunks", manifest.chunks);
    emit("nprobe", config.nprobe); emit("recall_at", config.recall_at); emit("rerank_factor", config.rerank_factor);
    emit("query_threads", config.query_threads); emit("active_query_threads", workers); emit("query_count", queries.rows);
    emit("query_breadth", manifest.routing.query_breadth); emit("buffer_pool_frames", args.u32("--buffer_frames", 64));
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
    emit("cache_policy", "uncontrolled_after_full_bundle_hash");
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
    record << "}"; publish_record(args, record.str());
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
void execute_search(const Arguments& args) {
    check(args.require("--workload") == "search_only", "Mutation execution is not implemented; native execution supports search_only");
    switch (parse_data_type(args.get("--data_type", "float"))) {
        case DataType::Float: return search_typed<float>(args, DataType::Float);
        case DataType::Int8: return search_typed<int8_t>(args, DataType::Int8);
        case DataType::UInt8: return search_typed<uint8_t>(args, DataType::UInt8);
    }
    throw std::invalid_argument("Unsupported dtype");
}
} // namespace ivf_cli
