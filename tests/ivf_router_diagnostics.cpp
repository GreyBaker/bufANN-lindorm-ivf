#include "bufann/ivf_router.h"
#include "bufann/ivf_io.h"
#include "bufann/ivf_artifacts.h"
#include "ivf_cli.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/resource.h>

namespace {
using namespace diskann::ivf;
using Clock = std::chrono::steady_clock;
void check(bool ok, const char* why) { if (!ok) throw std::invalid_argument(why); }
void add(uint64_t& bytes, uint64_t count, uint64_t stride) {
    check(count <= (UINT64_MAX - bytes) / stride, "Router diagnostic workspace overflow"); bytes += count * stride;
}
template<class T> void typed(const ivf_cli::Arguments& args, DataType type) {
    const auto prefix = args.require("--index_prefix"), path = args.require("--query_file");
    if (args.has("--result_file")) check(!std::filesystem::exists(std::filesystem::symlink_status(args.require("--result_file"))), "Router diagnostic result exists");
    const double gb = args.real("--memory_budget_gb", 1);
    check(gb > 0 && static_cast<long double>(gb) * (1ull<<30) < UINT64_MAX, "Invalid diagnostic memory budget");
    const uint64_t budget = uint64_t(static_cast<long double>(gb) * (1ull<<30));
    auto router = CentroidRouter::load(prefix, budget);
    const auto shape = inspect_vectors(path, type, router->dim());
    const auto rows = std::min(shape.rows, args.u32("--query_rows", 100));
    const auto probe = args.u32("--nprobe", 10), workers = std::min(rows, args.u32("--query_threads", 1));
    const auto exact_work = args.integer("--evaluation_work_limit");
    check(rows > 0 && workers > 0 && workers <= 256 && probe > 0 && probe <= router->nlist() &&
          uint64_t(rows) * router->nlist() <= exact_work, "Invalid diagnostic query/probe/thread settings or exact work allowance");
    RoutingRequest request{parse_router_mode(args.get("--router_mode", router_mode_name(router->options().mode))),
        args.real("--router_rerank_factor", router->options().router_rerank_factor), args.u32("--router_breadth", router->options().query_breadth)};
    check(request.breadth > 0 && request.breadth <= INT32_MAX && request.rerank_factor >= 1, "Invalid diagnostic router settings");
    uint64_t bytes = router->estimated_bytes();
    add(bytes, uint64_t(rows) * router->dim(), sizeof(T) + sizeof(float));
    add(bytes, uint64_t(rows) * probe, sizeof(CentroidHit));
    add(bytes, rows, 256); add(bytes, router->nlist(), 64);
    add(bytes, workers, 8ull*1024*1024 + uint64_t(router->nlist()) * 128 + uint64_t(router->dim()) * 8);
    check(bytes <= budget && bytes <= SIZE_MAX, "Router diagnostic composed workspace exceeds allowance");
    const auto query_identity = artifact_digest(path), centroid_identity = artifact_digest(prefix + ".centroids.bin");
    std::vector<T> raw(size_t(rows) * router->dim());
    std::ifstream input(path, std::ios::binary); input.seekg(8);
    input.read(reinterpret_cast<char*>(raw.data()), raw.size()*sizeof(T)); check(bool(input), "Truncated diagnostic query payload");
    std::vector<float> queries(raw.begin(), raw.end());
    std::vector<std::vector<CentroidHit>> truth(rows);
    uint64_t assignment_agreement = 0;
    for (uint32_t q = 0; q < rows; ++q) {
        const auto vector = std::span<const float>(queries).subspan(size_t(q)*router->dim(), router->dim());
        truth[q] = router->search(vector, probe, true);
        assignment_agreement += router->assign(vector) == truth[q][0].list_id;
    }
    // Validate mode/components before launching; warmup is explicitly disclosed.
    (void)router->search({queries.data(), router->dim()}, probe, false, nullptr, request);
    struct Stats { uint64_t matches = 0, distances = 0, counted = 0; double projection = 0, graph = 0, rerank = 0; std::exception_ptr error; };
    std::vector<Stats> stats(workers); std::vector<double> latency(rows); std::atomic<uint32_t> next{0};
    auto start = Clock::now();
    {
        std::vector<std::jthread> threads;
        for (uint32_t w = 0; w < workers; ++w) threads.emplace_back([&,w] {
            try { for (;;) {
                const auto q = next.fetch_add(1); if (q >= rows) break;
                const auto vector = std::span<const float>(queries).subspan(size_t(q)*router->dim(), router->dim());
                RoutingStats counts; const auto begin = Clock::now();
                const auto hits = router->search(vector, probe, false, &counts, request);
                latency[q] = std::chrono::duration<double, std::micro>(Clock::now()-begin).count();
                for (auto h : hits) for (auto gt : truth[q]) if (h.list_id == gt.list_id) ++stats[w].matches;
                stats[w].distances += counts.distance_evaluations; stats[w].counted += counts.distance_count_available;
                stats[w].projection += counts.projection_us; stats[w].graph += counts.graph_us; stats[w].rerank += counts.centroid_rerank_us;
            } } catch (...) { stats[w].error = std::current_exception(); }
        });
    }
    const auto wall = std::chrono::duration<double>(Clock::now()-start).count(); Stats total;
    for (const auto& s : stats) {
        if (s.error) std::rethrow_exception(s.error);
        total.matches += s.matches; total.distances += s.distances; total.counted += s.counted;
        total.projection += s.projection; total.graph += s.graph; total.rerank += s.rerank;
    }
    check(artifact_digest(path) == query_identity && artifact_digest(prefix + ".centroids.bin") == centroid_identity,
        "Diagnostic inputs changed during evaluation");
    std::sort(latency.begin(), latency.end()); rusage usage{}; check(getrusage(RUSAGE_SELF, &usage) == 0, "Cannot read RSS");
    std::ostringstream out; bool first = true; out << std::setprecision(12) << '{';
    auto emit = [&](const char* key, const auto& value) { bufann_bench::json_kv(out, first, key, value); };
    emit("kind", "ivf_router_diagnostics"); emit("version", 1); emit("production_qualified", false);
    emit("data_type", data_type_name(type)); emit("dim", router->dim()); emit("nlist", router->nlist());
    emit("router_mode", router_mode_name(request.mode)); emit("assignment_router_mode", router_mode_name(router->options().mode));
    emit("router_breadth", request.breadth); emit("router_rerank_factor", request.rerank_factor);
    emit("router_degree", router->options().degree); emit("construction_breadth", router->options().construction_breadth);
    emit("insert_breadth", router->options().insert_breadth); emit("router_seed", router->options().seed);
    emit("pca_retained_dim", router->projected_dim()); emit("pca_retained_variance", router->retained_variance()); emit("pca_degenerate", router->pca_degenerate());
    emit("patched_unreachable_before", router->unreachable_before(false)); emit("refined_unreachable_before", router->unreachable_before(true));
    emit("isolated_reinserted_nodes", router->reinserted_nodes(true));
    if (router->has_native_modes()) {
        emit("patched_unreachable_after", router->unreachable_after(false)); emit("refined_unreachable_after", router->unreachable_after(true));
        emit("patched_zero_indegree", router->zero_indegree(false)); emit("refined_zero_indegree", router->zero_indegree(true));
    }
    emit("nprobe", probe); emit("query_rows", rows); emit("query_threads", workers);
    emit("centroid_recall_fraction", double(total.matches)/double(uint64_t(rows)*probe));
    emit("assignment_agreement_fraction", double(assignment_agreement)/rows);
    emit("assignment_scope", "persisted_insert_default_on_first_query_rows_not_base_assignment_quality");
    emit("query_time_s", wall); emit("router_qps", rows/wall); emit("mean_us", std::accumulate(latency.begin(), latency.end(), 0.0)/rows);
    emit("p99_us", latency[size_t(.99*(rows-1))]); emit("projection_us_total", total.projection); emit("graph_us_total", total.graph);
    emit("centroid_rerank_us_total", total.rerank); emit("distance_evaluations", total.distances); emit("distance_counted_queries", total.counted);
    emit("distance_scope", "patched_graph_plus_full_centroid_rerank_exact_enumeration_hnswlib_unavailable");
    emit("query_bytes", query_identity.bytes); emit("query_hash_fnv1a", query_identity.hash);
    emit("centroid_bytes", centroid_identity.bytes); emit("centroid_hash_fnv1a", centroid_identity.hash);
    const auto descriptor = artifact_digest(prefix + ".router"); emit("router_descriptor_hash_fnv1a", descriptor.hash);
    emit("estimated_workspace_bytes", bytes); emit("peak_process_rss_mb", double(usage.ru_maxrss)/1024);
    emit("timing_scope", "joined_router_workers_excludes_exact_gt_assignment_hashes_load_includes_match_scoring");
    emit("warmup_policy", "full_artifact_validation_exact_gt_assignment_and_one_mode_query_no_cache_drop"); out << '}';
    if (args.has("--result_file")) {
        claim_artifact(args.require("--result_file")); std::ofstream result(args.require("--result_file")); result << out.str() << '\n';
        result.close(); check(bool(result), "Diagnostic record write failed");
    }
    std::cout << out.str() << '\n';
}
}
int main(int argc, char** argv) {
    try {
        ivf_cli::Arguments args(argc, argv, {"--index_prefix", "--query_file", "--data_type", "--query_rows", "--nprobe",
            "--query_threads", "--router_mode", "--router_breadth", "--router_rerank_factor", "--memory_budget_gb", "--evaluation_work_limit", "--result_file"});
        if (args.help) { std::cout << "Centroid-only routing ablation on one persisted graph family (not production acceptance).\n"
            "--index_prefix PREFIX --query_file TYPED_BIN --data_type float|int8|uint8\n"
            "--evaluation_work_limit N [--query_rows N] [--nprobe N] [--query_threads N]\n"
            "[--router_mode MODE] [--router_breadth N] [--router_rerank_factor F]\n"
            "[--memory_budget_gb F] [--result_file NEW_JSON]\n"; return 0; }
        switch (parse_data_type(args.get("--data_type", "float"))) {
            case DataType::Float: typed<float>(args, DataType::Float); break;
            case DataType::Int8: typed<int8_t>(args, DataType::Int8); break;
            case DataType::UInt8: typed<uint8_t>(args, DataType::UInt8); break;
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
