// Bounded training diagnostics on caller-supplied data, not canonical ANN metrics.
#include "../ivf_cli.h"
#include "../ivf_training_json.h"
#include "bufann/ivf_io.h"
#include "bufann/ivf_router.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sys/resource.h>

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
struct Evaluation { double sse = 0, assigned_sse = 0; uint32_t agreeing = 0; };
template<class T>
Evaluation evaluate(const std::string& path, const diskann::ivf::BuildConfig& config,
                const diskann::ivf::CentroidTraining& training, uint32_t rows) {
    std::ifstream input(path, std::ios::binary);
    std::vector<T> row(config.dim);
    Evaluation evaluation;
    for (uint32_t r = 0; r < rows; ++r) {
        // Same deterministic evenly spaced reservoir slots for both modes.
        const size_t slot = uint64_t(r) * training.sampled_ids.size() / rows;
        const uint32_t id = training.sampled_ids[slot];
        check(id < config.base_count, "Evaluation ID outside base prefix");
        input.seekg(8 + uint64_t(id) * config.dim * sizeof(T));
        input.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
        check(bool(input), "Evaluation vector read failed");
        for (T value : row) check(std::isfinite(double(value)), "Nonfinite evaluation vector");
        double best = std::numeric_limits<double>::infinity();
        double assigned = 0;
        for (uint32_t c = 0; c < config.nlist; ++c) {
            double distance = 0;
            for (uint32_t d = 0; d < config.dim; ++d) {
                const double delta = double(row[d]) - training.centroids[size_t(c) * config.dim + d];
                distance += delta * delta;
            }
            best = std::min(best, distance);
            if (c == training.assignments[slot]) assigned = distance;
        }
        evaluation.sse += best; evaluation.assigned_sse += assigned;
        evaluation.agreeing += assigned == best; // Exact distance tie is a valid assignment.
    }
    check(std::isfinite(evaluation.sse) && std::isfinite(evaluation.assigned_sse), "Nonfinite evaluation SSE");
    return evaluation;
}
template<class T>
void artifact(const std::string& path, uint32_t rows, uint32_t dim, const std::vector<T>& values) {
    const auto close = [](FILE* file) { std::fclose(file); };
    std::unique_ptr<FILE, decltype(close)> out(std::fopen(path.c_str(), "wbx"), close);
    check(bool(out), "Cannot exclusively create diagnostic artifact");
    const uint32_t header[]{rows, dim};
    check(std::fwrite(header, sizeof(header), 1, out.get()) == 1 &&
          std::fwrite(values.data(), sizeof(T), values.size(), out.get()) == values.size(),
          "Diagnostic artifact write failed");
    check(std::fclose(out.release()) == 0, "Diagnostic artifact close failed");
}
} // namespace

int main(int argc, char** argv) {
    using namespace diskann::ivf;
    try {
        const ivf_cli::Arguments args(argc, argv, {"--data_file", "--data_type", "--dim", "--base_count",
            "--nlist", "--sample_rate", "--max_reps", "--seed", "--build_threads", "--memory_budget_gb",
            "--mode", "--exact_work_limit", "--assignment_work_limit", "--evaluation_work_limit",
            "--evaluation_rows", "--router_degree", "--construction_breadth", "--insert_breadth",
            "--router_seed", "--output_prefix", "--init", "--training_breadth", "--training_rerank_factor", "--training_refine_breadth", "--routing_work_limit"});
        if (args.help) {
            std::cout << "Bounded trainer diagnostics: --mode exact|hnsw|ann-rebuild|ann-reuse|ann-reuse-seeded|ann-reuse-count|ann-reuse-seeded-count|lv|lv-plus\n"
                "  --init sample|random_partition --training_breadth N; use benchmark/IVF/qualify.py train --help\n";
            return 0;
        }
        BuildConfig config;
        config.data_type = parse_data_type(args.require("--data_type"));
        config.dim = args.u32("--dim"); config.base_count = args.u32("--base_count");
        config.nlist = args.u32("--nlist"); config.sample_rate = args.real("--sample_rate", 1);
        config.max_reps = args.u32("--max_reps", 15); config.seed = args.integer("--seed");
        config.build_threads = args.u32("--build_threads", 1);
        config.memory_budget_gb = args.real("--memory_budget_gb", 1);
        const auto mode = args.require("--mode"), path = args.require("--data_file");
        config.trainer_mode = parse_trainer_mode(mode == "hnsw" ? "ann-rebuild" : mode);
        config.init = parse_centroid_init(args.get("--init", "sample"));
        config.training_breadth = args.u32("--training_breadth", args.u32("--insert_breadth", 64));
        config.training_rerank_factor = args.real("--training_rerank_factor", 3);
        config.training_refine_breadth = args.u32("--training_refine_breadth", 128);
        config.validate();
        const auto shape = inspect_vectors(path, config.data_type, config.dim);
        check(config.base_count <= shape.rows, "Training base prefix exceeds source rows");
        const auto prefix = args.require("--output_prefix");
        for (const auto* suffix : {".centroids.bin", ".sample_ids.bin", ".assignments.bin", ".occupancy.bin"})
            check(!std::filesystem::exists(std::filesystem::symlink_status(prefix + suffix)),
                  "Diagnostic artifact already exists");
        const uint32_t rows = std::min(args.u32("--evaluation_rows"), config.training_count());
        const uint64_t evaluations = uint64_t(rows) * config.nlist;
        check(rows > 0 && evaluations <= args.integer("--evaluation_work_limit"),
              "Exact diagnostic evaluation exceeds declared work limit");
        const long double budget_value = static_cast<long double>(config.memory_budget_gb) * 1024 * 1024 * 1024;
        check(budget_value < static_cast<long double>(UINT64_MAX), "Diagnostic memory budget overflow");
        const uint64_t budget = static_cast<uint64_t>(budget_value);
        const uint64_t scratch = uint64_t(config.dim) * element_size(config.data_type);
        check(scratch < budget, "Evaluation scratch exceeds declared memory budget");
        // Reserve evaluation scratch even though the trainer's larger temporary
        // arrays have been released before evaluation. This is conservative.
        auto trainer_config = config;
        trainer_config.memory_budget_gb = double(budget - scratch) / (1024.0 * 1024 * 1024);
        TrainerLimits limits;
        limits.max_distance_evaluations = args.integer("--exact_work_limit");
        limits.max_assignment_searches = args.integer("--assignment_work_limit");
        RouterOptions routing;
        routing.native_work_limit = args.integer("--routing_work_limit",1000000000);
        routing.degree = args.u32("--router_degree", 16);
        routing.construction_breadth = args.u32("--construction_breadth", 200);
        routing.insert_breadth = args.u32("--insert_breadth", 64);
        routing.seed = args.u32("--router_seed", 100); routing.memory_budget_bytes = budget - scratch;
        const auto start = std::chrono::steady_clock::now();
        auto trained = train_centroids_ann(path, trainer_config, routing, limits);
        const auto eval_start = std::chrono::steady_clock::now();
        Evaluation evaluation;
        switch (config.data_type) {
            case DataType::Float: evaluation = evaluate<float>(path, config, trained, rows); break;
            case DataType::Int8: evaluation = evaluate<int8_t>(path, config, trained, rows); break;
            case DataType::UInt8: evaluation = evaluate<uint8_t>(path, config, trained, rows); break;
        }
        const double eval_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();
        artifact(prefix + ".centroids.bin", config.nlist, config.dim, trained.centroids);
        artifact(prefix + ".sample_ids.bin", config.training_count(), 1, trained.sampled_ids);
        artifact(prefix + ".assignments.bin", config.training_count(), 1, trained.assignments);
        artifact(prefix + ".occupancy.bin", config.nlist, 1, trained.occupancy);
        rusage usage{};
        check(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
        uint32_t empty = 0, largest = 0;
        for (auto count : trained.occupancy) { empty += count == 0; largest = std::max(largest, count); }
        std::cout << std::setprecision(17) << std::boolalpha << '{';
        bool first = true;
        auto emit = [&](const std::string& key, const auto& value) { bufann_bench::json_kv(std::cout, first, key, value); };
        emit("kind", "ivf_training_diagnostics"); emit("mode", mode);
        ivf_cli::training_settings(std::cout, first, config);
        emit("source_rows", shape.rows); emit("base_count", config.base_count); emit("dim", config.dim);
        emit("data_type", data_type_name(config.data_type)); emit("sample_rate", config.sample_rate);
        emit("nlist", config.nlist); emit("sample_count", config.training_count()); emit("seed", config.seed);
        emit("max_reps", config.max_reps); emit("build_threads", config.build_threads);
        emit("evaluation_time_s", eval_seconds);
        ivf_cli::training_metrics(std::cout, first, trained);
        emit("diagnostic_time_s", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        emit("peak_process_rss_mb", double(usage.ru_maxrss) / 1024);
        emit("rss_scope", "process_lifetime_including_evaluation_and_artifact_export");
        emit("estimated_training_and_eval_bytes", trained.estimated_bytes + scratch); emit("memory_budget_bytes", budget);
        emit("assignment_searches", trained.assignment_searches); emit("assignment_index_builds", trained.assignment_index_builds);
        emit("training_distance_evaluations_known", trained.distance_evaluations.has_value());
        if (trained.distance_evaluations) emit("training_distance_evaluations", *trained.distance_evaluations);
        emit("final_assignment_sample_sse", trained.final_objective);
        emit("exact_evaluation_sse", evaluation.sse); emit("evaluation_assigned_sse", evaluation.assigned_sse);
        emit("evaluation_assignment_agreement", double(evaluation.agreeing)/rows);
        emit("evaluation_rows", rows); emit("evaluation_distance_evaluations", evaluations);
        emit("evaluation_policy", "exact_nearest_centroid_evenly_spaced_training_reservoir_slots_not_holdout");
        emit("empty_sample_clusters", empty); emit("largest_sample_cluster", largest); emit("empty_recoveries", trained.empty_recoveries);
        if (!first) std::cout << ',';
        std::cout << "\"objective_history\":[";
        for (size_t i = 0; i < trained.objective_history.size(); ++i)
            std::cout << (i ? "," : "") << trained.objective_history[i];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n'; return 1;
    }
}
