#pragma once
#include "bufann/ivf_centroids.h"
#include "ivf_cli.h"

namespace ivf_cli {
inline void training_settings(std::ostream& out, bool& first, const diskann::ivf::BuildConfig& config) {
    using namespace diskann::ivf;
    bufann_bench::json_kv(out, first, "trainer_mode", trainer_mode_name(config.trainer_mode));
    bufann_bench::json_kv(out, first, "trainer_init", centroid_init_name(config.init));
    bufann_bench::json_kv(out, first, "training_breadth", config.training_breadth);
    bufann_bench::json_kv(out, first, "trainer_graph_reuse", config.trainer_mode == TrainerMode::AnnReuse || config.trainer_mode == TrainerMode::AnnReuseSeeded);
    bufann_bench::json_kv(out, first, "trainer_previous_assignment_seed", config.trainer_mode == TrainerMode::AnnReuseSeeded);
}
inline void training_metrics(std::ostream& out, bool& first, const diskann::ivf::TrainingStats& stats) {
    auto emit = [&](const char* key, const auto& value) { bufann_bench::json_kv(out, first, key, value); };
    emit("training_time_s", stats.elapsed_seconds); emit("training_estimated_bytes", stats.estimated_bytes);
    emit("training_assignment_searches", stats.assignment_searches);
    emit("training_graph_builds", stats.assignment_index_builds); emit("training_graph_refreshes", stats.assignment_index_refreshes);
    emit("training_reinserted_centroids", stats.reinserted_centroids);
    emit("training_seeded_searches", stats.seeded_searches); emit("training_seed_fallbacks", stats.seed_fallbacks);
    emit("training_assignment_graph_distances", stats.graph_distance_evaluations);
    emit("training_assignment_graph_distances_known", stats.assignment_index_refreshes > 0);
    emit("training_final_unreachable", stats.final_unreachable);
    emit("training_final_unreachable_known", stats.assignment_index_refreshes > 0);
    emit("training_final_assigned_sse", stats.final_objective); emit("training_empty_recoveries", stats.empty_recoveries);
    emit("training_empty_sample_clusters", stats.empty_sample_clusters); emit("training_largest_sample_cluster", stats.largest_sample_cluster);
    emit("training_objective_scope", stats.approximate_assignment ? "approximate_assigned_sample_sse_not_exact_nearest" : "exact_nearest_sample_sse");
    emit("training_iteration_policy", "fixed_max_reps_plus_final_assignment");
    if (!first) out << ',';
    first = false;
    out << "\"training_passes\":[";
    for (size_t i = 0; i < stats.passes.size(); ++i) {
        const auto& pass = stats.passes[i];
        out << (i ? "," : "") << "{\"pass\":" << i << ",\"final_assignment\":" << (i+1 == stats.passes.size() ? "true" : "false")
            << ",\"graph_time_s\":" << pass.graph_seconds << ",\"assignment_time_s\":" << pass.assignment_seconds
            << ",\"mean_time_s\":" << pass.mean_seconds << ",\"assigned_sse\":" << pass.assigned_sse << '}';
    }
    out << ']';
}
} // namespace ivf_cli
