#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "bufann/ivf_config.h"

namespace diskann::ivf {
struct TrainerLimits {
    uint32_t block_rows = 256;
    uint64_t max_distance_evaluations = 1000000000;
    uint64_t max_assignment_searches = 1000000000;
};

struct TrainingPass {
    double graph_seconds = 0, assignment_seconds = 0, mean_seconds = 0;
    double assigned_sse = 0;
};
struct TrainingStats {
    std::vector<double> objective_history; // Sample SSE before each mean update.
    std::vector<TrainingPass> passes; // max_reps + final assignment; final mean time is zero.
    double final_objective = 0;
    uint64_t empty_recoveries = 0;
    uint64_t estimated_bytes = 0; // Workspace + conservative thread-stack allowance.
    uint64_t assignment_searches = 0;
    uint64_t assignment_index_builds = 0;
    uint64_t assignment_index_refreshes = 0, reinserted_centroids = 0;
    uint64_t seeded_searches = 0, seed_fallbacks = 0;
    uint64_t graph_distance_evaluations = 0; // Assignment traversal only, not graph construction/update.
    uint32_t final_unreachable = 0, empty_sample_clusters = 0, largest_sample_cluster = 0;
    // Exact mode includes all distance evaluations. HNSW mode leaves this
    // unavailable rather than misreporting searches as distance evaluations.
    std::optional<uint64_t> distance_evaluations;
    bool approximate_assignment = false;
    double elapsed_seconds = 0;
};
struct CentroidTraining : TrainingStats {
    uint32_t dim = 0;
    uint32_t nlist = 0;
    std::vector<float> centroids; // list ID * dim + coordinate
    std::vector<uint32_t> sampled_ids;
    std::vector<uint32_t> occupancy; // Final sample occupancy, not base-list sizes.
    std::vector<uint32_t> assignments; // Final centroid per sampled_ids slot, not source-row labels.
};

// Deterministic reservoir sample of initial base only; exact, blocked Lloyd
// assignment. Serial ordered reduction makes results independent of workers.
// memory_budget_gb bounds the preflight estimate, not measured process RSS.
CentroidTraining train_centroids(const std::string& vectors_path,
                                const BuildConfig& config,
                                const TrainerLimits& limits = {});
} // namespace diskann::ivf
