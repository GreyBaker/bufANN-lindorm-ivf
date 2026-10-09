#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace diskann::ivf {
struct CalibrationSplit {
    std::vector<uint32_t> sample_ids, query_ids; // Sorted, disjoint base source-row IDs.
};
CalibrationSplit sample_calibration_rows(uint32_t base_count, uint32_t sample_count,
    uint32_t query_count, uint64_t seed, uint64_t workspace_bytes);

// Exact squared-L2 top-k within the calibration sample. Returned IDs are sample
// slots, not source rows; equal distances are ordered by slot (sorted source ID).
std::vector<uint32_t> calibration_ground_truth(std::span<const float> samples,
    std::span<const float> queries, uint32_t dim, uint32_t topk,
    uint64_t max_distance_evaluations, uint64_t workspace_bytes);

struct ProbeEstimate {
    double threshold_squared = 0;
    uint32_t nprobe = 0;
    uint64_t sample_codes = 0;
    double scaled_codes = 0;
};
// One centroid distance per GT neighbour, INCLUDING repeated cluster IDs, so
// ceil(recall*k) counts neighbours, not distinct clusters. Include radius ties.
ProbeEstimate estimate_probes(std::span<const double> centroid_distances,
    std::span<const uint32_t> assignments, std::span<const uint32_t> occupancy,
    std::span<const uint32_t> ground_truth_slots, double recall, uint32_t base_count);

struct CandidateCost { uint32_t nlist = 0; double t1_us = 0, t2_us = 0; };
// Minimum mean T1+T2; smallest nlist breaks an exact cost tie.
uint32_t choose_nlist(std::span<const CandidateCost> costs);
} // namespace diskann::ivf
