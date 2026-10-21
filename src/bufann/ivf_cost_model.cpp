#include "bufann/ivf_cost_model.h"
#include "bufann/ivf_numeric_internal.h"
#include "ivf_random_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

namespace diskann::ivf {
  namespace {
    void check(bool ok, const char* message) {
      if (!ok)
        throw std::invalid_argument(message);
    }
    uint64_t product(uint64_t a, uint64_t b) {
      check(!b || a <= UINT64_MAX / b, "Cost-model work/size overflow");
      return a * b;
    }
  }  // namespace
  CalibrationSplit sample_calibration_rows(uint32_t base_count,
                                           uint32_t sample_count,
                                           uint32_t query_count, uint64_t seed,
                                           uint64_t workspace_bytes) {
    const uint64_t count = uint64_t(sample_count) + query_count;
    check(sample_count && query_count && count <= base_count &&
              base_count <= INT32_MAX,
          "Invalid disjoint calibration sample/query counts");
    check(product(count, 8) <= workspace_bytes,
          "Calibration split exceeds workspace allowance");
    std::vector<uint32_t> ids(static_cast<size_t>(count));
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937_64 rng(seed);
    for (uint32_t r = static_cast<uint32_t>(count); r < base_count; ++r) {
      const auto slot = detail::bounded_random(rng, uint64_t(r) + 1);
      if (slot < count)
        ids[slot] = r;
    }
    for (uint32_t r = 0; r < count; ++r)
      std::swap(ids[r], ids[r + detail::bounded_random(rng, count - r)]);
    CalibrationSplit result;
    result.sample_ids.assign(ids.begin(), ids.begin() + sample_count);
    result.query_ids.assign(ids.begin() + sample_count, ids.end());
    std::sort(result.sample_ids.begin(), result.sample_ids.end());
    std::sort(result.query_ids.begin(), result.query_ids.end());
    return result;
  }
  std::vector<uint32_t> calibration_ground_truth(
      std::span<const float> samples, std::span<const float> queries,
      uint32_t dim, uint32_t topk, uint64_t max_distance_evaluations,
      uint64_t workspace_bytes) {
    check(dim && !samples.empty() && !queries.empty() &&
              samples.size() % dim == 0 && queries.size() % dim == 0,
          "Invalid calibration vector shapes");
    const uint64_t rows = samples.size() / dim,
                   query_count = queries.size() / dim;
    check(rows <= INT32_MAX && query_count <= INT32_MAX && topk && topk <= rows,
          "Invalid calibration top-k/population");
    check(product(rows, query_count) <= max_distance_evaluations,
          "Exact calibration GT exceeds work allowance");
    const auto cells = product(query_count, topk);
    check(cells <= SIZE_MAX / sizeof(uint32_t) &&
              product(cells, 4) <= workspace_bytes &&
              product(topk, 32) <= workspace_bytes - product(cells, 4),
          "Calibration GT exceeds workspace allowance");
    for (float value : samples)
      check(std::isfinite(value), "Nonfinite calibration sample");
    for (float value : queries)
      check(std::isfinite(value), "Nonfinite synthetic query");
    std::vector<uint32_t> result(static_cast<size_t>(cells));
    using Hit = std::pair<double, uint32_t>;
    for (uint32_t q = 0; q < query_count; ++q) {
      std::priority_queue<Hit> best;
      for (uint32_t r = 0; r < rows; ++r) {
        const double distance = detail::squared_l2_double_unchecked(
            queries.data() + size_t(q) * dim,
            samples.data() + size_t(r) * dim, dim);
        check(std::isfinite(distance), "Nonfinite calibration GT distance");
        const Hit hit{distance, r};
        if (best.size() < topk)
          best.push(hit);
        else if (hit < best.top()) {
          best.pop();
          best.push(hit);
        }
      }
      for (uint32_t i = topk; i; --i) {
        result[size_t(q) * topk + i - 1] = best.top().second;
        best.pop();
      }
    }
    return result;
  }
  ProbeEstimate estimate_probes(std::span<const double>   centroid_distances,
                                std::span<const uint32_t> assignments,
                                std::span<const uint32_t> occupancy,
                                std::span<const uint32_t> ground_truth_slots,
                                double recall, uint32_t base_count) {
    check(!centroid_distances.empty() &&
              centroid_distances.size() <= INT32_MAX &&
              centroid_distances.size() == occupancy.size() &&
              !assignments.empty() && assignments.size() <= base_count &&
              !ground_truth_slots.empty(),
          "Invalid probe-estimator shapes");
    check(std::isfinite(recall) && recall > 0 && recall <= 1,
          "Recall target must be in (0,1]");
    uint64_t count = 0;
    for (auto n : occupancy)
      count += n;
    check(count == assignments.size(),
          "Calibration occupancy does not cover sample");
    for (double distance : centroid_distances)
      check(std::isfinite(distance) && distance >= 0,
            "Invalid centroid distance");
    std::vector<double> ranks;
    ranks.reserve(ground_truth_slots.size());
    for (uint32_t slot : ground_truth_slots) {
      check(slot < assignments.size() && assignments[slot] < occupancy.size(),
            "Invalid GT slot/cluster label");
      ranks.push_back(centroid_distances[assignments[slot]]);
    }
    // Retain multiplicity: deduplicating neighbour clusters changes the recall
    // rank.
    const size_t rank =
        static_cast<size_t>(std::ceil(recall * ranks.size())) - 1;
    std::nth_element(ranks.begin(), ranks.begin() + rank, ranks.end());
    ProbeEstimate result;
    result.threshold_squared = ranks[rank];
    for (size_t c = 0; c < occupancy.size(); ++c)
      if (centroid_distances[c] <= result.threshold_squared) {
        ++result.nprobe;
        result.sample_codes += occupancy[c];
      }
    result.scaled_codes =
        double(result.sample_codes) * base_count / assignments.size();
    check(result.nprobe && std::isfinite(result.scaled_codes),
          "Invalid scaled scan estimate");
    return result;
  }
  uint32_t choose_nlist(std::span<const CandidateCost> costs) {
    check(!costs.empty(), "No eligible cost-model candidate");
    uint32_t              selected = 0;
    double                best = std::numeric_limits<double>::infinity();
    std::vector<uint32_t> seen;
    for (const auto& cost : costs) {
      check(cost.nlist && std::isfinite(cost.t1_us) && cost.t1_us >= 0 &&
                std::isfinite(cost.t2_us) && cost.t2_us >= 0 &&
                std::isfinite(cost.t1_us + cost.t2_us),
            "Invalid candidate cost");
      check(std::find(seen.begin(), seen.end(), cost.nlist) == seen.end(),
            "Duplicate cost-model candidate");
      seen.push_back(cost.nlist);
      const double total = cost.t1_us + cost.t2_us;
      if (total < best || (total == best && cost.nlist < selected)) {
        best = total;
        selected = cost.nlist;
      }
    }
    return selected;
  }
}  // namespace diskann::ivf
