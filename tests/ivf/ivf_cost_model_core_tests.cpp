// Dependency-isolated coverage: this fixture links only ivf_cost_model_core.
#include "bufann/ivf_cost_model.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace diskann::ivf;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    require(failed, "Expected cost-model rejection");
}
}
int main() {
    try {
        const auto split = sample_calibration_rows(256, 128, 16, 17, 4096);
        const auto again = sample_calibration_rows(256, 128, 16, 17, 4096);
        require(split.sample_ids == again.sample_ids && split.query_ids == again.query_ids,
                "Seeded split is not deterministic");
        auto ids = split.sample_ids;
        ids.insert(ids.end(), split.query_ids.begin(), split.query_ids.end());
        std::sort(ids.begin(), ids.end());
        require(ids.size() == 144 && ids.back() < 256 && std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
                "Sample/query split overlaps or is out of bounds");
        rejects([] { sample_calibration_rows(100, 100, 1, 0, 4096); });
        rejects([] { sample_calibration_rows(100, 50, 1, 0, 1); });
        const std::vector<float> samples{-1, 1, 3, 7, 9, 11}, queries{0, 8};
        require(calibration_ground_truth(samples, queries, 1, 3, 12, 4096) ==
                std::vector<uint32_t>({0, 1, 2, 3, 4, 5}), "Exact GT/tie order changed");
        rejects([&] { calibration_ground_truth(samples, queries, 1, 3, 11, 4096); });
        auto nonfinite = samples;
        nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { calibration_ground_truth(nonfinite, queries, 1, 3, 12, 4096); });
        nonfinite[0] = std::numeric_limits<float>::infinity();
        rejects([&] { calibration_ground_truth(nonfinite, queries, 1, 3, 12, 4096); });
        const std::vector<double> distances{1, 4, 4, 25};
        const std::vector<uint32_t> assignments{0, 0, 1, 1, 2, 3}, occupancy{2, 2, 1, 1}, neighbours{0, 1, 2};
        const auto probe = estimate_probes(distances, assignments, occupancy, neighbours, 1, 60);
        require(probe.threshold_squared == 4 && probe.nprobe == 3 && probe.sample_codes == 5 && probe.scaled_codes == 50,
                "Inclusive radius ties or N/m scaling changed");
        rejects([&] { estimate_probes(distances, assignments, occupancy, neighbours,
                                    std::numeric_limits<double>::quiet_NaN(), 60); });
        rejects([&] { estimate_probes(distances, assignments, occupancy, neighbours, 1, 5); });
        require(choose_nlist(std::vector<CandidateCost>{{8, 2, 2}, {4, 1, 3}, {16, 10, 1}}) == 4,
                "Cost minimum/tie policy changed");
        rejects([] { choose_nlist({}); });
        rejects([] { choose_nlist(std::vector<CandidateCost>{{8, -1, 1}}); });
        std::cout << "Dependency-isolated cost-model core fixtures passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
