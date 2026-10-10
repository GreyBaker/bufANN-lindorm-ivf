#include "bufann/ivf_cost_model.h"
#include "bufann/ivf_lists.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace diskann::ivf;
namespace {
  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<class F>
  void rejects(F f) {
    bool failed = false;
    try {
      f();
    } catch (const std::exception&) {
      failed = true;
    }
    require(failed, "Expected cost-model rejection");
  }
  void sampling() {
    const auto a = sample_calibration_rows(256, 128, 16, 17, 4096);
    const auto b = sample_calibration_rows(256, 128, 16, 17, 4096);
    require(a.sample_ids == b.sample_ids && a.query_ids == b.query_ids,
            "Split is not seeded deterministic");
    auto all = a.sample_ids;
    all.insert(all.end(), a.query_ids.begin(), a.query_ids.end());
    std::sort(all.begin(), all.end());
    require(all.back() < 256 &&
                std::adjacent_find(all.begin(), all.end()) == all.end(),
            "Calibration overlap/tail ID");
    require(sample_calibration_rows(256, 128, 16, 18, 4096).sample_ids !=
                a.sample_ids,
            "Seed ignored");
    const auto full = sample_calibration_rows(144, 128, 16, 17, 4096);
    auto       ids = full.sample_ids;
    ids.insert(ids.end(), full.query_ids.begin(), full.query_ids.end());
    std::sort(ids.begin(), ids.end());
    require(ids.front() == 0 && ids.back() == 143, "Full split omitted rows");
    rejects([] { sample_calibration_rows(100, 100, 1, 0, 4096); });
    rejects([] { sample_calibration_rows(100, 50, 1, 0, 1); });
    rejects([] { sample_calibration_rows(UINT32_MAX, 50, 1, 0, 4096); });
  }
  void gt_and_probes() {
    const std::vector<float> samples{-1, 1, 3, 7, 9, 11}, queries{0, 8};
    const auto gt = calibration_ground_truth(samples, queries, 1, 3, 12, 4096);
    require(gt == std::vector<uint32_t>({0, 1, 2, 3, 4, 5}),
            "Exact sample GT/tie order");
    rejects(
        [&] { calibration_ground_truth(samples, queries, 1, 3, 11, 4096); });
    rejects(
        [&] { calibration_ground_truth(samples, queries, 1, 7, 100, 4096); });
    rejects(
        [&] { calibration_ground_truth(samples, queries, 0, 3, 100, 4096); });
    const std::vector<double>   d{1, 4, 4, 25};
    const std::vector<uint32_t> assignment{0, 0, 1, 1, 2, 3},
        occupancy{2, 2, 1, 1}, neighbours{0, 1, 2};
    const auto low =
        estimate_probes(d, assignment, occupancy, neighbours, 2.0 / 3, 60);
    require(low.threshold_squared == 1 && low.nprobe == 1 &&
                low.sample_codes == 2 && low.scaled_codes == 20,
            "GT multiplicity or N/m scaling lost");
    const auto high =
        estimate_probes(d, assignment, occupancy, neighbours, 1, 60);
    require(high.threshold_squared == 4 && high.nprobe == 3 &&
                high.sample_codes == 5 && high.scaled_codes == 50,
            "Inclusive centroid-radius ties lost");
    rejects(
        [&] { estimate_probes(d, assignment, occupancy, neighbours, 0, 60); });
    rejects([&] {
      estimate_probes(d, assignment, occupancy, neighbours,
                      std::numeric_limits<double>::quiet_NaN(), 60);
    });
    rejects([&] {
      estimate_probes(d, assignment, occupancy, std::vector<uint32_t>{6}, 1,
                      60);
    });
    rejects([&] {
      estimate_probes(d, assignment, std::vector<uint32_t>{1, 1, 1, 1},
                      neighbours, 1, 60);
    });
    rejects(
        [&] { estimate_probes(d, assignment, occupancy, neighbours, 1, 5); });
    require(choose_nlist(std::vector<CandidateCost>{
                {8, 2, 2}, {4, 1, 3}, {16, 10, 1}}) == 4,
            "Cost minimum/tie policy");
    rejects([] { choose_nlist({}); });
    rejects(
        [] { choose_nlist(std::vector<CandidateCost>{{8, 1, 1}, {8, 2, 2}}); });
    rejects([] { choose_nlist(std::vector<CandidateCost>{{8, -1, 1}}); });
  }
  void scan_profile() {
    std::vector<uint8_t> codes(128 * 2);
    std::iota(codes.begin(), codes.end(), 0);
    std::vector<float> lut(512);
    std::iota(lut.begin(), lut.end(), 0);
    const auto profile =
        PostingLists::profile_scan(codes, 2, lut, 10, 3, 1 << 20);
    require(profile.scanned_codes == 384 && profile.per_code_us > 0 &&
                std::isfinite(profile.per_code_us),
            "W4 scan coefficient/work counter invalid");
    rejects([&] { PostingLists::profile_scan(codes, 2, lut, 10, 3, 1); });
    rejects([&] { PostingLists::profile_scan(codes, 2, lut, 10, 0, 1 << 20); });
    lut[0] = std::numeric_limits<float>::infinity();
    rejects([&] { PostingLists::profile_scan(codes, 2, lut, 10, 3, 1 << 20); });
  }
}  // namespace
int main() {
  try {
    sampling();
    gt_and_probes();
    scan_profile();
    std::cout << "Cost-model core fixtures passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
