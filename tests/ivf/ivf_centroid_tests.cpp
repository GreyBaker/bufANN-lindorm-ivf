#include "bufann/ivf_centroids.h"
#include "../../src/bufann/ivf_centroids_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace {
  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<typename F>
  void rejects(F&& operation) {
    bool rejected = false;
    try {
      operation();
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "expected trainer rejection");
  }
  template<typename T>
  T value(uint32_t r, uint32_t d) {
    return static_cast<T>(int(r % 4) * 20 + int((r + d) % 7) -
                          (std::is_same_v<T, uint8_t> ? 0 : 30));
  }
  template<typename T>
  void fixture(const std::filesystem::path& path, bool identical = false,
               bool bad_base = false) {
    std::ofstream  out(path, std::ios::binary);
    const uint32_t shape[]{160, 13};
    out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    for (uint32_t r = 0; r < 160; ++r)
      for (uint32_t d = 0; d < 13; ++d) {
        T x = identical ? T(7) : value<T>(r, d);
        if constexpr (std::is_same_v<T, float>)
          if (r >= 120 || (bad_base && r == 0))
            x = std::numeric_limits<float>::quiet_NaN();
        out.write(reinterpret_cast<const char*>(&x), sizeof(x));
      }
    require(bool(out), "trainer fixture write failed");
  }
  template<typename T>
  void exercise(const std::filesystem::path& path,
                diskann::ivf::DataType       type) {
    using namespace diskann::ivf;
    fixture<T>(path);
    BuildConfig config;
    config.data_type = type;
    config.dim = 13;
    config.base_count = 120;
    config.nlist = 4;
    config.sample_rate = .75;
    config.max_reps = 6;
    config.seed = 12345;
    config.build_threads = 1;
    const auto trained = train_centroids(path.string(), config, {7, 100000});
    config.build_threads = 4;
    const auto parallel = train_centroids(path.string(), config, {19, 100000});
    require(trained.centroids == parallel.centroids &&
                trained.sampled_ids == parallel.sampled_ids &&
                trained.objective_history == parallel.objective_history &&
                trained.final_objective == parallel.final_objective &&
                trained.occupancy == parallel.occupancy,
            "worker/block count changed training result");
    require(trained.sampled_ids.size() == 90 &&
                trained.centroids.size() == 52 &&
                std::set<uint32_t>(trained.sampled_ids.begin(),
                                   trained.sampled_ids.end())
                        .size() == 90 &&
                *std::max_element(trained.sampled_ids.begin(),
                                  trained.sampled_ids.end()) < 120,
            "invalid initial-base sample");
    require(std::accumulate(trained.occupancy.begin(), trained.occupancy.end(),
                            uint32_t(0)) == 90 &&
                trained.distance_evaluations == 90 * 4 * 7 &&
                trained.estimated_bytes > 0 && trained.elapsed_seconds >= 0,
            "invalid trainer diagnostics");
    double                objective = 0;
    std::vector<uint32_t> occupancy(4);
    for (uint32_t r : trained.sampled_ids) {
      double   best = std::numeric_limits<double>::infinity();
      uint32_t label = 0;
      for (uint32_t c = 0; c < 4; ++c) {
        double distance = 0;
        for (uint32_t d = 0; d < 13; ++d) {
          double diff =
              double(value<T>(r, d)) - trained.centroids[size_t(c) * 13 + d];
          distance += diff * diff;
        }
        if (distance < best) {
          best = distance;
          label = c;
        }
      }
      objective += best;
      ++occupancy[label];
    }
    require(
        objective == trained.final_objective && occupancy == trained.occupancy,
        "final diagnostics do not describe returned centroids");
    double previous = std::numeric_limits<double>::infinity();
    for (double x : trained.objective_history) {
      require(std::isfinite(x) && x <= previous + 1e-6,
              "exact objective increased");
      previous = x;
    }
    require(trained.final_objective <= previous + 1e-6,
            "final objective increased");
    ++config.seed;
    require(train_centroids(path.string(), config).sampled_ids !=
                trained.sampled_ids,
            "sampling ignored seed");
    config.nlist = 1;
    config.sample_rate = 1;
    const auto mean = train_centroids(path.string(), config);
    for (uint32_t d = 0; d < 13; ++d) {
      double expected = 0;
      for (uint32_t r = 0; r < 120; ++r)
        expected += value<T>(r, d);
      require(mean.centroids[d] == float(expected / 120),
              "centroid is not a float mean");
    }
    auto invalid = config;
    invalid.memory_budget_gb = 1e-9;
    rejects([&] { train_centroids(path.string(), invalid); });
    rejects([&] { train_centroids(path.string(), config, {1, 1}); });
    rejects([&] { train_centroids(path.string(), config, {0, 100000}); });
    invalid = config;
    invalid.base_count = 161;
    rejects([&] { train_centroids(path.string(), invalid); });
    invalid = config;
    invalid.nlist = 121;
    rejects([&] { train_centroids(path.string(), invalid); });
    invalid = config;
    invalid.dim = 12;
    rejects([&] { train_centroids(path.string(), invalid); });
    const auto same = path.string() + ".identical";
    fixture<T>(same, true);
    config.nlist = 4;
    const auto empty = train_centroids(same, config);
    require(empty.empty_recoveries == uint64_t(config.max_reps) * 3 &&
                empty.final_objective == 0 &&
                empty.occupancy == std::vector<uint32_t>({120, 0, 0, 0}) &&
                std::all_of(empty.centroids.begin(), empty.centroids.end(),
                            [](float x) { return x == 7; }),
            "identical-sample empty recovery invalid");
    if constexpr (std::is_same_v<T, float>) {
      const auto bad = path.string() + ".nonfinite";
      fixture<float>(bad, false, true);
      rejects([&] { train_centroids(bad, config); });
    }
    const auto truncated = path.string() + ".truncated";
    std::filesystem::copy_file(path, truncated);
    std::filesystem::resize_file(truncated, 9);
    rejects([&] { train_centroids(truncated, config); });
    const auto trailing = path.string() + ".trailing";
    std::filesystem::copy_file(path, trailing);
    {
      std::ofstream out(trailing, std::ios::binary | std::ios::app);
      out.put('x');
    }
    rejects([&] { train_centroids(trailing, config); });
    // Worker exceptions must propagate, not terminate the process or consume
    // partially written assignments. The internal injection seam is test-only.
    struct BrokenIndex : detail::AssignmentIndex {
      explicit BrokenIndex(bool invalid_id) : invalid(invalid_id) {
      }
      uint32_t assign(std::span<const float>, uint32_t,
                      uint64_t&) const override {
        if (invalid)
          return UINT32_MAX;
        throw std::runtime_error("injected worker failure");
      }
      bool invalid;
    };
    for (bool invalid : {false, true}) {
      const detail::AssignmentFactory factory = [&](std::span<const float>) {
        return std::make_unique<BrokenIndex>(invalid);
      };
      rejects([&] {
        detail::train_centroids_impl(path.string(), config, {1, 100000},
                                     &factory, 0);
      });
    }
  }
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    const auto root =
        std::filesystem::path(argv[1]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(root);
    exercise<float>(root / "float.bin", diskann::ivf::DataType::Float);
    exercise<int8_t>(root / "int8.bin", diskann::ivf::DataType::Int8);
    exercise<uint8_t>(root / "uint8.bin", diskann::ivf::DataType::UInt8);
    std::cout << "IVF exact centroid trainer tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
