#include "../ivf_query_warmup.h"

#include <array>
#include <iostream>
#include <string>

namespace {
  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<class F>
  void rejects(F operation, const char* expected) {
    try {
      operation();
    } catch (const std::exception& error) {
      require(std::string(error.what()).find(expected) != std::string::npos,
              "wrong warmup failure propagated");
      return;
    }
    throw std::runtime_error("warmup failure not propagated");
  }
  template<class T>
  void conversion(std::span<const T> expected) {
    const std::array<float, 8> centroids{-300, 300,   -0.5f, 0.5f,
                                         3.2f, -3.2f, 1,     2};
    uint32_t                   seen = 0, reported = 0;
    const auto                 completed = ivf_cli::run_centroid_warmup<T>(
        centroids, 2, 4,
        [&](std::span<const T> row) {
          require(row.size() == 2 && row[0] == expected[2 * seen] &&
                                      row[1] == expected[2 * seen + 1],
                                  "warmup dtype rounding/clamping changed");
          ++seen;
        },
        [&](uint32_t n) {
          require(n == ++reported && reported == seen,
                                  "warmup counted admission");
        });
    require(completed == 4 && seen == 4 && reported == 4,
            "warmup lost completed queries");
  }
  void selection_and_failures() {
    const std::array<float, 4> centroids{0, 1, 2, 3};
    const std::array<float, 6> expected{0, 0, 1, 2, 2, 3};
    uint32_t                   seen = 0;
    ivf_cli::run_centroid_warmup<float>(
        centroids, 1, 6,
        [&](std::span<const float> row) {
          require(row[0] == expected[seen++], "centroid grid changed");
        },
        [](uint32_t) {});
    require(seen == 6, "repeated centroids silently reduced warmup volume");
    seen = 0;
    require(ivf_cli::run_centroid_warmup<float>(
                {}, 0, 0, [&](std::span<const float>) { ++seen; },
                [&](uint32_t) { ++seen; }) == 0 &&
                seen == 0,
            "zero control accessed centroids or callbacks");
    rejects(
        [&] {
          ivf_cli::run_centroid_warmup<float>(
              centroids, 0, 1, [](auto) {}, [](auto) {});
        },
        "centroid shape");
    rejects(
        [&] {
          ivf_cli::run_centroid_warmup<float>(
              centroids, 3, 1, [](auto) {}, [](auto) {});
        },
        "centroid shape");
    rejects([] { ivf_cli::validate_warmup_count(100001); }, "[0, 100000]");
    uint32_t reported = 0;
    seen = 0;
    rejects(
        [&] {
          ivf_cli::run_centroid_warmup<float>(
              centroids, 1, 4,
              [&](auto) {
                if (++seen == 2)
                  throw std::runtime_error("query fault");
              },
              [&](uint32_t) { ++reported; });
        },
        "query fault");
    require(seen == 2 && reported == 1,
            "query failure continued/counted failed query");
    const std::array<float, 2> invalid{1,
                                       std::numeric_limits<float>::quiet_NaN()};
    seen = reported = 0;
    rejects(
        [&] {
          ivf_cli::run_centroid_warmup<float>(
              invalid, 1, 2, [&](auto) { ++seen; },
              [&](uint32_t) { ++reported; });
        },
        "Nonfinite warmup centroid");
    require(seen == 1 && reported == 1,
            "invalid centroid reached query callback");
    rejects(
        [&] {
          ivf_cli::run_centroid_warmup<float>(
              centroids, 1, 4, [](auto) {},
              [](uint32_t) { throw std::runtime_error("progress fault"); });
        },
        "progress fault");
  }
}  // namespace
int main() {
  try {
    const std::array<float, 8>   floats{-300, 300,   -0.5f, 0.5f,
                                      3.2f, -3.2f, 1,     2};
    const std::array<int8_t, 8>  signed_values{-128, 127, -1, 1, 3, -3, 1, 2};
    const std::array<uint8_t, 8> unsigned_values{0, 255, 0, 1, 3, 0, 1, 2};
    conversion<float>(floats);
    conversion<int8_t>(signed_values);
    conversion<uint8_t>(unsigned_values);
    selection_and_failures();
    std::cout << "Frozen-centroid warmup grid, dtype conversion, zero control, "
                 "counts and failure propagation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
