#include "../ivf_insert_input.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
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
              "unexpected staging failure");
      return;
    }
    throw std::runtime_error("staging refusal not propagated");
  }
  template<class T>
  void ordered() {
    const std::array<uint32_t, 4> ids{17, 2, 9, 1};
    unsigned                      reads = 0, reported = 0;
    const auto bytes = ivf_cli::insert_staging_bytes(ids.size(), 3, sizeof(T));
    auto       values = ivf_cli::stage_insert_rows<T>(
        ids, 3, bytes,
        [&](uint32_t id, std::span<T> row) {
          require(id == ids[reads++], "staging reordered counted IDs");
          for (unsigned d = 0; d < 3; ++d)
            row[d] = T(id + d);
        },
        [&](uint64_t completed) {
          require(completed == ++reported,
                        "staging reported before row completion");
        });
    require(values.size() == 12 && reads == 4 && reported == 4,
            "staging lost rows/counts");
    for (size_t i = 0; i < ids.size(); ++i)
      for (unsigned d = 0; d < 3; ++d)
        require(values[i * 3 + d] == T(ids[i] + d),
                "staging changed coordinates");
    reads = reported = 0;
    rejects(
        [&] {
          ivf_cli::stage_insert_rows<T>(
              ids, 3, bytes - 1, [&](uint32_t, std::span<T>) { ++reads; },
              [&](uint64_t) { ++reported; });
        },
        "exceeds workspace");
    require(reads == 0 && reported == 0, "memory refusal invoked callbacks");
    const auto empty = ivf_cli::stage_insert_rows<T>(
        {}, 3, 0, [&](uint32_t, std::span<T>) { ++reads; },
        [&](uint64_t) { ++reported; });
    require(empty.empty() && reads == 0 && reported == 0,
            "empty staging allocated/invoked readers");
  }
  void failures() {
    const std::array<uint32_t, 3> ids{1, 2, 3};
    unsigned                      reads = 0, reported = 0;
    rejects(
        [&] {
          ivf_cli::stage_insert_rows<float>(
              ids, 3, 36,
              [&](uint32_t id, std::span<float> row) {
                ++reads;
                if (id == 2)
                  throw std::runtime_error("deliberate reader failure");
                std::fill(row.begin(), row.end(), 1.0f);
              },
              [&](uint64_t) { ++reported; });
        },
        "deliberate reader failure");
    require(reads == 2 && reported == 1,
            "reader fault admitted further staging/reported failed row");
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
      reads = reported = 0;
      rejects(
          [&] {
            ivf_cli::stage_insert_rows<float>(
                ids, 3, 36,
                [&](uint32_t, std::span<float> row) {
                  ++reads;
                  row[0] = value;
                },
                [&](uint64_t) { ++reported; });
          },
          "Nonfinite insert");
      require(reads == 1 && reported == 0,
              "nonfinite row reported or further rows read");
    }
    rejects(
        [&] {
          ivf_cli::stage_insert_rows<float>(
              ids, 3, 36,
              [](uint32_t, std::span<float> row) {
                std::fill(row.begin(), row.end(), 1.0f);
              },
              [](uint64_t) { throw std::runtime_error("progress failure"); });
        },
        "progress failure");
  }
}  // namespace
int main() {
  try {
    require(ivf_cli::parse_insert_input_policy("streaming") ==
                    ivf_cli::InsertInputPolicy::Streaming &&
                ivf_cli::parse_insert_input_policy("staged") ==
                    ivf_cli::InsertInputPolicy::Staged,
            "policy names changed");
    rejects([] { ivf_cli::parse_insert_input_policy("auto"); },
            "streaming or staged");
    rejects([] { ivf_cli::insert_staging_bytes(UINT64_MAX, 3, 4); },
            "overflow");
    rejects([] { ivf_cli::insert_staging_bytes(uint64_t(PTRDIFF_MAX), 1, 4); },
            "overflow");
    rejects([] { ivf_cli::insert_staging_bytes(1, 0, 4); },
            "invalid dimension");
    ordered<float>();
    ordered<int8_t>();
    ordered<uint8_t>();
    failures();
    std::cout << "Insertion staging order, complete validation, budgets and "
                 "failure cleanup passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
