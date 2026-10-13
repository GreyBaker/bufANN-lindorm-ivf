#include "../ivf_insert_workload.h"

#include <array>
#include <atomic>
#include <barrier>
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
              "wrong insertion failure propagated");
      return;
    }
    throw std::runtime_error("insertion failure not propagated");
  }
  void complete(uint32_t requested, uint64_t count) {
    std::vector<std::atomic<unsigned>> seen(count);
    std::array<std::atomic<bool>, 256> owned{};
    uint64_t                           reported = 0;
    const auto effective = ivf_cli::effective_insert_workers(count, requested);
    const auto result = ivf_cli::run_insert_round(
        count, requested,
        [&](uint32_t w, uint64_t i) {
          require(w < effective && !owned[w].exchange(true),
                  "worker scratch ownership lost");
          require(seen[i].fetch_add(1) == 0, "input position inserted twice");
          std::this_thread::yield();
          owned[w] = false;
        },
        [&](uint64_t completed) {
          require(completed == ++reported,
                  "progress counted admission/not completion");
        });
    require(result.completed == count && reported == count &&
                result.workers == effective && result.peak_active <= effective,
            "completion counts/worker bound changed");
    for (auto& visits : seen)
      require(visits == 1, "insertion lost an input position");
  }
  void failure(const char* stage, bool progress_error) {
    std::barrier          start(4);
    std::atomic<bool>     release{false};
    std::atomic<unsigned> admitted{0}, exited{0};
    const std::string     message = stage;
    rejects(
        [&] {
          ivf_cli::run_insert_round(
              1000, 4,
              [&](uint32_t, uint64_t i) {
                ++admitted;
                start.arrive_and_wait();
                if (i == 0) {
                  if (!progress_error)
                    throw std::runtime_error(message);
                } else {
                  while (!release)
                    std::this_thread::yield();
                }
              },
              [&](uint64_t) {
                if (progress_error)
                  throw std::runtime_error(message);
              },
              [&](auto callback) {
                return std::jthread([&, callback] {
                  callback();
                  ++exited;
                  release =
                      true;  // Failing worker exited only after cancellation.
                });
              });
        },
        stage);
    require(admitted == 4 && exited == 4,
            "failure admitted new work or returned before joins");
  }
  void launch_failure() {
    unsigned              launches = 0;
    std::atomic<unsigned> operations{0}, exited{0};
    rejects(
        [&] {
          ivf_cli::run_insert_round(
              100, 4, [&](uint32_t, uint64_t) { ++operations; },
              [](uint64_t) {},
              [&](auto callback) {
                if (++launches == 3)
                  throw std::runtime_error("launch fixture");
                return std::jthread([&, callback] {
                  callback();
                  ++exited;
                });
              });
        },
        "launch fixture");
    require(operations == 0 && exited == 2,
            "partial launch admitted mutations/leaked workers");
  }
}  // namespace
int main() {
  try {
    for (uint32_t w : {1, 2, 4, 8}) {
      complete(w, 1003);
      complete(w, 3);
      complete(w, 0);
    }
    for (const auto* stage :
         {"reader fixture", "routing fixture", "publication fixture"})
      failure(stage, false);
    failure("progress fixture", true);
    launch_failure();
    for (uint32_t w : {0, 257})
      rejects([&] { ivf_cli::effective_insert_workers(1, w); }, "[1, 256]");
    std::cout << "Insertion counts, bounds, ownership, failure stop/join and "
                 "launch gate passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
