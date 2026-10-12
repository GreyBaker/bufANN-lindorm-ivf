#include "../../src/bufann/ivf_lists_internal.h"

#include <array>
#include <atomic>
#include <barrier>
#include <iostream>
#include <string>
#include <unordered_set>

namespace {
  void require(bool ok, const char* why) {
    if (!ok)
      throw std::runtime_error(why);
  }
  template<class F>
  void rejects(F operation, const char* expected) {
    try {
      operation();
    } catch (const std::exception& error) {
      require(std::string(error.what()).find(expected) != std::string::npos,
              "unexpected scheduler refusal");
      return;
    }
    throw std::runtime_error("scheduler did not propagate failure");
  }
  void complete(uint32_t workers) {
    constexpr uint32_t                  rows = 30017, block = 17;
    const uint32_t                      capacity = workers * block;
    std::vector<uint32_t>               buffer(capacity), labels(rows);
    std::vector<std::atomic<uint32_t>>  visits(rows);
    std::atomic<uint32_t>               active{0}, maximum{0};
    uint32_t                            read_end = 0, progress_end = 0;
    std::barrier                        first_wave(workers);
    std::mutex                          seen_mutex;
    std::unordered_set<std::thread::id> seen;
    diskann::ivf::detail::route_list_blocks(
        rows, block, workers,
        [&](uint32_t first, uint32_t count) {
          require(active == 0 && first == read_end && count <= capacity,
                  "reader overwrote an active wave or skipped source rows");
          for (uint32_t r = 0; r < count; ++r)
            buffer[r] = first + r;
          read_end = first + count;
        },
        [&](uint32_t row, uint32_t local) {
          const uint32_t concurrent = active.fetch_add(1) + 1;
          auto           prior = maximum.load();
          while (prior < concurrent &&
                 !maximum.compare_exchange_weak(prior, concurrent)) {
          }
          bool first_call;
          {
            std::lock_guard lock(seen_mutex);
            first_call = seen.insert(std::this_thread::get_id()).second;
          }
          if (first_call)
            first_wave.arrive_and_wait();  // Prove actual overlap, not just a
                                           // setting.
          require(local < capacity && buffer[local] == row &&
                      visits[row].fetch_add(1) == 0,
                  "worker duplicated or misrouted an input row");
          labels[row] = row % 257;
          active.fetch_sub(1);
        },
        [&](uint32_t done) {
          require(active == 0 && done == read_end && done > progress_end,
                  "progress ran before workers joined");
          progress_end = done;
        });
    require(read_end == rows && progress_end == rows && maximum == workers,
            "schedule did not use the declared routing workers");
    for (uint32_t r = 0; r < rows; ++r)
      require(visits[r] == 1 && labels[r] == r % 257,
              "schedule lost/reordered labels");
  }
  void failures() {
    for (const char* point :
         {"reader-start", "reader-next", "worker", "progress"}) {
      uint32_t              reads = 0, progress = 0;
      std::atomic<uint32_t> assigned{0};
      rejects(
          [&] {
            diskann::ivf::detail::route_list_blocks(
                100, 7, 4,
                [&](uint32_t first, uint32_t) {
                  ++reads;
                  if ((std::string(point) == "reader-start" && first == 0) ||
                      (std::string(point) == "reader-next" && first > 0))
                    throw std::runtime_error(point);
                },
                [&](uint32_t row, uint32_t) {
                  if (std::string(point) == "worker" && row == 5)
                    throw std::runtime_error(point);
                  ++assigned;
                },
                [&](uint32_t) {
                  ++progress;
                  if (std::string(point) == "progress")
                    throw std::runtime_error(point);
                });
          },
          point);
      if (std::string(point) == "reader-start")
        require(reads == 1 && assigned == 0 && progress == 0,
                "startup reader failure admitted work");
      if (std::string(point) == "reader-next")
        require(reads == 2 && assigned == 28 && progress == 1,
                "reader failure failed to join preceding wave");
      if (std::string(point) == "worker")
        require(reads == 1 && assigned < 28 && progress == 0,
                "worker failure published completed progress");
      if (std::string(point) == "progress")
        require(reads == 1 && assigned == 28 && progress == 1,
                "progress failure admitted another wave");
    }
  }
}  // namespace

int main() {
  try {
    for (uint32_t workers : {1u, 2u, 4u, 20u})
      complete(workers);
    failures();
    std::cout << "Bounded list routing, overlap, exact-once labels, tails and "
                 "joined failures passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
