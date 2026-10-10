#include "../ivf_mixed_workload.h"
#include <barrier>
#include <iostream>
#include <stdexcept>

using namespace ivf_cli;
void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Mixed scheduler fixture failed");
}
int main() {
  try {
    // Startup release/abort before and racing with wait must wake every
    // worker. In production abort is used when later thread creation throws.
    // Exercise that gate directly, without injecting native/system failures.
    for (uint32_t repeat = 0; repeat < 8; ++repeat)
      for (bool abort : {false, true})
        for (bool early : {false, true}) {
          detail::MixedStartGate startup;
          if (early)
            startup.release(abort);
          std::barrier              entered(9);
          std::atomic<uint32_t>     admitted{0}, returned{0};
          std::vector<std::jthread> workers;
          for (uint32_t w = 0; w < 8; ++w)
            workers.emplace_back([&] {
              entered.arrive_and_wait();
              if (startup.wait())
                ++admitted;
              ++returned;
            });
          entered.arrive_and_wait();
          if (!early)
            startup.release(abort);
          for (auto& worker : workers)
            worker.join();
          require(returned == 8 && admitted == (abort ? 0u : 8u));
        }
    for (bool fixed : {false, true}) {
      MixedPlan plan{fixed, 16, 128, 96, 512};
      std::array<std::vector<std::atomic<uint32_t>>, 3> seen{
          std::vector<std::atomic<uint32_t>>(128),
          std::vector<std::atomic<uint32_t>>(96),
          std::vector<std::atomic<uint32_t>>(512)};
      std::array<std::function<void(uint64_t)>, 3> tasks;
      for (size_t op = 0; op < 3; ++op)
        tasks[op] = [&, op](uint64_t i) {
          ++seen[op][size_t(i)];
          std::this_thread::yield();
        };
      auto stats = run_mixed_round(plan, tasks);
      for (const auto& stream : seen)
        for (const auto& count : stream)
          require(count == 1);
      require(stats.completed == std::array<uint64_t, 3>{128, 96, 512} &&
              stats.wall_s > 0 && stats.query_latency_us.size() == 512);
      for (auto latency : stats.query_latency_us)
        require(latency >= 0);
      tasks[0] = [](uint64_t) {
        throw std::runtime_error("expected task failure");
      };
      bool refused = false;
      try {
        (void) run_mixed_round(plan, tasks);
      } catch (const std::runtime_error&) {
        refused = true;
      }
      require(refused);
    }
    // Deterministic dedicated-role overlap: eight mutation and eight query
    // callbacks are simultaneously admitted, without timing/sleep assertions.
    MixedPlan             fixed{true, 16, 8, 8, 8};
    std::barrier          gate(16);
    std::atomic<uint32_t> first_mutations{0};
    auto                  overlap =
        run_mixed_round(fixed, {[&](uint64_t) {
                                  if (first_mutations.fetch_add(1) < 8)
                                    gate.arrive_and_wait();
                                },
                                [&](uint64_t) {
                                  if (first_mutations.fetch_add(1) < 8)
                                    gate.arrive_and_wait();
                                },
                                [&](uint64_t) { gate.arrive_and_wait(); }});
    require(overlap.overlapping_queries == 8 && overlap.peak_mutations == 8 &&
            overlap.peak_queries == 8);
    // Update-driven admission has no query quota. Hold the first callbacks of
    // each dedicated role until all workers enter, proving overlap without
    // sleeps.
    for (uint32_t workers : {2u, 3u, 6u}) {
      MixedPlan plan{false, workers, 12, 12, 0};
      plan.update_driven = true;
      std::barrier          entered(workers);
      std::atomic<uint32_t> first_mutations{0}, first_queries{0};
      std::atomic<uint64_t> queries{0};
      std::array<std::array<std::atomic<uint32_t>, 12>, 2> seen{};
      auto mutation = [&](size_t op, uint64_t i) {
        if (first_mutations.fetch_add(1) < workers / 2)
          entered.arrive_and_wait();
        ++seen[op][size_t(i)];
      };
      auto stats = run_mixed_round(
          plan, {[&](uint64_t i) { mutation(0, i); },
                 [&](uint64_t i) { mutation(1, i); },
                 [&](uint64_t) {
                   if (first_queries.fetch_add(1) < workers - workers / 2)
                     entered.arrive_and_wait();
                   ++queries;
                 }});
      require(stats.completed[0] == 12 && stats.completed[1] == 12 &&
              stats.completed[2] == queries &&
              queries >= workers - workers / 2 &&
              stats.overlapping_queries >= workers - workers / 2 &&
              stats.query_latency_us.size() ==
                  std::min(uint64_t(queries), plan.latency_capacity()));
      for (const auto& stream : seen)
        for (const auto& count : stream)
          require(count == 1);
    }
    // Query volume can exceed retained latency storage; completion means and
    // counts must still cover every callback. A predicate, not timing, ends
    // work.
    {
      MixedPlan plan{false, 2, 1, 1, 0};
      plan.update_driven = true;
      std::mutex              mutex;
      std::condition_variable ready;
      uint64_t                queries = 0;
      const auto              threshold = plan.latency_capacity() + 1024;
      auto                    stats = run_mixed_round(
          plan, {[&](uint64_t) {
                   std::unique_lock lock(mutex);
                   ready.wait(lock, [&] { return queries >= threshold; });
                 },
                                    [](uint64_t) {},
                                    [&](uint64_t) {
                   {
                     std::lock_guard lock(mutex);
                     ++queries;
                   }
                   ready.notify_one();
                 }});
      require(stats.completed[2] == queries && queries >= threshold &&
              stats.query_latency_us.size() == plan.latency_capacity());
      for (auto latency : stats.query_latency_us)
        require(latency >= 0);
    }
    // Failure in either role cancels admissions and joins both worker groups.
    for (bool fail_query : {false, true}) {
      MixedPlan plan{false, 2, 1, 1, 0};
      plan.update_driven = true;
      std::barrier entered(2);
      bool         refused = false;
      try {
        (void) run_mixed_round(
            plan, {[&](uint64_t) {
                     entered.arrive_and_wait();
                     if (!fail_query)
                       throw std::runtime_error("mutation failure");
                   },
                   [](uint64_t) {},
                   [&](uint64_t) {
                     entered.arrive_and_wait();
                     throw std::runtime_error("query failure");
                   }});
      } catch (const std::runtime_error&) {
        refused = true;
      }
      require(refused);
    }
    for (auto plan :
         {MixedPlan{false, 1, 1, 1, 0}, MixedPlan{true, 16, 1, 1, 0},
          MixedPlan{false, 2, 1, 1, 1}}) {
      plan.update_driven = true;
      bool refused = false;
      try {
        plan.validate();
      } catch (const std::invalid_argument&) {
        refused = true;
      }
      require(refused);
    }
    for (auto plan :
         {MixedPlan{false, 0, 1, 1, 1}, MixedPlan{true, 8, 1, 1, 1},
          MixedPlan{false, 4, 0, 1, 1}, MixedPlan{false, 4, UINT64_MAX, 1, 1},
          MixedPlan{false, 4, 1, 1, 0}}) {
      bool refused = false;
      try {
        plan.validate();
      } catch (const std::invalid_argument&) {
        refused = true;
      }
      require(refused);
    }
    require(ann_bench::mixed_update_query_count(8, 8, 80) == 64);
    std::cout << "Mixed update-driven/weighted/fixed allocation, bounded "
                 "latency storage, dense dispatch, overlap and "
                 "joined failure fixtures passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
