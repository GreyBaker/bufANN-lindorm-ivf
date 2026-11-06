#pragma once
#include "weighted_update_pool.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>

namespace ivf_cli {
  namespace detail {
    // Both release and abort change the predicate under the wait mutex. An
    // atomic cancellation flag plus an unlocked notify can lose a wake during
    // launch failure.
    class MixedStartGate {
      std::mutex              mutex_;
      std::condition_variable ready_;
      bool                    released_ = false, aborted_ = false;

     public:
      bool wait() {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return released_; });
        return !aborted_;
      }
      void release(bool aborted = false) {
        {
          std::lock_guard lock(mutex_);
          released_ = true;
          aborted_ = aborted;
        }
        ready_.notify_all();
      }
    };
  }  // namespace detail
  struct MixedPlan {
    bool     fixed_8_8 = false;
    uint32_t pool_threads = 16;
    uint64_t inserts = 0, deletes = 0, queries = 0;
    uint64_t seed = ann_bench::WeightedUpdatePool::kDefaultSeed;
    bool     update_driven = false;
    // Unknown query volume must not turn a long update into unbounded latency
    // storage. Reservoir percentiles are labeled separately from all-query
    // means.
    static constexpr uint64_t latency_sample_limit = 65536;
    uint64_t                  latency_capacity() const {
      return update_driven ? latency_sample_limit : queries;
    }
    void validate() const {
      if (!pool_threads || pool_threads > 256 || (!inserts && !deletes) ||
          (update_driven ? (queries != 0 || fixed_8_8 || pool_threads < 2)
                         : !queries) ||
          inserts > UINT64_MAX - deletes ||
          queries > UINT64_MAX - inserts - deletes ||
          queries > SIZE_MAX / sizeof(double) ||
          (fixed_8_8 && pool_threads != 16))
        throw std::invalid_argument(
            "Invalid mixed plan/counts/worker allocation");
    }
  };
  struct MixedStats {
    std::array<uint64_t, 3> completed{};
    uint64_t                overlapping_queries = 0;
    uint32_t                peak_mutations = 0, peak_queries = 0;
    uint32_t                runtime_threads_before_launch =
        0;  // Filled by the native Linux CLI.
    double                wall_s = 0;
    std::array<double, 3> callback_us{};
    std::vector<double>   query_latency_us;
    uint32_t              rounds = 1;
  };
  struct MixedRoundStats {
    std::array<uint64_t, 3> completed{};
    uint64_t overlapping_queries = 0;
    double wall_s = 0;
  };
  // One worker pool for an ordered schedule. Queries are attributed at
  // completion, so an in-flight query may cross a mutation-round boundary.
  // prepare() runs with no mutations active, but query workers remain live.
  // complete() exports accounting only: no index snapshot or query barrier.
  inline MixedStats run_mixed_rounds(
      const std::vector<MixedPlan>& plans,
      const std::array<std::function<void(size_t, uint64_t)>, 3>& tasks,
      const std::function<void(size_t)>& prepare,
      const std::function<void(size_t, const MixedRoundStats&)>& complete) {
    if (plans.empty() || plans.size() > 4096)
      throw std::invalid_argument("Invalid mixed schedule length");
    const auto& policy = plans.front();
    uint64_t total_queries = 0;
    std::array<uint64_t, 3> expected{};
    for (const auto& plan : plans) {
      plan.validate();
      if (plan.pool_threads != policy.pool_threads ||
          plan.update_driven != policy.update_driven ||
          plan.fixed_8_8 != policy.fixed_8_8 || plan.seed != policy.seed ||
          plan.queries > SIZE_MAX / sizeof(double) - total_queries)
        throw std::invalid_argument("Inconsistent/oversized mixed schedule");
      total_queries += plan.queries;
      const uint64_t counts[]{plan.inserts, plan.deletes, plan.queries};
      for (size_t op = 0; op < 3; ++op) {
        if (counts[op] > UINT64_MAX - expected[op])
          throw std::invalid_argument("Mixed schedule total overflow");
        expected[op] += counts[op];
      }
    }
    for (const auto& task : tasks)
      if (!task)
        throw std::invalid_argument("Missing mixed callback");
    using Clock = std::chrono::steady_clock;
    MixedStats result;
    result.rounds = uint32_t(plans.size());
    if (policy.update_driven)
      result.query_latency_us.reserve(MixedPlan::latency_sample_limit);
    else
      result.query_latency_us.resize(size_t(total_queries));
    std::mt19937_64 latency_rng(policy.seed);
    std::mutex mutex;
    std::condition_variable changed;
    std::exception_ptr failure;
    bool stopped = false, mutations_ready = true;
    size_t round = 0;
    std::array<uint64_t, 3> next{};
    uint64_t query_cursor = 0, mutation_epoch = 0;
    uint32_t active_mutations = 0, active_queries = 0;
    MixedRoundStats current;
    const uint32_t mutation_workers = policy.update_driven
        ? policy.pool_threads / 2 : policy.fixed_8_8 ? 8 : policy.pool_threads;
    prepare(0);
    const auto start = Clock::now();
    auto round_start = start;
    detail::MixedStartGate gate;
    std::vector<std::jthread> workers;
    workers.reserve(policy.pool_threads);
    auto fail = [&](std::exception_ptr error) {
      std::lock_guard lock(mutex);
      if (!failure) failure = error;
      stopped = true;
      changed.notify_all();
    };
    try {
      for (uint32_t w = 0; w < policy.pool_threads; ++w)
        workers.emplace_back([&, w] {
          if (!gate.wait()) return;
          std::mt19937_64 rng(policy.seed + w);
          for (;;) {
            size_t admitted_round = 0, op = 0;
            uint64_t id = 0, epoch = 0;
            bool overlaps = false;
            {
              std::unique_lock lock(mutex);
              auto available = [&] {
                const auto& p = plans[round];
                std::array<uint64_t, 3> remaining{
                    mutations_ready ? p.inserts - next[0] : 0,
                    mutations_ready ? p.deletes - next[1] : 0,
                    p.update_driven ? 1 : p.queries - next[2]};
                if (p.update_driven && round + 1 == plans.size() &&
                    current.completed[0] == p.inserts && current.completed[1] == p.deletes)
                  remaining[2] = 0;  // drain admitted queries; no post-update quota
                if (p.update_driven || p.fixed_8_8) {
                  if (w < mutation_workers) remaining[2] = 0;
                  else remaining[0] = remaining[1] = 0;
                }
                return remaining;
              };
              changed.wait(lock, [&] {
                const auto remaining = available();
                return stopped || remaining[0] || remaining[1] || remaining[2];
              });
              if (stopped) return;
              const auto remaining = available();
              if (!policy.update_driven && !policy.fixed_8_8) {
                auto draw = std::uniform_int_distribution<uint64_t>(
                    0, remaining[0] + remaining[1] + remaining[2] - 1)(rng);
                while (op < 2 && draw >= remaining[op]) draw -= remaining[op++];
              } else if (w >= mutation_workers) op = 2;
              else {
                const auto common = std::min(plans[round].inserts, plans[round].deletes);
                const auto n = next[0] + next[1];
                op = n < 2 * common ? size_t(n % 2)
                     : plans[round].inserts > plans[round].deletes ? 0 : 1;
              }
              admitted_round = round;
              id = op == 2 ? query_cursor++ : next[op];
              ++next[op];
              if (op == 2) {
                epoch = mutation_epoch;
                overlaps = active_mutations != 0;
                result.peak_queries = std::max(result.peak_queries, ++active_queries);
              } else {
                ++mutation_epoch;
                result.peak_mutations = std::max(result.peak_mutations, ++active_mutations);
              }
              changed.notify_all();
            }
            const auto begin = Clock::now();
            std::exception_ptr error;
            try { tasks[op](admitted_round, id); }
            catch (...) { error = std::current_exception(); }
            const auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
            {
              std::lock_guard lock(mutex);
              if (op == 2) {
                --active_queries;
                const bool overlap = overlaps || active_mutations || epoch != mutation_epoch;
                result.overlapping_queries += overlap;
                current.overlapping_queries += overlap;
                if (policy.update_driven) {
                  const auto seen = result.completed[2];
                  if (seen < MixedPlan::latency_sample_limit)
                    result.query_latency_us.push_back(elapsed);
                  else {
                    const auto slot = std::uniform_int_distribution<uint64_t>(0, seen)(latency_rng);
                    if (slot < MixedPlan::latency_sample_limit)
                      result.query_latency_us[size_t(slot)] = elapsed;
                  }
                } else result.query_latency_us[size_t(id)] = elapsed;
              } else { --active_mutations; ++mutation_epoch; }
              if (error) {
                if (!failure) failure = error;
                stopped = true;
              } else { ++result.completed[op]; ++current.completed[op]; }
              result.callback_us[op] += elapsed;
            }
            changed.notify_all();
          }
        });
      gate.release();
      for (;;) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] {
          const auto& p = plans[round];
          return failure || (current.completed[0] == p.inserts &&
              current.completed[1] == p.deletes &&
              (p.update_driven || next[2] == p.queries));
        });
        if (failure) break;
        if (round + 1 == plans.size()) { stopped = true; break; }
        const auto boundary = Clock::now();
        current.wall_s = std::chrono::duration<double>(boundary - round_start).count();
        const auto finished = current;
        const auto previous = round++;
        current = {};
        next = {};
        mutations_ready = false;
        round_start = boundary;
        lock.unlock();
        changed.notify_all();
        complete(previous, finished);
        prepare(round);  // bounded next-round staging; queries continue
        lock.lock();
        mutations_ready = true;
        lock.unlock();
        changed.notify_all();
      }
    } catch (...) {
      fail(std::current_exception());
      gate.release(true);
    }
    changed.notify_all();
    for (auto& worker : workers) worker.join();
    result.wall_s = std::chrono::duration<double>(Clock::now() - start).count();
    if (failure) std::rethrow_exception(failure);
    if (result.completed[0] != expected[0] || result.completed[1] != expected[1] ||
        (!policy.update_driven && result.completed[2] != expected[2]))
      throw std::runtime_error("Mixed scheduler lost operations");
    current.wall_s = std::chrono::duration<double>(Clock::now() - round_start).count();
    complete(round, current);
    return result;
  }
  // Callback exceptions are retained, admission stops, admitted callbacks
  // finish, and all workers join before rethrow. No success stats on partial
  // execution. Overlap is measured admitted operation intervals (including
  // monitor/native lock/I/O waits and retirement bookkeeping), not bare
  // callback timing, CPU simultaneity, device IOPS, collection snapshots or
  // final-GT recall.
  inline MixedStats run_mixed_round(
      const MixedPlan&                                    plan,
      const std::array<std::function<void(uint64_t)>, 3>& tasks) {
    for (const auto& task : tasks)
      if (!task)
        throw std::invalid_argument("Missing mixed callback");
    return run_mixed_rounds(
        {plan}, {[&](size_t, uint64_t i) { tasks[0](i); },
                 [&](size_t, uint64_t i) { tasks[1](i); },
                 [&](size_t, uint64_t i) { tasks[2](i); }},
        [](size_t) {}, [](size_t, const MixedRoundStats&) {});
  }
}  // namespace ivf_cli
