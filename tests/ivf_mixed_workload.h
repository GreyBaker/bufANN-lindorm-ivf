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
      if (!pool_threads || pool_threads > 256 || !inserts || !deletes ||
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
  };
  // Callback exceptions are retained, admission stops, admitted callbacks
  // finish, and all workers join before rethrow. No success stats on partial
  // execution. Overlap is measured admitted operation intervals (including
  // monitor/native lock/I/O waits and retirement bookkeeping), not bare
  // callback timing, CPU simultaneity, device IOPS, collection snapshots or
  // final-GT recall.
  inline MixedStats run_mixed_round(
      const MixedPlan&                                    plan,
      const std::array<std::function<void(uint64_t)>, 3>& tasks) {
    plan.validate();
    for (const auto& task : tasks)
      if (!task)
        throw std::invalid_argument("Missing mixed callback");
    using Clock = std::chrono::steady_clock;
    MixedStats result;
    if (plan.update_driven)
      result.query_latency_us.reserve(size_t(plan.latency_capacity()));
    else
      result.query_latency_us.resize(size_t(plan.queries));
    std::mt19937_64    latency_rng(plan.seed);
    std::mutex         mutex;
    std::atomic<bool>  cancelled{false};
    std::exception_ptr failure;
    uint32_t           active_mutations = 0, active_queries = 0;
    uint64_t           mutation_epoch = 0;
    auto               invoke = [&](size_t op, uint64_t id) noexcept {
      uint64_t epoch = 0;
      bool     overlaps = false;
      {
        std::lock_guard lock(mutex);
        if (cancelled.load() || (op == 2 && plan.update_driven &&
                                 result.completed[0] == plan.inserts &&
                                 result.completed[1] == plan.deletes))
          return false;
        if (op == 2) {
          epoch = mutation_epoch;
          overlaps = active_mutations != 0;
          result.peak_queries = std::max(result.peak_queries, ++active_queries);
        } else {
          ++mutation_epoch;
          result.peak_mutations =
              std::max(result.peak_mutations, ++active_mutations);
        }
      }
      const auto         start = Clock::now();
      std::exception_ptr error;
      try {
        tasks[op](id);
      } catch (...) {
        error = std::current_exception();
      }
      const auto elapsed =
          std::chrono::duration<double, std::micro>(Clock::now() - start)
              .count();
      {
        std::lock_guard lock(mutex);
        if (op == 2) {
          --active_queries;
          result.overlapping_queries +=
              overlaps || active_mutations || epoch != mutation_epoch;
          if (plan.update_driven) {
            const auto seen = result.completed[2];
            if (seen < plan.latency_capacity())
              result.query_latency_us.push_back(elapsed);
            else {
              const auto slot =
                  std::uniform_int_distribution<uint64_t>(0, seen)(latency_rng);
              if (slot < plan.latency_capacity())
                result.query_latency_us[size_t(slot)] = elapsed;
            }
          } else
            result.query_latency_us[size_t(id)] = elapsed;
        } else {
          --active_mutations;
          ++mutation_epoch;
        }
        if (error) {
          if (!failure)
            failure = error;
          cancelled = true;
        } else
          ++result.completed[op];
        result.callback_us[op] += elapsed;
      }
      return !error;
    };
    const auto start = Clock::now();
    if (!plan.fixed_8_8 && !plan.update_driven) {
      const uint64_t counts[]{plan.inserts, plan.deletes, plan.queries};
      const std::function<void(uint64_t)> wrapped[]{
          [&](uint64_t i) { invoke(0, i); }, [&](uint64_t i) { invoke(1, i); },
          [&](uint64_t i) { invoke(2, i); }};
      ann_bench::WeightedUpdatePool::run(plan.pool_threads, counts, wrapped,
                                         plan.seed);
    } else {
      // All workers are created before admission. No extra scheduler thread;
      // the calling thread orchestrates/joins. Dedicated mutation claims
      // alternate insert/delete while both streams remain, not serial mutation
      // phases.
      std::atomic<uint64_t>     next_mutation{0}, next_query{0};
      detail::MixedStartGate    gate;
      std::vector<std::jthread> workers;
      const auto                mutation_threads =
          plan.update_driven ? plan.pool_threads / 2 : 8u;
      workers.reserve(plan.pool_threads);
      try {
        for (uint32_t w = 0; w < plan.pool_threads; ++w)
          workers.emplace_back([&, w] {
            if (!gate.wait())
              return;
            if (w < mutation_threads) {
              const auto common = std::min(plan.inserts, plan.deletes);
              while (!cancelled.load()) {
                const auto n = next_mutation.fetch_add(1);
                if (n >= plan.inserts + plan.deletes)
                  break;
                if (n < 2 * common)
                  invoke(size_t(n % 2), n / 2);
                else
                  invoke(plan.inserts > plan.deletes ? 0 : 1,
                         common + n - 2 * common);
              }
            } else
              while (!cancelled.load()) {
                const auto n = next_query.fetch_add(1);
                if (!plan.update_driven && n >= plan.queries)
                  break;
                // Admission checks completion under the same mutex used to
                // retire mutations. Already admitted queries finish and join.
                if (!invoke(2, n))
                  break;
              }
          });
        gate.release();
      } catch (...) {
        cancelled = true;
        gate.release(true);
        throw;
      }
      // Explicit join before inspecting stats or destroying captured counters.
      for (auto& worker : workers)
        worker.join();
    }
    result.wall_s = std::chrono::duration<double>(Clock::now() - start).count();
    if (failure)
      std::rethrow_exception(failure);
    if (result.completed[0] != plan.inserts ||
        result.completed[1] != plan.deletes ||
        (!plan.update_driven && result.completed[2] != plan.queries))
      throw std::runtime_error("Mixed scheduler lost operations");
    return result;
  }
}  // namespace ivf_cli
