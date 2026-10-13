#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ivf_cli {
  struct InsertStats {
    uint64_t completed = 0;
    uint32_t workers = 0, peak_active = 0;
  };
  inline uint32_t effective_insert_workers(uint64_t rows, uint32_t requested) {
    if (!requested || requested > 256)
      throw std::invalid_argument("Insertion workers must be in [1, 256]");
    return uint32_t(std::min<uint64_t>(rows, requested));
  }
  // Stable worker index and unique input position. All threads launch before
  // admission; partial launch failure admits no mutation. Admission, first
  // failure and completed-only progress share one mutex. In-flight callbacks
  // finish and every worker joins before rethrow; no partial success stats.
  template<class Insert, class Progress, class Launch>
  InsertStats run_insert_round(uint64_t rows, uint32_t requested, Insert insert,
                               Progress progress, Launch launch) {
    InsertStats result;
    result.workers = effective_insert_workers(rows, requested);
    if (!result.workers)
      return result;
    if (result.workers == 1) {
      for (uint64_t i = 0; i < rows; ++i) {
        insert(0, i);
        progress(++result.completed);
      }
      result.peak_active = 1;
      return result;
    }
    std::mutex                mutex;
    std::condition_variable   ready;
    bool                      released = false, cancelled = false;
    uint64_t                  next = 0;
    uint32_t                  active = 0;
    std::exception_ptr        failure;
    std::vector<std::jthread> threads;
    threads.reserve(result.workers);
    auto worker = [&](uint32_t w) {
      {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] { return released; });
      }
      while (true) {
        uint64_t position;
        {
          std::lock_guard lock(mutex);
          if (cancelled || next == rows)
            return;
          position = next++;
          result.peak_active = std::max(result.peak_active, ++active);
        }
        std::exception_ptr error;
        try {
          insert(w, position);
        } catch (...) {
          error = std::current_exception();
        }
        {
          std::lock_guard lock(mutex);
          --active;
          if (!error) {
            ++result.completed;
            if (!cancelled) {
              try {
                progress(result.completed);
              } catch (...) {
                error = std::current_exception();
              }
            }
          }
          if (error) {
            if (!failure)
              failure = error;
            cancelled = true;
          }
        }
      }
    };
    try {
      for (uint32_t w = 0; w < result.workers; ++w)
        threads.push_back(launch([&, w] { worker(w); }));
      {
        std::lock_guard lock(mutex);
        released = true;
      }
      ready.notify_all();
    } catch (...) {
      {
        std::lock_guard lock(mutex);
        cancelled = true;
        released = true;
      }
      ready.notify_all();
      for (auto& thread : threads)
        thread.join();
      throw;
    }
    for (auto& thread : threads)
      thread.join();
    if (failure)
      std::rethrow_exception(failure);
    if (result.completed != rows || active)
      throw std::runtime_error("Insertion scheduler lost operations");
    return result;
  }
  template<class Insert, class Progress>
  InsertStats run_insert_round(uint64_t rows, uint32_t requested, Insert insert,
                               Progress progress) {
    return run_insert_round(
        rows, requested, insert, progress,
        [](auto callback) { return std::jthread(callback); });
  }
}  // namespace ivf_cli
