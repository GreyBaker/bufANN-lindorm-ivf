#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

namespace diskann::ivf::detail {
  // Diagnostic subintervals of fetch wall time, never additional query stages.
  // The clock is injectable so tests need no timing thresholds or sleeps.
  template<typename Clock = std::chrono::steady_clock>
  uint64_t elapsed_nanoseconds(typename Clock::time_point begin) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             Clock::now() - begin)
                             .count();
    return elapsed > 0 ? uint64_t(elapsed) : 0;
  }

  template<typename Clock = std::chrono::steady_clock, typename Mutex>
  std::unique_lock<Mutex> timed_mutex_lock(Mutex& mutex, uint64_t& wait_ns) {
    const auto              begin = Clock::now();
    std::unique_lock<Mutex> lock(mutex);
    wait_ns = elapsed_nanoseconds<Clock>(begin);
    return lock;  // Same lock scope/order as the caller's original guard.
  }
}  // namespace diskann::ivf::detail
