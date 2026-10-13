#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>

namespace ivf_cli {
  // Single-owner phase reporting, not a heartbeat thread or durable checkpoint.
  // Call observe only after the described operation completes successfully.
  // Clock is injectable for deterministic rate-limit tests. Reports never turn
  // a diagnostic stream failure into a workload failure or mask an exception.
  template<typename Clock = std::chrono::steady_clock>
  class DriverProgress {
   public:
    using Details = std::function<void(std::ostream&)>;
    DriverProgress(std::string_view        phase,
                   std::optional<uint64_t> total = std::nullopt,
                   std::string_view unit = "operations", Details details = {},
                   std::ostream& output = std::cerr)
        : phase_(phase), unit_(unit), total_(total),
          details_(std::move(details)), output_(output), start_(Clock::now()),
          last_report_(start_), exceptions_(std::uncaught_exceptions()) {
      report("begin", start_);
    }
    DriverProgress(const DriverProgress&) = delete;
    DriverProgress& operator=(const DriverProgress&) = delete;
    ~DriverProgress() {
      if (!finished_)
        report(
            std::uncaught_exceptions() > exceptions_ ? "failed" : "abandoned",
            Clock::now());
    }
    void observe(uint64_t completed) noexcept {
      completed_ = completed;
      // No per-vector clock read or output lock. Check time at bounded count
      // intervals; very slow work may not report until the next such boundary.
      if (completed_ - last_check_ < 1024 || (total_ && completed_ == *total_))
        return;
      last_check_ = completed_;
      const auto now = Clock::now();
      if (now - last_report_ < std::chrono::seconds(1))
        return;
      last_report_ = now;
      report("running", now);
    }
    double finish() noexcept {
      if (finished_)
        return finished_elapsed_;
      const auto now = Clock::now();
      report(total_ && completed_ != *total_ ? "incomplete" : "complete", now);
      finished_ = true;
      finished_elapsed_ = elapsed(now);
      return finished_elapsed_;
    }

   private:
    double elapsed(typename Clock::time_point now) const noexcept {
      return std::chrono::duration<double>(now - start_).count();
    }
    void report(std::string_view           event,
                typename Clock::time_point now) noexcept {
      try {
        std::ostringstream line;
        const auto         seconds = elapsed(now);
        line << std::fixed << std::setprecision(3)
             << "[IVF driver] phase=" << phase_ << " event=" << event
             << " elapsed_s=" << seconds;
        if (total_) {
          line << " completed=" << completed_ << " total=" << *total_
               << " unit=" << unit_;
          if (seconds > 0)
            line << " rate_per_s=" << double(completed_) / seconds;
        }
        if (details_)
          details_(line);
        line << '\n';
        // Phases are single-owner, but future independent phases can share the
        // diagnostic destination. Do not interleave lines or mutate its format.
        static std::mutex output_mutex;
        std::lock_guard   lock(output_mutex);
        output_ << line.str() << std::flush;
      } catch (...) {
        // Owned canonical metrics and the workload's original failure remain
        // authoritative; this diagnostic is deliberately best-effort.
      }
    }
    std::string_view           phase_, unit_;
    std::optional<uint64_t>    total_;
    Details                    details_;
    std::ostream&              output_;
    typename Clock::time_point start_, last_report_;
    uint64_t                   completed_ = 0, last_check_ = 0;
    int                        exceptions_;
    bool                       finished_ = false;
    double                     finished_elapsed_ = 0;
  };
}  // namespace ivf_cli
