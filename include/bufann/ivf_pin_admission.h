#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace diskann::ivf::detail {
  // FIFO aggregate credits for the worst-case shard. Never hold this mutex
  // during backend work. Every pin-owning operation must hold a lease, and
  // destroy its frames before its lease. Poison is terminal and wakes waiters.
  class PinAdmission {
   public:
    class Lease {
     public:
      Lease() = default;
      Lease(const Lease&) = delete;
      Lease& operator=(const Lease&) = delete;
      Lease(Lease&& other) noexcept
          : owner_(std::exchange(other.owner_, nullptr)),
            credits_(other.credits_) {
      }
      Lease& operator=(Lease&&) = delete;
      ~Lease() {
        if (owner_)
          owner_->release(credits_);
      }

     private:
      friend class PinAdmission;
      Lease(PinAdmission* owner, uint32_t credits)
          : owner_(owner), credits_(credits) {
      }
      PinAdmission* owner_ = nullptr;
      uint32_t      credits_ = 0;
    };
    explicit PinAdmission(uint32_t capacity) : capacity_(capacity) {
      if (!capacity)
        throw std::invalid_argument(
            "IVF pin admission requires positive capacity");
    }
    Lease acquire(uint32_t credits, uint64_t& wait_ns) {
      if (credits > capacity_)
        throw std::invalid_argument(
            "IVF pin request exceeds aggregate capacity");
      const auto       begin = std::chrono::steady_clock::now();
      std::unique_lock lock(mutex_);
      if (failure_)
        std::rethrow_exception(failure_);
      if (!credits) {
        wait_ns = 0;
        return {};
      }
      if (next_ == std::numeric_limits<uint64_t>::max() ||
          waiting_ == std::numeric_limits<uint32_t>::max())
        throw std::overflow_error(
            "IVF pin admission ticket/waiter limit exceeded");
      const auto ticket = next_++;
      ++waiting_;
      try {
        changed_.wait(lock, [&] {
          return failure_ ||
                 (ticket == serving_ && used_ <= capacity_ - credits);
        });
      } catch (...) {
        --waiting_;
        if (!failure_)
          failure_ = std::current_exception();
        changed_.notify_all();
        throw;
      }
      --waiting_;
      if (failure_)
        std::rethrow_exception(failure_);
      used_ += credits;
      ++serving_;
      changed_.notify_all();
      wait_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count());
      return Lease(this, credits);
    }
    void poison(std::exception_ptr error) noexcept {
      if (!error)
        std::terminate();
      std::lock_guard lock(mutex_);
      if (!failure_)
        failure_ = error;
      changed_.notify_all();
    }
    uint32_t capacity() const {
      return capacity_;
    }
    uint32_t queued_requests() const {
      std::lock_guard lock(mutex_);
      return waiting_;
    }

   private:
    void release(uint32_t credits) noexcept {
      std::lock_guard lock(mutex_);
      if (credits > used_)
        std::terminate();
      used_ -= credits;
      changed_.notify_all();
    }
    const uint32_t          capacity_;
    uint32_t                used_ = 0, waiting_ = 0;
    uint64_t                next_ = 0, serving_ = 0;
    mutable std::mutex      mutex_;
    std::condition_variable changed_;
    std::exception_ptr      failure_;
  };
}  // namespace diskann::ivf::detail
