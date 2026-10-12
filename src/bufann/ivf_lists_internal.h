#pragma once

// Internal scheduling seam: production callbacks route through the immutable
// router; tests can inject reader/worker/progress failures without real inputs.
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace diskann::ivf::detail {
  template<class Read, class Assign, class Progress>
  void route_list_blocks(uint32_t rows, uint32_t block_rows, uint32_t workers,
                         Read read, Assign assign, Progress progress) {
    if (!rows || !block_rows || !workers || workers > 256)
      throw std::invalid_argument("Invalid IVF list routing schedule");
    const uint32_t capacity =
        uint32_t(std::min<uint64_t>(rows, uint64_t(workers) * block_rows));
    if (workers == 1) {
      for (uint64_t first = 0; first < rows; first += capacity) {
        const auto count = uint32_t(std::min<uint64_t>(capacity, rows - first));
        read(uint32_t(first), count);
        for (uint32_t r = 0; r < count; ++r)
          assign(uint32_t(first + r), r);
        progress(uint32_t(first + count));
      }
      return;
    }

    std::mutex                mutex;
    std::condition_variable   start, finished;
    std::atomic<uint64_t>     cursor{0};
    uint64_t                  generation = 0;
    uint32_t                  first = 0, count = 0, remaining = 0;
    bool                      stopping = false;
    std::exception_ptr        error;
    std::vector<std::jthread> threads;
    auto                      shutdown = [&] {
      {
        std::lock_guard lock(mutex);
        stopping = true;
      }
      start.notify_all();
      threads
          .clear();  // Join even after a reader/worker/spawn/progress failure.
    };
    auto work = [&] {
      uint64_t seen = 0;
      for (;;) {
        std::unique_lock lock(mutex);
        start.wait(lock, [&] { return stopping || generation != seen; });
        if (stopping)
          return;
        seen = generation;
        const uint32_t wave_first = first, wave_count = count;
        lock.unlock();
        try {
          for (;;) {
            const auto begin = cursor.fetch_add(block_rows);
            if (begin >= wave_count)
              break;
            const auto end = std::min<uint64_t>(wave_count, begin + block_rows);
            for (uint64_t r = begin; r < end; ++r)
              assign(uint32_t(wave_first + r), uint32_t(r));
          }
        } catch (...) {
          std::lock_guard failure_lock(mutex);
          if (!error)
            error = std::current_exception();
          cursor.store(wave_count);  // No new blocks; admitted work joins.
        }
        lock.lock();
        if (--remaining == 0)
          finished.notify_one();
      }
    };
    try {
      threads.reserve(workers);
      for (uint32_t w = 0; w < workers; ++w)
        threads.emplace_back(work);
      for (uint64_t begin = 0; begin < rows; begin += capacity) {
        const auto size = uint32_t(std::min<uint64_t>(capacity, rows - begin));
        // Only the reader touches the stream/buffer between joined waves.
        read(uint32_t(begin), size);
        std::unique_lock lock(mutex);
        first = uint32_t(begin);
        count = size;
        cursor.store(0);
        remaining = workers;
        ++generation;
        start.notify_all();
        finished.wait(lock, [&] { return remaining == 0; });
        if (error)
          std::rethrow_exception(error);
        lock.unlock();
        progress(uint32_t(begin + size));
      }
      shutdown();
    } catch (...) {
      shutdown();
      throw;
    }
  }
}  // namespace diskann::ivf::detail
