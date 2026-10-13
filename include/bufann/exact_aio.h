#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <exception>
#include <libaio.h>
#include <span>
#include <stdexcept>
#include <string>

namespace diskann::inplace::detail {
  // No allocations while kernel requests own these control blocks/buffers.
  // Ops::abort_pending must synchronously quiesce the context. If that fails,
  // fail-stop rather than unwind into buffers that the kernel may still write.
  template<typename Ops>
  void exact_aio_batch(Ops& ops, std::span<iocb> blocks) {
    if (blocks.empty())
      return;
    if (blocks.size() > 256)
      throw std::invalid_argument("Exact AIO batch exceeds 256 requests");
    std::array<iocb*, 256>    pointers{};
    std::array<io_event, 256> events{};
    for (size_t i = 0; i < blocks.size(); ++i)
      pointers[i] = &blocks[i];
    auto fail = [&](const char* operation, size_t index, long actual,
                    long secondary) {
      throw std::runtime_error(
          std::string("Exact batch read failed: operation=") + operation +
          " offset=" + std::to_string(blocks[index].u.c.offset) +
          " requested=" + std::to_string(blocks[index].u.c.nbytes) +
          " actual=" + std::to_string(actual) +
          " errno=" + std::to_string(actual < 0 ? -actual : EIO) +
          " res2=" + std::to_string(secondary));
    };
    auto abort = [&] {
      if (ops.abort_pending() != 0)
        std::terminate();
    };
    size_t first = 0;
    while (first < blocks.size()) {
      int accepted;
      do {
        accepted =
            ops.submit(long(blocks.size() - first), pointers.data() + first);
      } while (accepted == -EINTR);
      if (accepted <= 0)
        fail("submit", first, accepted < 0 ? accepted : -EIO, 0);
      if (size_t(accepted) > blocks.size() - first) {
        abort();
        fail("submit_count", first, -EIO, 0);
      }
      // Drain the accepted prefix before submitting any suffix or throwing a
      // completion error. A partial submission is not permission to resubmit
      // it.
      std::array<bool, 256> seen{};
      size_t                pending = size_t(accepted), bad_index = first;
      long                  bad_result = 0, bad_secondary = 0;
      bool                  failed = false;
      while (pending) {
        int completed;
        do {
          completed = ops.collect(1, long(pending), events.data());
        } while (completed == -EINTR);
        if (completed <= 0 || size_t(completed) > pending) {
          abort();
          fail("collect", first, completed < 0 ? completed : -EIO, 0);
        }
        for (int e = 0; e < completed; ++e) {
          auto found =
              std::find(pointers.begin() + first,
                        pointers.begin() + first + accepted, events[e].obj);
          if (found == pointers.begin() + first + accepted ||
              seen[size_t(found - pointers.begin())]) {
            abort();
            fail("completion_identity", first, -EIO, 0);
          }
          const size_t index = size_t(found - pointers.begin());
          seen[index] = true;
          const long result = long(events[e].res),
                     secondary = long(events[e].res2);
          if (!failed &&
              (result != long(blocks[index].u.c.nbytes) || secondary != 0)) {
            failed = true;
            bad_index = index;
            bad_result = result;
            bad_secondary = secondary;
          }
        }
        pending -= size_t(completed);
      }
      if (failed)
        fail("completion", bad_index, bad_result, bad_secondary);
      first += size_t(accepted);
    }
  }
}  // namespace diskann::inplace::detail
