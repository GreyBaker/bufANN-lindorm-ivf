#include "bufann/ivf_fetch_timing.h"

#include <array>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
  void require(bool condition, const char* message) {
    if (!condition)
      throw std::runtime_error(message);
  }
  struct FakeClock {
    using time_point = std::chrono::steady_clock::time_point;
    static inline std::array<time_point, 2> times{};
    static inline size_t                    next = 0;
    static time_point                       now() {
      return times.at(next++);
    }
  };
  struct TestMutex {
    bool owned = false, fail = false;
    void lock() {
      if (fail)
        throw std::runtime_error("deliberate acquisition failure");
      require(!owned, "duplicate lock acquisition");
      owned = true;
    }
    void unlock() {
      require(owned, "unlock without ownership");
      owned = false;
    }
  };
  struct BlockingClock {
    using time_point = std::chrono::steady_clock::time_point;
    static inline std::promise<void> beginning;
    static inline size_t next = 0;  // Only the controlled worker calls now().
    static time_point    now() {
      if (next++ == 0) {
        beginning.set_value();
        return time_point{};
      }
      return time_point{} + std::chrono::nanoseconds(7654321);
    }
  };
}  // namespace

int main() {
  using diskann::ivf::detail::timed_mutex_lock;
  TestMutex mutex;
  uint64_t  wait = 0;
  FakeClock::times[1] += std::chrono::nanoseconds(1234567);
  {
    auto guard = timed_mutex_lock<FakeClock>(mutex, wait);
    require(guard.owns_lock() && mutex.owned, "guard lost ownership");
    require(wait == 1234567, "lock subinterval conversion changed");
  }
  require(!mutex.owned, "lock leaked on scope exit");
  FakeClock::next = 0;
  FakeClock::times = {};
  mutex.fail = true;
  wait = 99;
  try {
    auto guard = timed_mutex_lock<FakeClock>(mutex, wait);
    throw std::logic_error("lock unexpectedly succeeded");
  } catch (const std::runtime_error& error) {
    require(std::string(error.what()) == "deliberate acquisition failure",
            "original acquisition exception changed");
  }
  require(wait == 99 && !mutex.owned,
          "failed acquisition wrote a success metric");
  // A real mutex contender must wait for ownership; diagnostic timing must not
  // replace its synchronization. A promise avoids sleeps and timing assertions.
  std::mutex blocked;
  blocked.lock();
  uint64_t     blocked_wait = 0;
  auto         beginning = BlockingClock::beginning.get_future();
  std::jthread contender([&] {
    auto guard = timed_mutex_lock<BlockingClock>(blocked, blocked_wait);
    require(guard.owns_lock(), "contender did not acquire its mutex");
  });
  beginning.wait();
  blocked.unlock();
  contender.join();
  require(blocked_wait == 7654321, "blocked acquisition was not attributed");
  require(blocked.try_lock(), "contender leaked its lock");
  blocked.unlock();
}
