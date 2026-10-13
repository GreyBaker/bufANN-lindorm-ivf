#include "bufann/ivf_pin_admission.h"

#include <array>
#include <atomic>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

namespace {
  using diskann::ivf::detail::PinAdmission;
  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  void await_queue(PinAdmission& gate, uint32_t size) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (gate.queued_requests() != size &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    require(gate.queued_requests() == size, "contender did not queue");
  }
  void fifo() {
    PinAdmission                   gate(2);
    uint64_t                       wait = 0;
    std::array<unsigned, 2>        order{};
    std::atomic<unsigned>          arrived{0};
    std::vector<std::future<void>> work;
    {
      auto held = gate.acquire(2, wait);
      for (unsigned i = 0; i < 2; ++i) {
        work.push_back(std::async(std::launch::async, [&, i] {
          uint64_t delay = 0;
          auto     lease = gate.acquire(2, delay);
          order[arrived.fetch_add(1)] = i;
        }));
        await_queue(gate, i + 1);
      }
      require(arrived == 0, "capacity was oversubscribed");
    }
    for (auto& task : work)
      task.get();
    require(order == std::array<unsigned, 2>{0, 1}, "FIFO order changed");
    require(gate.queued_requests() == 0, "completed leases leaked waiters");
  }
  void failure() {
    PinAdmission gate(1);
    uint64_t     wait = 0;
    auto         held = gate.acquire(1, wait);
    auto         worker = std::async(std::launch::async, [&] {
      uint64_t delay = 0;
      try {
        auto lease = gate.acquire(1, delay);
        throw std::logic_error("poisoned acquire succeeded");
      } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "deliberate read failure",
                        "original failure changed");
      }
    });
    await_queue(gate, 1);
    gate.poison(
        std::make_exception_ptr(std::runtime_error("deliberate read failure")));
    worker.get();  // Poison must wake the waiter even while our lease is held.
    require(gate.queued_requests() == 0, "terminal waiter count leaked");
    try {
      auto lease = gate.acquire(1, wait);
      throw std::logic_error("terminal gate reused");
    } catch (const std::runtime_error& error) {
      require(std::string(error.what()) == "deliberate read failure",
              "sticky failure changed");
    }
  }
  void pressure() {
    for (uint32_t capacity : {1, 2, 15}) {
      PinAdmission                      gate(capacity);
      std::atomic<unsigned>             active{0}, completed{0};
      std::array<std::exception_ptr, 8> errors{};
      std::vector<std::jthread>         workers;
      for (unsigned i = 0; i < 8; ++i)
        workers.emplace_back([&, i] {
          try {
            for (unsigned r = 0; r < 1000; ++r) {
              const uint32_t credits = 1 + (r + i) % capacity;
              uint64_t       delay = 0;
              auto           lease = gate.acquire(credits, delay);
              require(active.fetch_add(credits) + credits <= capacity,
                      "aggregate pin bound exceeded");
              std::this_thread::yield();
              active.fetch_sub(credits);
              ++completed;
            }
          } catch (...) {
            errors[i] = std::current_exception();
            gate.poison(errors[i]);
          }
        });
      workers.clear();
      for (auto error : errors)
        if (error)
          std::rethrow_exception(error);
      require(active == 0 && completed == 8000, "pressure leases/counts lost");
    }
  }
}  // namespace
int main() {
  fifo();
  failure();
  pressure();
  std::cout << "Aggregate admission FIFO/failure/pressure checks passed\n";
}
