#include "bufann/ivf_heap.h"
#include <libaio.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
  using namespace diskann::ivf;
  constexpr uint32_t    dim = 128;
  std::atomic<bool>     block_io{false}, fail_completion{false};
  std::atomic<bool>     block_scalar{false}, scalar_entered{false};
  std::atomic<unsigned> entered{0};
  void                  require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<typename F>
  bool await(F predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    return predicate();
  }
  template<typename T>
  std::array<T, dim> vector(uint32_t id) {
    std::array<T, dim> values;
    for (uint32_t d = 0; d < dim; ++d)
      values[d] = T((id + d) % 31);
    return values;
  }
  template<typename T>
  std::unique_ptr<HeapStorage<T>> heap(const std::filesystem::path& path,
                                       uint32_t frames, uint32_t rows,
                                       bool mutable_mode) {
    const HeapOptions options{dim, rows + 512, 4096, frames};
    auto              result = mutable_mode
                                   ? HeapStorage<T>::create_mutable(options, path.string())
                                   : HeapStorage<T>::create(options, path.string());
    for (uint32_t id = 0; id < rows; ++id) {
      if (mutable_mode) {
        result->prepare_insert(id, vector<T>(id));
        result->publish_insert(id);
      } else {
        result->append(id, vector<T>(id));
      }
    }
    if (!mutable_mode)
      result->checkpoint(path.string() + ".meta");
    return result;
  }
  std::filesystem::path new_attempt(const std::filesystem::path& parent) {
    std::filesystem::create_directories(parent);
    for (unsigned i = 1; i <= 10000; ++i) {
      const auto path = parent / ("run-" + std::to_string(i));
      if (std::filesystem::create_directory(path))
        return path;
    }
    throw std::runtime_error("concurrent fetch attempts exhausted");
  }
  void overlap(const std::filesystem::path& root, bool mutable_mode) {
    auto storage =
        heap<float>(root / (mutable_mode ? "mutable.heap" : "immutable.heap"),
                    64, 4096, mutable_mode);
    require(storage->pin_admission_capacity() == 15 &&
                storage->copy_batch_limit() == 7,
            "unexpected worst-shard admission geometry");
    entered = 0;
    block_io = true;
    auto fetch = [&](uint32_t id) {
      return std::async(std::launch::async, [&, id] {
        alignas(64) std::array<float, dim> output;
        output.fill(42);
        const uint32_t ids[]{id};
        const auto result = storage->copy_coordinates_batch(ids, output, dim);
        if (result.present[0])
          require(output == vector<float>(id),
                  "overlapped fetch returned wrong coordinates");
        else
          require(output[0] == 42, "retired fetch modified output");
        return bool(result.present[0]);
      });
    };
    auto              first = fetch(0), second = fetch(30);
    const bool        overlapping = await([] { return entered.load() == 2; });
    bool              mutation_completed = true;
    std::future<void> mutation;
    if (overlapping && mutable_mode) {
      mutation = std::async(std::launch::async, [&] {
        require(storage->retire(0), "concurrent retirement failed");
        storage->prepare_insert(4096, vector<float>(4096));
        storage->publish_insert(4096);
      });
      mutation_completed = mutation.wait_for(std::chrono::seconds(5)) ==
                           std::future_status::ready;
    }
    block_io = false;  // Always unblock before joining/throwing.
    if (mutation.valid())
      mutation.get();
    const bool first_present = first.get(), second_present = second.get();
    require(overlapping,
            "independent batched reads did not overlap in the backend");
    require(mutation_completed,
            "publication/retirement blocked behind query page I/O");
    require(first_present == !mutable_mode && second_present,
            "post-pin live revalidation failed");
    require(!storage->failed() && storage->pin_admission_waiters() == 0,
            "overlap leaked admission or poisoned a healthy heap");
  }
  void failed_admission(const std::filesystem::path& root, bool cached_waiter) {
    const auto label = cached_waiter ? "failed-cached" : "failed-cold";
    auto       storage =
        heap<float>(root / (std::string(label) + ".heap"), 2, 64, false);
    require(storage->pin_admission_capacity() == 1,
            "tiny pool did not serialize pins");
    if (cached_waiter) {
      alignas(64) std::array<float, dim> output{};
      const auto                         before = storage->io_counters();
      require(storage->copy_coordinates(63, output) &&
                  output == vector<float>(63) &&
                  storage->io_counters().since(before).page_reads == 0,
              "cached contender fixture was not resident");
    }
    entered = 0;
    block_io = true;
    fail_completion = true;
    auto operation = [&](uint32_t id) {
      return std::async(std::launch::async, [&, id] {
        alignas(64) std::array<float, dim> output{};
        const uint32_t                     ids[]{id};
        try {
          storage->copy_coordinates_batch(ids, output, dim);
          throw std::logic_error("faulted fetch succeeded");
        } catch (const std::runtime_error& error) {
          return std::string(error.what());
        }
      });
    };
    auto       owner = operation(0);
    const bool blocked = await([] { return entered.load() == 1; });
    auto       waiter = operation(cached_waiter ? 63 : 0);
    const bool queued =
        await([&] { return storage->pin_admission_waiters() == 1; });
    block_io = false;
    const auto original = owner.get(), propagated = waiter.get();
    require(blocked && queued,
            "failure fixture did not retain a queued contender");
    require(original.find("Exact batch read failed") != std::string::npos &&
                propagated == original,
            "queued admission did not receive original read failure");
    require(storage->failed() && storage->pin_admission_waiters() == 0,
            "read failure did not poison/wake admission");
    try {
      storage->checkpoint(
          (root / (std::string(label) + "-refused.meta")).string());
      throw std::logic_error("failed session published checkpoint");
    } catch (const std::runtime_error&) {
    }
    require(
        !std::filesystem::exists(root / (std::string(label) + "-refused.meta")),
        "failed checkpoint was claimed");
  }
  void cancel_retry(const std::filesystem::path& root) {
    auto storage = heap<float>(root / "cancel-retry.heap", 2, 0, true);
    storage->prepare_insert(0, vector<float>(0));
    storage->cancel_insert(0);
    for (uint32_t id = 1; id <= 17; ++id) {
      storage->prepare_insert(id, vector<float>(id));
      storage->publish_insert(id);
    }
    scalar_entered = false;
    block_scalar = true;
    auto       retry = std::async(std::launch::async, [&] {
      storage->prepare_insert(0, vector<float>(1));
    });
    const bool reading = await([] { return scalar_entered.load(); });
    auto       cancel =
        std::async(std::launch::async, [&] { storage->cancel_insert(0); });
    const bool queued = await([&] { return storage->writer_waiters() == 1; });
    block_scalar = false;
    retry.get();
    cancel.get();
    require(reading && queued,
            "retry/cancel fixture did not retain a queued writer");
    alignas(64) std::array<float, dim> output;
    output.fill(42);
    require(!storage->contains(0) && !storage->copy_coordinates(0, output) &&
                output[0] == 42,
            "retry completion undid its concurrent cancellation");
    require(storage->size() == 17 && storage->retained_size() == 18 &&
                storage->writer_waiters() == 0,
            "retry cancellation changed ownership/counts");
    storage->prepare_insert(0, vector<float>(2));
    storage->publish_insert(0);
    require(storage->retained_size() == 18 &&
                storage->copy_coordinates(0, output) &&
                output == vector<float>(2),
            "canceled retry lost its retained slot");
  }
  template<typename T>
  uint32_t slots_per_page() {
    uint32_t slots = 1;
    while (8 + (slots + 1 + 7) / 8 + (slots + 1) * (8 + dim * sizeof(T)) <=
           4096)
      ++slots;
    return slots;
  }
  template<typename T>
  void pressure(const std::filesystem::path& root, const char* dtype,
                uint32_t frames) {
    const std::string label = std::string(dtype) + "-" + std::to_string(frames);
    auto storage = heap<T>(root / (label + ".heap"), frames, 1024, true);
    const uint32_t stride = 4 * slots_per_page<T>();
    // All hit pages hash to shard zero, with duplicates and an absent ID.
    const std::array<uint32_t, 7> ids{
        0, stride, 0, 2 * stride, 3 * stride, 4 * stride, UINT32_MAX};
    std::array<std::exception_ptr, 10> errors{};
    std::vector<std::jthread>          workers;
    for (unsigned t = 0; t < 8; ++t)
      workers.emplace_back([&, t] {
        try {
          for (unsigned r = 0; r < 64; ++r) {
            alignas(64) std::array<T, 7 * dim> output;
            output.fill(T(42));
            const auto result =
                storage->copy_coordinates_batch(ids, output, dim);
            for (size_t i = 0; i < ids.size(); ++i) {
              for (uint32_t d = 0; d < dim; ++d)
                require(output[i * dim + d] ==
                            (result.present[i] ? vector<T>(ids[i])[d] : T(42)),
                        "skewed concurrent fetch lost coordinate "
                        "ownership/absence");
            }
          }
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    workers.emplace_back([&] {
      try {
        for (uint32_t id = 1024; id < 1536; ++id) {
          storage->prepare_insert(id, vector<T>(id));
          storage->publish_insert(id);
        }
      } catch (...) {
        errors[8] = std::current_exception();
      }
    });
    workers.emplace_back([&] {
      try {
        for (uint32_t id = 0; id < 1024; id += 17)
          require(storage->retire(id), "concurrent retire failed");
      } catch (...) {
        errors[9] = std::current_exception();
      }
    });
    workers.clear();
    for (auto error : errors)
      if (error)
        std::rethrow_exception(error);
    require(!storage->failed() && storage->size() == 1536 - (1024 + 16) / 17 &&
                storage->retained_size() == 1536 &&
                storage->pin_admission_waiters() == 0,
            "concurrent mutation counts/admission changed");
    alignas(64) std::array<T, dim> output;
    for (uint32_t id = 1024; id < 1536; ++id)
      require(storage->copy_coordinates(id, output) && output == vector<T>(id),
              "inserted coordinates changed after join");
  }
  template<typename T>
  void scaling(const std::filesystem::path& root, const char* dtype) {
    auto storage =
        heap<T>(root / (std::string(dtype) + "-scaling.heap"), 64, 1024, true);
    const uint32_t                stride = 4 * slots_per_page<T>();
    const std::array<uint32_t, 7> ids{
        0, stride, 0, 2 * stride, 3 * stride, 4 * stride, UINT32_MAX};
    alignas(64) std::array<T, 7 * dim> warm;
    storage->copy_coordinates_batch(ids, warm, dim);  // Outside all timers.
    for (unsigned repetition = 1; repetition <= 3; ++repetition)
      for (unsigned threads : {1, 2, 4, 8})
        for (bool serial : {true, false}) {
          std::atomic<unsigned>             next{0}, completed{0};
          std::mutex                        control;
          std::vector<std::jthread>         workers;
          std::array<std::exception_ptr, 8> errors{};
          const auto begin = std::chrono::steady_clock::now();
          for (unsigned t = 0; t < threads; ++t)
            workers.emplace_back([&, t] {
              try {
                alignas(64) std::array<T, 7 * dim> output;
                while (next.fetch_add(1) < 1024) {
                  std::unique_lock lock(control, std::defer_lock);
                  if (serial)
                    lock.lock();
                  const auto result =
                      storage->copy_coordinates_batch(ids, output, dim);
                  require(result.present[0] && !result.present.back(),
                          "measurement fetch failed");
                  ++completed;
                }
              } catch (...) {
                errors[t] = std::current_exception();
              }
            });
          workers.clear();
          const double seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - begin)
                                     .count();
          for (auto error : errors)
            if (error)
              std::rethrow_exception(error);
          require(completed == 1024,
                  "measurement counted submitted, not completed requests");
          std::cout << "{\"dtype\":\"" << dtype << "\",\"threads\":" << threads
                    << ",\"serial_control\":" << (serial ? "true" : "false")
                    << ",\"repetition\":" << repetition
                    << ",\"completed_fetches\":" << completed
                    << ",\"seconds\":" << seconds
                    << ",\"fetches_per_s\":" << completed / seconds << "}\n";
        }
  }
}  // namespace

extern "C" int io_getevents(io_context_t context, long minimum, long maximum,
                            io_event* events, timespec* timeout) {
  const long result =
      syscall(SYS_io_getevents, context, minimum, maximum, events, timeout);
  if (result < 0)
    return -errno;
  if (result > 0 && block_io) {
    ++entered;
    while (block_io)
      std::this_thread::yield();
    if (fail_completion.exchange(false))
      events[0].res = 512;
  }
  return int(result);
}
extern "C" ssize_t pread(int fd, void* buffer, size_t count, off_t offset) {
  const long result = syscall(SYS_pread64, fd, buffer, count, offset);
  if (result >= 0 && block_scalar) {
    scalar_entered = true;
    while (block_scalar)
      std::this_thread::yield();
  }
  return result;
}
int main(int argc, char** argv) {
  try {
    require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--measure"),
            "expected fixture directory");
    const auto root = new_attempt(argv[1]);
    if (argc == 3) {
      scaling<float>(root, "float");
      scaling<uint8_t>(root, "uint8");
    } else {
      overlap(root, false);
      overlap(root, true);
      failed_admission(root, false);
      failed_admission(root, true);
      cancel_retry(root);
      for (uint32_t frames : {2, 8, 64}) {
        pressure<float>(root, "float", frames);
        pressure<int8_t>(root, "int8", frames);
        pressure<uint8_t>(root, "uint8", frames);
      }
      std::cout << "Batched overlap, live visibility, failure and skewed-pool "
                   "checks passed\n";
    }
    return 0;
  } catch (const diskann::ANNException& error) {
    block_io = false;
    block_scalar = false;
    std::cerr << error.message() << '\n';
  } catch (const std::exception& error) {
    block_io = false;
    block_scalar = false;
    std::cerr << error.what() << '\n';
  }
  return 1;
}
