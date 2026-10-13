#include "bufann/ivf_heap.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
  // Test executable only: production uses the normal aligned pread syscall.
  std::atomic<int>      failure{0};
  std::atomic<unsigned> calls{0}, interrupted{0}, misaligned{0};
  std::atomic<bool>     block_read{false}, entered{false};
  constexpr int         short_read = -2, zero_read = -3;
  using diskann::inplace::BufferPool;
  using diskann::inplace::FrameRegion;

  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<typename F>
  std::string rejects(F operation) {
    try {
      operation();
    } catch (const std::exception& error) {
      return error.what();
    }
    throw std::runtime_error("expected terminal read failure");
  }
  void diagnostic(const std::string& error) {
    for (const char* field : {"page read failed", "page_id=", "offset=",
                              "requested=4096", "actual=", "errno="})
      require(error.find(field) != std::string::npos,
              "missing read diagnostic");
  }
  void verify(BufferPool& pool, uint32_t id, unsigned char value) {
    auto& frame = pool.pin(id);
    bool  ok = true;
    for (size_t i = 0; i < 4096; ++i)
      ok &= static_cast<unsigned char>(frame.data[i]) == value;
    pool.unpin_frame(&frame);
    require(ok, "full-page read contents changed");
  }
  void fill(BufferPool& pool, uint32_t id, unsigned char value) {
    auto& frame = pool.pin(id, FrameRegion::UPDATE);
    memset(frame.data, value, 4096);
    pool.unpin_frame(&frame, true, FrameRegion::UPDATE);
  }
  void sparse_and_interrupted(const std::filesystem::path& root) {
    const auto path = (root / "sparse.heap").string();
    {
      BufferPool pool;
      pool.init(4096, 16, path, nullptr);
      pool.grow_heap(3);
      for (unsigned id = 0; id < 3; ++id)
        verify(pool, id,
               0);  // New ftruncate-grown sparse pages are legitimate.
      fill(pool, 1, 37);
      pool.flush_all_dirty();
    }
    BufferPool reopened;
    reopened.init(4096, 16, path, nullptr, 4, 1, false);
    const auto before = interrupted.load();
    failure = EINTR;
    verify(reopened, 1, 37);
    require(interrupted == before + 1, "interrupted pread was not retried");
    verify(reopened, 2, 0);
  }
  void faults(const std::filesystem::path& root) {
    for (auto hint : {FrameRegion::QUERY, FrameRegion::UPDATE}) {
      for (int code : {EINVAL, EIO, short_read, zero_read}) {
        BufferPool pool;
        const auto name =
            std::to_string(int(hint)) + "-" + std::to_string(code);
        pool.init(4096, 16, (root / (name + ".heap")).string(), nullptr);
        pool.grow_heap(2);
        const auto before = diskann::inplace::thread_physical_reads_count();
        failure = code;
        const auto error = rejects([&] { pool.pin(1, hint); });
        failure = 0;
        diagnostic(error);
        require(pool.pin_count_of(1) == 0, "failed owner leaked its pin");
        require(diskann::inplace::thread_physical_reads_count() == before,
                "failed read counted as a completed page read");
        require(rejects([&] { pool.pin(1); }) == error,
                "failed page became readable");
        require(rejects([&] { pool.try_pin_resident(1); }) == error,
                "failed page became resident-readable");
        require(rejects([&] { pool.flush_all_dirty(); }) == error,
                "read failure allowed checkpoint flushing");
        require(rejects([&] { pool.reset(); }) == error,
                "reset revived a failed pool");
        require(rejects([&] { pool.grow_heap(1); }) == error,
                "failed pool grew");
        const uint32_t               ids[]{0, 1};
        diskann::inplace::PageFrame* frames[2]{};
        require(rejects([&] { pool.pin_batch(ids, frames, 2); }) == error,
                "read failure admitted a batch");
      }
    }
  }
  void truncation(const std::filesystem::path& root) {
    for (uint64_t bytes : {0, 512}) {
      const auto path =
          (root / ("truncated-" + std::to_string(bytes) + ".heap")).string();
      {
        BufferPool pool;
        pool.init(4096, 16, path, nullptr);
        pool.grow_heap(1);
        fill(pool, 0, 39);
        pool.flush_all_dirty();
      }
      std::filesystem::resize_file(path, bytes);
      BufferPool reopened;
      reopened.init(4096, 16, path, nullptr, 4, 1, false);
      diagnostic(rejects([&] { reopened.pin(0); }));
      require(reopened.pin_count_of(0) == 0, "truncated page leaked a pin");
    }
  }
  void waiter_failure(const std::filesystem::path& root) {
    BufferPool pool;
    pool.init(4096, 16, (root / "waiter.heap").string(), nullptr);
    pool.grow_heap(2);
    entered = false;
    block_read = true;
    failure = EIO;
    auto       owner = std::async(std::launch::async, [&] {
      return rejects([&] { pool.pin(0, FrameRegion::UPDATE); });
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    if (!entered) {
      block_read = false;
      throw std::runtime_error("owner never reached blocked pread");
    }
    auto waiter = std::async(std::launch::async, [&] {
      return rejects([&] {
        const uint32_t               ids[]{0};
        diskann::inplace::PageFrame* frames[1]{};
        pool.pin_batch(ids, frames, 1, FrameRegion::UPDATE);
        pool.unpin_frame(frames[0], false, FrameRegion::UPDATE);
      });
    });
    // Capture one observation, not two. pin_batch first takes an optimistic
    // pin under the page-table callback, drops it for LOADING, then takes its
    // retained waiter pin. A second sample can land in that gap. Observing two
    // pins proves the contender reached this blocked loader, not that it has
    // necessarily parked on the condition variable yet.
    bool observed_contender = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (pool.pin_count_of(0) == 2) {
        observed_contender = true;
        break;
      }
      std::this_thread::yield();
    }
    block_read = false;
    const auto error = owner.get();
    diagnostic(error);
    require(waiter.get() == error, "waiter did not receive original failure");
    require(observed_contender,
            "fixture did not observe the contender on the LOADING page");
    require(pool.pin_count_of(0) == 0, "read failure leaked owner/waiter pins");
    failure = 0;
  }
  void checkpoint_failure(const std::filesystem::path& root) {
    using namespace diskann::ivf;
    auto heap = HeapStorage<uint8_t>::create(
        {128, 16, 4096, 2}, (root / "checkpoint.heap").string());
    std::array<uint8_t, 128> row{};
    for (uint32_t id = 0; id < 16; ++id) {
      row.fill(uint8_t(id + 1));
      heap->append(id, row);  // Round-robin placement evicts the first page.
    }
    alignas(64) std::array<uint8_t, 128> output;
    output.fill(77);
    failure = EIO;
    diagnostic(rejects([&] { heap->copy_coordinates(0, output); }));
    failure = 0;
    require(heap->failed(), "read failure did not poison IVF heap state");
    require(std::all_of(output.begin(), output.end(),
                        [](uint8_t v) { return v == 77; }),
            "failed read manufactured coordinate output");
    const auto meta = (root / "failed.meta").string();
    rejects([&] { heap->checkpoint(meta); });
    require(!std::filesystem::exists(meta) &&
                !std::filesystem::exists(meta + ".ivf"),
            "failed read published checkpoint metadata");
  }
  std::filesystem::path new_attempt(const std::filesystem::path& parent) {
    std::filesystem::create_directories(parent);
    for (unsigned i = 1; i <= 10000; ++i) {
      const auto path = parent / ("run-" + std::to_string(i));
      if (std::filesystem::create_directory(path))
        return path;
    }
    throw std::runtime_error("read fixture attempts exhausted");
  }
}  // namespace

extern "C" ssize_t pread(int fd, void* buffer, size_t count, off_t offset) {
  ++calls;
  if (reinterpret_cast<uintptr_t>(buffer) % 4096 || count % 4096 ||
      offset % 4096) {
    ++misaligned;
    errno = EINVAL;
    return -1;
  }
  if (block_read) {
    entered = true;
    while (block_read)
      std::this_thread::yield();
  }
  const int code = failure.exchange(0);
  if (code == short_read)
    return 512;
  if (code == zero_read)
    return 0;
  if (code > 0) {
    if (code == EINTR)
      ++interrupted;
    errno = code;
    return -1;
  }
  return syscall(SYS_pread64, fd, buffer, count, offset);
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    const auto root = new_attempt(argv[1]);
    sparse_and_interrupted(root);
    faults(root);
    truncation(root);
    waiter_failure(root);
    checkpoint_failure(root);
    require(calls > 20 && misaligned == 0,
            "aligned read paths were not exercised");
    std::cout
        << "Synchronous read persistence/failure checks passed; pread_calls="
        << calls << " interrupted=" << interrupted << '\n';
    return 0;
  } catch (const diskann::ANNException& error) {
    std::cerr << error.message() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
  }
  return 1;
}
