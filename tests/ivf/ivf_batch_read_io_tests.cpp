#include "bufann/ivf_heap.h"
#include "linux_aligned_file_reader.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
  std::atomic<int>      completion{0}, collection{0};
  std::atomic<bool>     secondary{false};
  std::atomic<unsigned> destroys{0};
  constexpr int         short_read = 1, zero_read = 2, error_read = 3;
  using diskann::inplace::BufferPool;
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
    throw std::runtime_error("expected terminal batch read failure");
  }
  void pin(BufferPool& pool) {
    const uint32_t               ids[]{0, 1, 0, 2};
    diskann::inplace::PageFrame* frames[4]{};
    pool.pin_batch(ids, frames, 4);
    for (auto* frame : frames)
      pool.unpin_frame(frame);
  }
  std::filesystem::path new_attempt(const std::filesystem::path& parent) {
    std::filesystem::create_directories(parent);
    for (unsigned i = 1; i <= 10000; ++i) {
      const auto path = parent / ("run-" + std::to_string(i));
      if (std::filesystem::create_directory(path))
        return path;
    }
    throw std::runtime_error("batch read attempts exhausted");
  }
}  // namespace

extern "C" int io_getevents(io_context_t ctx, long minimum, long maximum,
                            io_event* events, timespec* timeout) {
  const int fault = collection.exchange(0);
  if (fault)
    return -fault;
  const long result =
      syscall(SYS_io_getevents, ctx, minimum, maximum, events, timeout);
  if (result < 0)
    return -errno;
  if (result > 0) {
    switch (completion.exchange(0)) {
      case short_read:
        events[0].res = 512;
        break;
      case zero_read:
        events[0].res = 0;
        break;
      case error_read:
        events[0].res = -EIO;
        break;
    }
    if (secondary.exchange(false))
      events[0].res2 = -EIO;
  }
  return int(result);
}
extern "C" int io_destroy(io_context_t ctx) {
  ++destroys;
  const long result = syscall(SYS_io_destroy, ctx);
  return result < 0 ? -errno : int(result);
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    const auto root = new_attempt(argv[1]);
    for (int code : {short_read, zero_read, error_read, 4, 5}) {
      BufferPool pool;
      pool.init(4096, 16,
                (root / ("fault-" + std::to_string(code) + ".heap")).string(),
                nullptr);
      pool.grow_heap(
          3);  // Full sparse pages, no synchronous pin warms the pool.
      const auto before = destroys.load();
      if (code == 4)
        secondary = true;
      else if (code == 5)
        collection = EIO;
      else
        completion = code;
      const auto error = rejects([&] { pin(pool); });
      require(error.find("Exact batch read failed") != std::string::npos,
              "missing batch failure diagnostic");
      for (uint32_t id = 0; id < 3; ++id)
        require(pool.pin_count_of(id) == 0,
                "failed batch leaked ownership pins");
      require(rejects([&] { pool.flush_all_dirty(); }) == error,
              "failed batch admitted checkpoint flush");
      require(rejects([&] { pin(pool); }) == error,
              "batch failure was not sticky");
      require(destroys.load() - before == (code == 5 ? 1 : 0),
              "collection failure did not quiesce exactly one context");
    }
    {
      BufferPool pool;
      pool.init(4096, 16, (root / "sparse.heap").string(), nullptr);
      pool.grow_heap(3);
      collection = EINTR;
      pin(pool);
      for (uint32_t id = 0; id < 3; ++id)
        require(pool.pin_count_of(id) == 0, "successful batch leaked pins");
    }
    {
      using namespace diskann::ivf;
      auto heap = HeapStorage<uint8_t>::create({128, 16, 4096, 2},
                                               (root / "heap.heap").string());
      std::array<uint8_t, 128> row{};
      for (uint32_t id = 0; id < 16; ++id)
        heap->append(id, row);
      alignas(64) std::array<uint8_t, 128> output{};
      const uint32_t                       ids[]{0};
      completion = short_read;
      rejects([&] { heap->copy_coordinates_batch(ids, output, 128); });
      require(heap->failed(), "batch failure did not poison IVF heap");
      const auto meta = (root / "failed.meta").string();
      rejects([&] { heap->checkpoint(meta); });
      require(!std::filesystem::exists(meta),
              "failed read published a checkpoint");
    }
    // Real EOF, not simulated completion bytes.
    const auto truncated = (root / "truncated.heap").string();
    {
      BufferPool pool;
      pool.init(4096, 16, truncated, nullptr);
      pool.grow_heap(3);
    }
    std::filesystem::resize_file(truncated, 512);
    BufferPool reopened;
    reopened.init(4096, 16, truncated, nullptr, 4, 1, false);
    require(rejects([&] { pin(reopened); }).find("Exact batch read failed") !=
                std::string::npos,
            "real truncated pages were padded or published READY");
    std::cout << "Exact batch completion/draining/pool failure checks passed\n";
    return 0;
  } catch (const diskann::ANNException& error) {
    std::cerr << error.message() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
  }
  return 1;
}
