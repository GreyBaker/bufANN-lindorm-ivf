#include "bufann/ivf_heap.h"

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
  // Interpose only in this test executable, not in the production library.
  // Always enforce strict alignment, even on permissive mounted filesystems.
  std::atomic<int>      failure{0};
  std::atomic<unsigned> writes{0}, misaligned{0}, interrupted{0};
  std::atomic<bool>     block_write{false}, entered{false};
  constexpr int         short_write = -2, zero_write = -3;

  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<class F>
  std::string rejects(F operation) {
    try {
      operation();
    } catch (const std::exception& error) {
      return error.what();
    }
    throw std::runtime_error("expected terminal write failure");
  }
  void diagnostic(const std::string& error) {
    for (const char* field : {"page write failed", "page_id=", "offset=",
                              "requested=4096", "actual=", "errno="})
      require(error.find(field) != std::string::npos,
              "missing write diagnostic");
  }
  using diskann::inplace::BufferPool;
  using diskann::inplace::FrameRegion;
  void fill(BufferPool& pool, uint32_t page, unsigned char value) {
    auto& frame = pool.pin(page, FrameRegion::UPDATE);
    memset(frame.data, value, 4096);
    pool.unpin_frame(&frame, true, FrameRegion::UPDATE);
  }
  void verify(BufferPool& pool, uint32_t page, unsigned char value) {
    auto& frame = pool.pin(page);
    bool  ok = true;
    for (size_t i = 0; i < 4096; ++i)
      ok &= static_cast<unsigned char>(frame.data[i]) == value;
    pool.unpin_frame(&frame);
    require(ok, "persisted page differs after eviction/reopen");
  }
  void persistence(const std::filesystem::path& root) {
    for (bool background : {false, true}) {
      const auto path =
          (root / (background ? "background.heap" : "explicit.heap")).string();
      {
        BufferPool pool;
        // Leave reusable frames in each shard while a four-page flush is in
        // flight; this fixture tests persistence, not frame-budget admission.
        pool.init(4096, 64, path, nullptr, 4, 1);
        pool.grow_heap(256);
        if (background)
          pool.start_bg_flush(0.0f, 0.0f);
        // Revisiting pages forces writes of newer content, not just new pages.
        for (unsigned pass = 1; pass <= 3; ++pass)
          for (unsigned page = 0; page < 256; ++page)
            fill(pool, page,
                 static_cast<unsigned char>((page + pass) % 251 + 1));
        pool.stop_bg_flush();
        pool.flush_all_dirty();
        require(pool.dirty_page_count() == 0,
                "successful flush left dirty pages");
      }
      BufferPool reopened;
      reopened.init(4096, 16, path, nullptr, 4, 1, false);
      for (unsigned page = 0; page < 256; ++page)
        verify(reopened, page,
               static_cast<unsigned char>((page + 3) % 251 + 1));
    }
    using namespace diskann::ivf;
    const HeapOptions        options{128, 8000, 4096, 64};
    const auto               path = (root / "rows.heap").string();
    const auto               meta = (root / "rows.meta").string();
    auto                     heap = HeapStorage<uint8_t>::create(options, path);
    std::array<uint8_t, 128> row;
    for (uint32_t id = 0; id < 8000; ++id) {
      for (unsigned d = 0; d < row.size(); ++d)
        row[d] = (id + d) % 251;
      heap->append(id, row);
    }
    heap->checkpoint(meta);
    heap.reset();
    heap = HeapStorage<uint8_t>::reopen(options, path, meta);
    alignas(64) std::array<uint8_t, 128> got;
    for (uint32_t id = 0; id < 8000; ++id) {
      const std::array<uint32_t, 1> ids{id};
      require(heap->copy_coordinates_batch(ids, got, 128).present[0],
              "missing row");
      for (unsigned d = 0; d < got.size(); ++d)
        require(got[d] == (id + d) % 251, "wrong persisted coordinates");
    }
  }
  void foreground_failures(const std::filesystem::path& root) {
    for (int code : {EINVAL, ENOSPC, EIO, short_write, zero_write}) {
      BufferPool pool;
      pool.init(4096, 16, (root / ("failure" + std::to_string(code))).string(),
                nullptr);
      pool.grow_heap(1);
      fill(pool, 0, 17);
      failure = code;
      const auto error = rejects([&] { pool.flush_all_dirty(); });
      diagnostic(error);
      failure = 0;
      require(pool.dirty_page_count() == 1,
              "failed write discarded dirty state");
      require(rejects([&] { pool.flush_all_dirty(); }) == error,
              "failure was not sticky");
      require(rejects([&] { pool.pin(0); }) == error,
              "failed pool admitted pins");
      require(rejects([&] { pool.reset(); }) == error,
              "reset revived failed pool");
      require(rejects([&] { pool.grow_heap(1); }) == error,
              "failed pool grew heap");
    }
    {
      BufferPool pool;
      pool.init(4096, 16, (root / "interrupted").string(), nullptr);
      pool.grow_heap(1);
      fill(pool, 0, 19);
      const auto before = interrupted.load();
      failure = EINTR;
      pool.flush_all_dirty();
      require(interrupted == before + 1, "EINTR retry not exercised");
      verify(pool, 0, 19);
    }
    // Eviction and destructor-facing discard must also fail closed.
    for (bool discard : {false, true}) {
      BufferPool pool;
      pool.init(4096, 1, (root / (discard ? "discard" : "eviction")).string(),
                nullptr);
      pool.grow_heap(2);
      fill(pool, 0, 21);
      failure = EIO;
      if (discard) {
        auto& frame = pool.pin(0);
        pool.unpin_frame(&frame);
        pool.discard_frame(&frame);  // Must not throw during guard destruction.
      } else {
        diagnostic(rejects([&] { pool.pin(1); }));
      }
      failure = 0;
      diagnostic(rejects([&] { pool.flush_all_dirty(); }));
      require(pool.dirty_page_count() == 1,
              "eviction/discard lost failed data");
    }
  }
  void background_failure(const std::filesystem::path& root, bool batch) {
    BufferPool pool;
    pool.init(4096, 16, (root / (batch ? "bg_batch" : "bg_single")).string(),
              nullptr, 4, 1);
    pool.grow_heap(2);
    fill(pool, 0, 23);
    entered = false;
    block_write = true;
    failure = EIO;
    pool.start_bg_flush(0.0f, 0.0f);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    if (!entered) {
      block_write = false;
      throw std::runtime_error("background write was not reached");
    }
    auto waiter = std::async(std::launch::async, [&] {
      return rejects([&] {
        if (batch) {
          const uint32_t               ids[]{1, 0};
          diskann::inplace::PageFrame* frames[2];
          pool.pin_batch(ids, frames, 2, FrameRegion::UPDATE);
          for (auto* frame : frames)
            pool.unpin_frame(frame, false, FrameRegion::UPDATE);
        } else {
          auto& frame = pool.pin(0, FrameRegion::UPDATE);
          pool.unpin_frame(&frame, false, FrameRegion::UPDATE);
        }
      });
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    block_write = false;
    pool.stop_bg_flush();
    diagnostic(waiter.get());
    failure = 0;
    diagnostic(rejects([&] { pool.flush_all_dirty(); }));
    require(pool.pin_count_of(0) == 0 && pool.pin_count_of(1) == 0,
            "failed batch leaked admitted pins");
  }
  void checkpoint_failure(const std::filesystem::path& root) {
    using namespace diskann::ivf;
    auto heap = HeapStorage<uint8_t>::create(
        {128, 2, 4096, 16}, (root / "checkpoint.heap").string());
    std::array<uint8_t, 128> row{};
    heap->append(0, row);
    failure = ENOSPC;
    const auto meta = (root / "checkpoint.meta").string();
    diagnostic(rejects([&] { heap->checkpoint(meta); }));
    failure = 0;
    require(!std::filesystem::exists(meta + ".ivf"),
            "failed checkpoint published descriptor");
    rejects([&] { heap->checkpoint((root / "retry.meta").string()); });
  }
}  // namespace

extern "C" ssize_t pwrite(int fd, const void* buffer, size_t count,
                          off_t offset) {
  ++writes;
  if (reinterpret_cast<uintptr_t>(buffer) % 4096 || count % 4096 ||
      offset % 4096) {
    ++misaligned;
    errno = EINVAL;
    return -1;
  }
  if (block_write) {
    entered = true;
    while (block_write)
      std::this_thread::yield();
  }
  const int code = failure.exchange(0);
  if (code == short_write)
    return 512;
  if (code == zero_write)
    return 0;
  if (code > 0) {
    if (code == EINTR)
      ++interrupted;
    errno = code;
    return -1;
  }
  return syscall(SYS_pwrite64, fd, buffer, count, offset);
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    const auto root =
        std::filesystem::path(argv[1]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(root);
    persistence(root);
    foreground_failures(root);
    background_failure(root, false);
    background_failure(root, true);
    checkpoint_failure(root);
    require(writes > 1000 && misaligned == 0,
            "strict-alignment writes not qualified");
    std::cout << "Direct-I/O persistence and injected failures passed; writes="
              << writes << '\n';
    return 0;
  } catch (const diskann::ANNException& error) {
    std::cerr << error.message() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
