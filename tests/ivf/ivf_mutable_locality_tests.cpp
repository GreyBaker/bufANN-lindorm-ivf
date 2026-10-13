#include "bufann/ivf_heap.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
  using namespace diskann::ivf;
  constexpr uint32_t rows = 8192, first = 17, dim = 128, tail = 64;
  void               require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<typename F>
  void rejects(F operation) {
    try {
      operation();
    } catch (const std::exception&) {
      return;
    }
    throw std::runtime_error("expected retained-ID reuse refusal");
  }
  template<typename T>
  std::array<T, dim> vector(uint32_t row) {
    std::array<T, dim> result{};
    for (uint32_t d = 0; d < dim; ++d)
      result[d] = T(int((row * 3 + d) % 31) - 15);
    return result;
  }
  std::filesystem::path new_attempt(const std::filesystem::path& parent) {
    std::filesystem::create_directories(parent);
    for (unsigned i = 1; i <= 10000; ++i) {
      const auto path = parent / ("run-" + std::to_string(i));
      if (std::filesystem::create_directory(path))
        return path;
    }
    throw std::runtime_error("locality fixture attempts exhausted");
  }
  template<typename T>
  uint32_t first_page_second_id(const std::filesystem::path& path) {
    // Both controls evict this completed first page before the timed loop ends.
    std::ifstream                   source(path, std::ios::binary);
    std::array<unsigned char, 4096> page{};
    source.read(reinterpret_cast<char*>(page.data()), page.size());
    require(bool(source), "cannot inspect materialized first page");
    const uint32_t slots = uint32_t(page[0]) | uint32_t(page[1]) << 8;
    require(slots >= 2, "invalid completed first-page geometry");
    const size_t offset = 8 + (slots + 7) / 8 + (8 + dim * sizeof(T)) + 4;
    uint32_t     id = 0;
    std::memcpy(&id, page.data() + offset, sizeof(id));
    return id;
  }
  template<typename T>
  void exercise(const std::filesystem::path& root, const char* dtype,
                uint32_t frames) {
    const auto prefix =
        root / (std::string(dtype) + "-" + std::to_string(frames));
    const HeapOptions options{dim, first + rows + tail + 4, 4096, frames};
    auto              legacy =
        HeapStorage<T>::create(options, prefix.string() + "-legacy.heap");
    auto local = HeapStorage<T>::create_mutable(
        options, prefix.string() + "-local.heap");
    std::array<HeapIOCounters, 2> io{};
    std::array<double, 2>         elapsed{};
    for (unsigned mode = 0; mode < 2; ++mode) {
      auto&      heap = mode ? *local : *legacy;
      const auto before = heap.io_counters();
      const auto begin = std::chrono::steady_clock::now();
      for (uint32_t r = 0; r < rows; ++r) {
        const auto values = vector<T>(r);
        if (mode) {
          heap.prepare_insert(first + r, values);
          heap.publish_insert(first + r);
        } else {
          heap.append(first + r, values);
        }
      }
      elapsed[mode] = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
      io[mode] = heap.io_counters().since(before);
      require(heap.size() == rows, "locality changed the indexed population");
      require(heap.retained_file_bytes() > 256ull * 4096,
              "fixture did not cross a growth-batch boundary");
    }
    require(first_page_second_id<T>(prefix.string() + "-legacy.heap") ==
                first + 256,
            "generic allocation stopped being round-robin");
    require(
        first_page_second_id<T>(prefix.string() + "-local.heap") == first + 1,
        "mutable IVF did not fill the current page");
    require(io[1].page_reads <= io[0].page_reads &&
                io[1].dirty_evictions <= io[0].dirty_evictions,
            "page-local placement increased physical pressure in the "
            "fixed-pool control");
    if (frames == 64)
      require(io[1].page_reads * 5 < io[0].page_reads &&
                  io[1].dirty_evictions * 5 < io[0].dirty_evictions,
              "pressure fixture did not reduce read/dirty-eviction churn");
    // Full logical comparison is outside the insertion timer and I/O deltas.
    alignas(64) std::array<T, dim> output{};
    for (uint32_t r = 0; r < rows; ++r) {
      const auto expected = vector<T>(r);
      require(legacy->copy_coordinates(first + r, output) && output == expected,
              "round-robin control coordinates changed");
      require(local->copy_coordinates(first + r, output) && output == expected,
              "page-local coordinates/IDs changed");
    }
    uint32_t retired = 0;
    for (uint32_t r = 0; r < rows; r += 97) {
      require(local->retire(first + r), "retirement failed");
      ++retired;
    }
    require(local->retained_size() == rows && local->size() == rows - retired,
            "logical deletion reclaimed retained slots");
    output.fill(T(42));
    require(!local->copy_coordinates(first, output) && output[0] == T(42),
            "deleted row was copied");
    rejects([&] { local->prepare_insert(first, vector<T>(0)); });
    for (uint32_t r = rows; r < rows + tail; ++r) {
      local->prepare_insert(first + r, vector<T>(r));
      local->publish_insert(first + r);
    }
    const uint32_t canceled = first + rows + tail;
    local->prepare_insert(canceled, vector<T>(0));
    local->cancel_insert(canceled);
    const auto retained = local->retained_size();
    local->prepare_insert(canceled, vector<T>(1));
    require(local->retained_size() == retained && !local->contains(canceled),
            "aborted retry allocated another slot or became visible");
    local->publish_insert(canceled);
    require(local->copy_coordinates(canceled, output) && output == vector<T>(1),
            "aborted retry changed coordinate ownership");
    std::cout << "{\"dtype\":\"" << dtype << "\",\"frames\":" << frames
              << ",\"rows\":" << rows
              << ",\"legacy_reads\":" << io[0].page_reads
              << ",\"local_reads\":" << io[1].page_reads
              << ",\"legacy_dirty_evictions\":" << io[0].dirty_evictions
              << ",\"local_dirty_evictions\":" << io[1].dirty_evictions
              << ",\"legacy_foreground_s\":" << elapsed[0]
              << ",\"local_foreground_s\":" << elapsed[1] << "}\n";
  }
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    const auto root = new_attempt(argv[1]);
    for (uint32_t frames : {64, 256}) {
      exercise<float>(root, "float", frames);
      exercise<int8_t>(root, "int8", frames);
      exercise<uint8_t>(root, "uint8", frames);
    }
    return 0;
  } catch (const diskann::ANNException& error) {
    std::cerr << error.message() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
  }
  return 1;
}
