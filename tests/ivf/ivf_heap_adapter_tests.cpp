#include "bufann/ivf_heap.h"

#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<typename F>
  void rejects(F&& operation) {
    bool rejected = false;
    try {
      operation();
    } catch (const std::exception&) {
      rejected = true;
    } catch (const diskann::ANNException&) {
      rejected = true;
    }
    require(rejected, "expected rejection");
  }
  template<typename T>
  T coordinate(uint32_t row, uint32_t d) {
    return static_cast<T>(int((row + d) % 31) - 15);
  }
  template<typename T>
  void write_vectors(const std::filesystem::path& path,
                     bool                         nonfinite = false) {
    std::ofstream out(path, std::ios::binary);
    uint32_t      header[]{160, 13};
    out.write(reinterpret_cast<char*>(header), sizeof(header));
    for (uint32_t row = 0; row < header[0]; ++row) {
      std::array<T, 13> coords;
      for (uint32_t d = 0; d < 13; ++d)
        coords[d] = coordinate<T>(row, d);
      if constexpr (std::is_same_v<T, float>)
        if (nonfinite && row == 3)
          coords[2] = std::numeric_limits<float>::infinity();
      out.write(reinterpret_cast<char*>(coords.data()), sizeof(coords));
    }
    require(bool(out), "fixture write failed");
  }

  template<typename T>
  void exercise(const std::filesystem::path& root) {
    using diskann::ivf::HeapOptions;
    using diskann::ivf::HeapStorage;
    std::filesystem::create_directory(root);
    const auto        input = root / "vectors.bin";
    const auto        heap_path = (root / "coordinates.heap").string();
    const auto        snapshot = (root / "coordinates.meta").string();
    const HeapOptions options{13, 200, 4096, 2};
    write_vectors<T>(input);
    auto prefix =
        HeapStorage<T>::create(options, (root / "prefix.heap").string());
    prefix->load_vectors(input.string(), 10, 7);
    require(
        prefix->size() == 7 && prefix->contains(16) && !prefix->contains(17),
        "base prefix included insertion tail");
    auto invalid_prefix =
        HeapStorage<T>::create(options, (root / "bad_prefix.heap").string());
    rejects([&] { invalid_prefix->load_vectors(input.string(), 0, 161); });
    require(invalid_prefix->size() == 0 && invalid_prefix->failed(),
            "out-of-range prefix published");
    auto heap = HeapStorage<T>::create(options, heap_path);
    require(heap->size() == 0 && heap->aligned_dim() == 16, "invalid new heap");
    heap->load_vectors(input.string(), 5);
    require(heap->size() == 160 && !heap->contains(4) && heap->contains(164) &&
                !heap->contains(165) && !heap->contains(UINT32_MAX),
            "ID range mismatch");
    auto verify = [](HeapStorage<T>& storage, uint32_t start, uint32_t step) {
      alignas(64) std::array<T, 16> coords{};
      for (uint32_t row = start; row < 160; row += step) {
        require(storage.copy_coordinates(row + 5, coords), "missing vector");
        for (uint32_t d = 0; d < 13; ++d)
          require(coords[d] == coordinate<T>(row, d), "coordinate mismatch");
        for (uint32_t d = 13; d < 16; ++d)
          require(coords[d] == T{}, "nonzero padding");
      }
    };
    verify(*heap, 0, 1);
    auto verify_batch = [](HeapStorage<T>& storage) {
      constexpr size_t                        stride = 64 / sizeof(T);
      alignas(64) std::array<T, 337 * stride> coordinates;
      coordinates.fill(T(42));
      std::array<uint32_t, 337> ids;
      for (size_t i = 0; i < ids.size(); ++i)
        ids[i] = i % 11 == 0 ? UINT32_MAX : 5 + ((i * 31) % 160);
      const auto result =
          storage.copy_coordinates_batch(ids, coordinates, stride);
      require(result.backend_batch_calls ==
                  (ids.size() + storage.copy_batch_limit() - 1) /
                      storage.copy_batch_limit(),
              "incorrect backend batch count");
      for (size_t i = 0; i < ids.size(); ++i) {
        require(result.present[i] == (ids[i] != UINT32_MAX),
                "batch presence mismatch");
        if (ids[i] == UINT32_MAX)
          require(coordinates[i * stride] == T(42),
                  "missing batch row modified");
        else {
          for (uint32_t d = 0; d < 13; ++d)
            require(coordinates[i * stride + d] == coordinate<T>(ids[i] - 5, d),
                    "batch coordinate mismatch");
          for (uint32_t d = 13; d < storage.aligned_dim(); ++d)
            require(coordinates[i * stride + d] == T{},
                    "batch padding mismatch");
        }
      }
      require(
          storage.copy_coordinates_batch({}, {}, stride).backend_batch_calls ==
              0,
          "empty batch pinned");
      rejects([&] {
        storage.copy_coordinates_batch(
            ids, std::span<T>(coordinates).subspan(1), stride);
      });
      rejects([&] { storage.copy_coordinates_batch(ids, coordinates, 13); });
      rejects(
          [&] { storage.copy_coordinates_batch(ids, coordinates, SIZE_MAX); });
    };
    verify_batch(
        *heap);  // Tiny pool: batches must split instead of exhausting it.
    std::array<T, 13> row{};
    rejects([&] { heap->append(5, row); });  // Duplicate, before allocation.
    rejects([&] { heap->append(200, row); });
    rejects([&] { heap->append(0, std::span<const T>(row).first(12)); });
    require(!heap->failed() && heap->size() == 160, "validation mutated heap");
    alignas(64) std::array<T, 32> buffer{};
    rejects([&] {
      heap->copy_coordinates(5, std::span<T>(buffer).subspan(1, 16));
    });
    rejects([&] { heap->copy_coordinates(5, std::span<T>(buffer).first(13)); });
    buffer.fill(T(42));
    require(!heap->copy_coordinates(0, buffer) && buffer[0] == T(42),
            "missing read modified output");
    rejects([&] { HeapStorage<T>::create(options, heap_path); });
    const auto collision = root / "existing.meta";
    {
      std::ofstream out(collision);
      out << "preserve";
    }
    rejects([&] { heap->checkpoint(collision.string()); });
    {
      std::ifstream in(collision);
      std::string   content;
      in >> content;
      require(content == "preserve", "checkpoint overwrote user file");
    }
    heap->checkpoint(snapshot);
    rejects([&] { heap->append(0, row); });  // Immutable checkpoint generation.
    heap.reset();
    auto reader_options = options;
    reader_options.buffer_frames =
        64;  // Multi-page batches with per-shard slack.
    auto reopened = HeapStorage<T>::reopen(reader_options, heap_path, snapshot);
    std::array<std::exception_ptr, 4> errors{};
    std::vector<std::thread>          readers;
    for (uint32_t thread = 0; thread < 4; ++thread)
      readers.emplace_back([&, thread] {
        try {
          verify(*reopened, thread, 4);
          verify_batch(*reopened);
        } catch (...) {
          errors[thread] = std::current_exception();
        }
      });
    for (auto& reader : readers)
      reader.join();
    for (auto error : errors)
      if (error)
        std::rethrow_exception(error);
    verify(*reopened, 0, 1);
    rejects([&] { reopened->append(0, row); });
    auto wrong = options;
    ++wrong.dim;
    rejects([&] { HeapStorage<T>::reopen(wrong, heap_path, snapshot); });
    wrong = options;
    --wrong.id_capacity;
    rejects([&] { HeapStorage<T>::reopen(wrong, heap_path, snapshot); });
    if constexpr (std::is_same_v<T, int8_t>)
      rejects(
          [&] { HeapStorage<uint8_t>::reopen(options, heap_path, snapshot); });
    {
      std::ofstream corrupt(snapshot, std::ios::binary | std::ios::app);
      corrupt.put('x');
    }
    rejects([&] { HeapStorage<T>::reopen(options, heap_path, snapshot); });

    auto malformed =
        HeapStorage<T>::create(options, (root / "truncated.heap").string());
    const auto truncated = root / "truncated.bin";
    std::filesystem::copy_file(input, truncated);
    std::filesystem::resize_file(truncated, 9);
    rejects([&] { malformed->load_vectors(truncated.string()); });
    require(malformed->failed() && malformed->size() == 0,
            "bad stream published");
    const auto refused = (root / "refused.meta").string();
    rejects([&] { malformed->checkpoint(refused); });
    require(!std::filesystem::exists(refused), "failed build wrote checkpoint");
    auto trailing_heap =
        HeapStorage<T>::create(options, (root / "trailing.heap").string());
    const auto trailing = root / "trailing.bin";
    std::filesystem::copy_file(input, trailing);
    {
      std::ofstream out(trailing, std::ios::binary | std::ios::app);
      out.put('x');
    }
    rejects([&] { trailing_heap->load_vectors(trailing.string()); });
    require(trailing_heap->size() == 0 && trailing_heap->failed(),
            "trailing input published");
    auto wrong_dim = HeapStorage<T>::create(HeapOptions{12, 200, 4096, 2},
                                            (root / "wrong_dim.heap").string());
    rejects([&] { wrong_dim->load_vectors(input.string()); });
    require(wrong_dim->size() == 0, "wrong dimension input published");
    auto overflow =
        HeapStorage<T>::create(options, (root / "overflow.heap").string());
    rejects([&] { overflow->load_vectors(input.string(), 100); });
    require(overflow->size() == 0 && overflow->failed(),
            "capacity overflow published");
    // Explicit sparse IDs, including the last valid ID, are preserved too.
    auto sparse =
        HeapStorage<T>::create(options, (root / "sparse.heap").string());
    row.fill(T(7));
    sparse->append(199, row);
    sparse->append(0, row);
    require(
        sparse->size() == 2 && sparse->contains(199) && !sparse->contains(1),
        "sparse IDs remapped");
    const auto sparse_meta = (root / "sparse.meta").string();
    sparse->checkpoint(sparse_meta);
    sparse.reset();
    sparse = HeapStorage<T>::reopen(options, (root / "sparse.heap").string(),
                                    sparse_meta);
    require(sparse->contains(199) && sparse->copy_coordinates(199, buffer) &&
                buffer[12] == T(7) && buffer[13] == T{},
            "sparse reopen mismatch");
    auto empty =
        HeapStorage<T>::create(options, (root / "empty.heap").string());
    const auto empty_meta = (root / "empty.meta").string();
    empty->checkpoint(empty_meta);
    empty.reset();
    empty = HeapStorage<T>::reopen(options, (root / "empty.heap").string(),
                                   empty_meta);
    require(empty->size() == 0 && !empty->contains(0), "empty reopen mismatch");
    if constexpr (std::is_same_v<T, float>) {
      const auto bad = root / "infinite.bin";
      write_vectors<float>(bad, true);
      auto partial =
          HeapStorage<float>::create(options, (root / "partial.heap").string());
      rejects([&] { partial->load_vectors(bad.string()); });
      require(
          partial->size() == 3 && !partial->contains(3) && partial->failed(),
          "nonfinite row published or partial build not failed");
      rejects([&] { partial->checkpoint((root / "partial.meta").string()); });
      row[0] = std::numeric_limits<float>::quiet_NaN();
      auto valid =
          HeapStorage<float>::create(options, (root / "nan.heap").string());
      rejects([&] { valid->append(0, row); });
      require(!valid->failed() && valid->size() == 0,
              "NaN append mutated state");
    }
    auto invalid = options;
    invalid.dim = 0;
    const auto absent = (root / "invalid.heap").string();
    rejects([&] { HeapStorage<T>::create(invalid, absent); });
    require(!std::filesystem::exists(absent), "invalid options created heap");
    invalid = options;
    invalid.dim = UINT32_MAX;
    rejects([&] { HeapStorage<T>::create(invalid, absent); });
    invalid = options;
    invalid.buffer_frames = 1;
    rejects([&] { HeapStorage<T>::create(invalid, absent); });
    invalid = options;
    invalid.page_size = 513;
    rejects([&] { HeapStorage<T>::create(invalid, absent); });
    invalid = options;
    invalid.id_capacity = 0;
    rejects([&] { HeapStorage<T>::create(invalid, absent); });
    require(!std::filesystem::exists(absent),
            "invalid options created artifacts");
  }
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    auto root =
        std::filesystem::path(argv[1]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(root);
    exercise<float>(root / "float");
    exercise<int8_t>(root / "int8");
    exercise<uint8_t>(root / "uint8");
    std::cout << "IVF heap adapter/streaming/checkpoint tests passed\n";
    return 0;
  } catch (const diskann::ANNException& error) {
    std::cerr << error.message() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
