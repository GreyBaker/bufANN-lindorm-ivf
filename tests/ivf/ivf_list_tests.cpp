#include "bufann/ivf_lists.h"
#include "bufann/ivf_artifacts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace {
  using namespace diskann::ivf;
  void require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<typename F>
  void rejects(F&& f) {
    try {
      f();
    } catch (const std::exception&) {
      return;
    }
    throw std::runtime_error("expected rejection");
  }
  bool nearer(PQCandidate a, PQCandidate b) {
    return a.squared_distance < b.squared_distance ||
           (a.squared_distance == b.squared_distance && a.id < b.id);
  }
  template<typename T>
  void patch(const std::filesystem::path& path, uint64_t offset, T value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(offset);
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    require(bool(file), "fixture patch failed");
  }
  void rehash(const std::filesystem::path& path) {
    const auto bytes = std::filesystem::file_size(path);
    patch(path, bytes - 8, artifact_hash_prefix(path.string(), bytes - 8));
  }
  template<typename T>
  void exercise(const std::filesystem::path& root) {
    constexpr uint32_t dim = 13, rows = 128, base = 112, chunks = 3;
    const auto         vectors = (root / "vectors.bin").string(),
               codes = (root / "codes.bin").string();
    std::filesystem::create_directories(root);
    std::vector<float> centroids;
    for (uint32_t c = 0; c < 8; ++c)
      for (uint32_t d = 0; d < dim; ++d)
        centroids.push_back(float(c * 16 + d % 3));
    auto                 router = CentroidRouter::build(dim, centroids);
    std::vector<uint8_t> original_codes;
    {
      std::ofstream out(vectors, std::ios::binary), pq(codes, std::ios::binary);
      const uint32_t shape[]{rows, dim}, code_shape[]{rows, chunks};
      out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
      pq.write(reinterpret_cast<const char*>(code_shape), sizeof(code_shape));
      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t d = 0; d < dim; ++d) {
          T value = T((r % 4) * 16 + d % 3);
          if constexpr (std::is_same_v<T, float>)
            if (r >= base)
              value = std::numeric_limits<float>::infinity();
          out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        for (uint32_t c = 0; c < chunks; ++c) {
          uint8_t code = uint8_t((r * 7 + c * 13) % 256);
          original_codes.push_back(code);
          pq.put(static_cast<char>(code));
        }
      }
    }
    ListBuildOptions options{base, 5, 200, chunks};
    options.exact_assignment = true;
    const auto lists = PostingLists::build<T>(vectors, codes, *router, options);
    require(
        lists.size() == base && lists.nlist() == 8 && lists.chunks() == chunks,
        "list shape mismatch");
    uint32_t total = 0;
    for (uint32_t c = 0; c < 8; ++c) {
      const auto ids = lists.ids(c);
      const auto packed = lists.codes(c);
      require(ids.size() == (c < 4 ? base / 4 : 0), "list occupancy mismatch");
      for (size_t i = 0; i < ids.size(); ++i) {
        const auto r = ids[i] - 5;
        require(r < base && r % 4 == c && (i == 0 || ids[i] > ids[i - 1]),
                "IDs remapped or reordered");
        for (uint32_t j = 0; j < chunks; ++j)
          require(packed[i * chunks + j] == original_codes[r * chunks + j],
                  "PQ code repacking changed bytes");
      }
      total += ids.size();
    }
    require(total == base, "base row lost/duplicated");
    options.exact_assignment = false;
    const auto ann = PostingLists::build<T>(vectors, codes, *router, options);
    for (uint32_t c = 0; c < 8; ++c)
      require(std::equal(lists.ids(c).begin(), lists.ids(c).end(),
                         ann.ids(c).begin(), ann.ids(c).end()),
              "tiny HNSW assignment differs from oracle");
    std::vector<float> table(chunks * 256);
    for (uint32_t c = 0; c < chunks; ++c)
      for (uint32_t i = 0; i < 256; ++i)
        table[c * 256 + i] =
            float((int(i) - int(50 + c)) * (int(i) - int(50 + c)));
    const std::array<uint32_t, 4> selected{
        3, 6, 0, 1};  // Empty list and nonascending order.
    std::vector<PQCandidate> oracle;
    for (uint32_t r = 0; r < base; ++r)
      if (r % 4 != 2) {
        float distance = 0;
        for (uint32_t c = 0; c < chunks; ++c)
          distance += table[c * 256 + original_codes[r * chunks + c]];
        oracle.push_back({r + 5, distance});
      }
    std::sort(oracle.begin(), oracle.end(), nearer);
    for (uint32_t budget : {1u, 10u, 200u}) {
      const auto scanned = lists.scan(selected, table, budget);
      auto       expected = oracle;
      expected.resize(std::min<size_t>(budget, expected.size()));
      require(scanned.candidates == expected && scanned.scanned_codes == 84,
              "ADC scan disagrees with oracle");
    }
    std::fill(table.begin(), table.end(), 1);
    const std::array<uint32_t, 8> all{7, 6, 5, 4, 3, 2, 1, 0};
    const auto                    tied = lists.scan(all, table, 20);
    for (uint32_t r = 0; r < 20; ++r)
      require(tied.candidates[r] == PQCandidate{r + 5, 3},
              "ADC ties not stable by ID");
    const auto saved = root / "lists.bin";
    lists.save(saved.string());
    auto loaded = PostingLists::load(saved.string());
    require(loaded.dim() == dim && loaded.data_type() == lists.data_type() &&
                loaded.first_id() == 5 && loaded.id_capacity() == 200 &&
                loaded.estimated_load_bytes() > 0 &&
                loaded.scan(all, table, 20).candidates == tied.candidates,
            "list save/load changed metadata or ADC");
    for (uint32_t c = 0; c < 8; ++c) {
      require(std::equal(lists.ids(c).begin(), lists.ids(c).end(),
                         loaded.ids(c).begin(), loaded.ids(c).end()),
              "list load changed membership");
      require(std::equal(lists.codes(c).begin(), lists.codes(c).end(),
                         loaded.codes(c).begin(), loaded.codes(c).end()),
              "list load changed code bytes");
    }
    rejects([&] { lists.save(saved.string()); });
    rejects([&] {
      PostingLists::load(saved.string(), loaded.estimated_load_bytes() - 1);
    });
    require(PostingLists::load(saved.string(), loaded.estimated_load_bytes())
                    .size() == base,
            "list exact workspace boundary refused");
    const auto second = root / "lists_repeat.bin";
    lists.save(second.string());
    require(artifact_digest(second.string()) == artifact_digest(saved.string()),
            "list persistence not deterministic");
    auto       moved_from = PostingLists::load(saved.string());
    auto       moved = std::move(moved_from);
    const auto refused_save = (root / "moved_from.bin").string();
    rejects([&] { moved_from.save(refused_save); });
    require(moved.size() == base && !std::filesystem::exists(refused_save),
            "moved-from save created an artifact");
    auto corrupt = [&](const char* name, auto edit, bool recompute = true) {
      const auto path = root / name;
      std::filesystem::copy_file(saved, path);
      edit(path);
      if (recompute)
        rehash(path);
      rejects([&] { PostingLists::load(path.string()); });
    };
    corrupt("bad_magic", [&](auto path) { patch(path, 0, uint8_t('X')); });
    corrupt("bad_version", [&](auto path) { patch(path, 8, uint32_t(2)); });
    corrupt("bad_type", [&](auto path) { patch(path, 12, uint32_t(3)); });
    corrupt("bad_nlist", [&](auto path) { patch(path, 20, UINT32_MAX); });
    corrupt("bad_chunks", [&](auto path) { patch(path, 24, uint32_t(0)); });
    corrupt("bad_offset_start",
            [&](auto path) { patch(path, 40, uint64_t(1)); });
    corrupt("bad_offset_order",
            [&](auto path) { patch(path, 40 + 2 * 8, uint64_t(1)); });
    corrupt("bad_offset_end",
            [&](auto path) { patch(path, 40 + 8 * 8, uint64_t(base - 1)); });
    constexpr uint64_t id_start = 40 + 9 * 8;
    corrupt("bad_id_range",
            [&](auto path) { patch(path, id_start, uint32_t(200)); });
    // Different lists: local ordering still holds, but global membership is
    // duplicated.
    corrupt("bad_global_duplicate", [&](auto path) {
      patch(path, id_start + base / 4 * 4, uint32_t(5));
    });
    corrupt("bad_local_order",
            [&](auto path) { patch(path, id_start + 4, uint32_t(5)); });
    corrupt(
        "bad_checksum", [&](auto path) { patch(path, id_start, uint32_t(7)); },
        false);
    corrupt(
        "trailing",
        [&](auto path) {
          std::ofstream out(path, std::ios::binary | std::ios::app);
          out.put('x');
        },
        false);
    corrupt(
        "truncated", [&](auto path) { std::filesystem::resize_file(path, 10); },
        false);
    require(
        lists.scan(std::array<uint32_t, 1>{7}, table, 10).candidates.empty(),
        "empty list produced candidates");
    require(lists.scan({}, table, 10).scanned_codes == 0,
            "empty selection scanned rows");
    rejects([&] { lists.scan(std::array<uint32_t, 2>{1, 1}, table, 10); });
    rejects([&] { lists.scan(std::array<uint32_t, 1>{8}, table, 10); });
    rejects([&] { lists.scan(all, table, 0); });
    rejects([&] { lists.scan(all, table, 10, 1); });
    rejects(
        [&] { lists.scan(all, std::span<const float>(table).first(2), 10); });
    rejects([&] { lists.ids(8); });
    rejects([&] { lists.codes(UINT32_MAX); });
    table[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { lists.scan(all, table, 10); });
    table[0] = -1;
    rejects([&] { lists.scan(all, table, 10); });
    std::fill(table.begin(), table.end(), std::numeric_limits<float>::max());
    rejects([&] { lists.scan(all, table, 10); });
    std::fill(table.begin(), table.end(), 1);
    std::array<std::exception_ptr, 4> errors{};
    std::vector<std::thread>          readers;
    for (uint32_t t = 0; t < errors.size(); ++t)
      readers.emplace_back([&, t] {
        try {
          for (uint32_t i = 0; i < 20; ++i)
            require(lists.scan(all, table, 20).candidates == tied.candidates,
                    "concurrent scan changed output");
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    for (auto& reader : readers)
      reader.join();
    for (auto error : errors)
      if (error)
        std::rethrow_exception(error);
    options.first_id = UINT32_MAX - base;
    options.id_capacity = UINT32_MAX;
    const auto high_ids =
        PostingLists::build<T>(vectors, codes, *router, options);
    require(high_ids.ids(3).back() == UINT32_MAX - 1,
            "last valid external ID remapped");
    options.first_id = 5;
    options.id_capacity = 200;
    options.memory_budget_bytes = ann.estimated_build_bytes() - 1;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, options); });
    options.memory_budget_bytes = 1000000;
    options.first_id = 199;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, options); });
    options.first_id = 5;
    options.base_count = rows + 1;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, options); });
    options.base_count = base;
    options.chunks = 2;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, options); });
    options.chunks = chunks;
    auto invalid = options;
    invalid.base_count = 0;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, invalid); });
    invalid = options;
    invalid.chunks = 0;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, invalid); });
    invalid = options;
    invalid.id_capacity = 0;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, invalid); });
    invalid = options;
    invalid.max_assignment_searches = base - 1;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, invalid); });
    invalid = options;
    invalid.exact_assignment = true;
    invalid.max_exact_distance_evaluations = base * 8 - 1;
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, invalid); });
    invalid.exact_assignment = false;
    invalid.max_exact_distance_evaluations = 1;
    require(
        PostingLists::build<T>(vectors, codes, *router, invalid).size() == base,
        "ANN list assignment reused exact work cap");
    if constexpr (std::is_same_v<T, float>) {
      options.base_count = rows;
      rejects(
          [&] { PostingLists::build<T>(vectors, codes, *router, options); });
      options.base_count = base;
    }
    {
      std::ofstream out(codes, std::ios::binary | std::ios::app);
      out.put('x');
    }
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, options); });
    std::filesystem::resize_file(codes, 10);
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, options); });
    std::filesystem::resize_file(vectors, 9);
    rejects([&] { PostingLists::build<T>(vectors, codes, *router, options); });
  }
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected fixture directory");
    const auto root =
        std::filesystem::path(argv[1]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    exercise<float>(root / "float");
    exercise<int8_t>(root / "int8");
    exercise<uint8_t>(root / "uint8");
    std::cout << "IVF list construction/ADC tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
