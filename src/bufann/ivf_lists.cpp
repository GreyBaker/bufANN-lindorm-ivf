#include "bufann/ivf_lists.h"
#include "bufann/ivf_io.h"
#include "bufann/ivf_artifacts.h"
#include "ivf_lists_internal.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <type_traits>

namespace diskann::ivf {
  namespace {
    void check(bool ok, const char* message) {
      if (!ok)
        throw std::invalid_argument(message);
    }
    uint64_t mul(uint64_t a, uint64_t b) {
      check(!b || a <= UINT64_MAX / b, "IVF list workspace overflow");
      return a * b;
    }
    void add(uint64_t& a, uint64_t b) {
      check(a <= UINT64_MAX - b, "IVF list workspace overflow");
      a += b;
    }
    void header(std::ifstream& in, BinMetadata expected) {
      uint32_t shape[2]{};
      in.read(reinterpret_cast<char*>(shape), sizeof(shape));
      check(bool(in) && shape[0] == expected.rows && shape[1] == expected.cols,
            "IVF list input header changed");
    }
    bool nearer(const PQCandidate& a, const PQCandidate& b) {
      return a.squared_distance < b.squared_distance ||
             (a.squared_distance == b.squared_distance && a.id < b.id);
    }
    template<typename T>
    T read(std::istream& in) {
      T value{};
      in.read(reinterpret_cast<char*>(&value), sizeof(value));
      check(bool(in), "Truncated IVF list artifact");
      return value;
    }
    template<typename T>
    void write(std::ostream& out, const T& value) {
      out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    constexpr uint64_t list_header_bytes = 40;
    uint64_t list_file_bytes(uint32_t nlist, uint32_t count, uint32_t chunks) {
      uint64_t bytes = list_header_bytes + 8;  // Header and checksum trailer.
      add(bytes, mul(uint64_t(nlist) + 1, 8));
      add(bytes, mul(count, uint64_t(chunks) + 4));
      return bytes;
    }
  }  // namespace

  uint32_t PostingLists::routing_workers(const ListBuildOptions& options) {
    check(options.base_count && options.build_threads &&
              options.build_threads <= 256 && options.routing_block_rows,
          "Invalid IVF list routing worker/block settings");
    const uint64_t blocks =
        (uint64_t(options.base_count) + options.routing_block_rows - 1) /
        options.routing_block_rows;
    return uint32_t(std::min<uint64_t>(options.build_threads, blocks));
  }

  uint64_t PostingLists::estimate_build_bytes(uint32_t dim, uint32_t nlist,
                                              DataType                type,
                                              const RouterOptions&    routing,
                                              const ListBuildOptions& options) {
    check(dim && nlist && options.chunks && options.chunks <= dim &&
              uint64_t(options.first_id) + options.base_count <=
                  options.id_capacity,
          "Invalid IVF list build options");
    const auto workers = routing_workers(options);
    check(options.max_assignment_searches &&
              options.base_count <= options.max_assignment_searches,
          "IVF list assignment exceeds search count allowance");
    if (options.exact_assignment || routing.mode == RouterMode::Exact)
      check(mul(options.base_count, nlist) <=
                options.max_exact_distance_evaluations,
            "Exact IVF list assignment exceeds distance count allowance");
    uint64_t bytes = mul(options.base_count, uint64_t(options.chunks) + 8);
    add(bytes, mul(uint64_t(nlist) + 1, 16));
    const auto buffered_rows = std::min<uint64_t>(
        options.base_count, uint64_t(workers) * options.routing_block_rows);
    add(bytes, mul(mul(buffered_rows, dim), element_size(type)));
    // Conservative heaps/results/visited allowance per active reader, including
    // exact enumeration and arbitrarily large declared centroid rerank factors.
    uint64_t scratch = mul(nlist, 66);
    add(scratch, mul(dim, 2 * sizeof(float)));
    add(scratch, 65536);
    add(bytes, mul(workers, scratch));
    if (workers > 1)
      add(bytes, mul(workers, 8ull * 1024 * 1024));
    add(bytes, options.chunks);
    add(bytes, 65536);
    return bytes;
  }

  template<typename T>
  PostingLists PostingLists::build(const std::string&      vectors,
                                   const std::string&      codes_path,
                                   const CentroidRouter&   router,
                                   const ListBuildOptions& options) {
    static_assert(std::endian::native == std::endian::little);
    constexpr DataType type =
        std::is_same_v<T, float>
            ? DataType::Float
            : (std::is_same_v<T, int8_t> ? DataType::Int8 : DataType::UInt8);
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int8_t> ||
                  std::is_same_v<T, uint8_t>);
    const auto workers = routing_workers(options);
    const auto bytes = estimate_build_bytes(router.dim(), router.nlist(), type,
                                            router.options(), options);
    check(bytes <= options.memory_budget_bytes && bytes <= SIZE_MAX,
          "IVF list build exceeds workspace allowance");
    const auto metadata = inspect_vectors(vectors, type, router.dim());
    check(options.base_count <= metadata.rows,
          "IVF list base prefix exceeds input rows");
    inspect_pq_codes(codes_path, metadata.rows, options.chunks);
    PostingLists result;
    result._chunks = options.chunks;
    result._dim = router.dim();
    result._type = type;
    result._first_id = options.first_id;
    result._id_capacity = options.id_capacity;
    result._estimated_bytes =
        bytes;  // Excludes the caller-owned router/heap/codebook.
    std::vector<uint32_t> labels(options.base_count);
    result._offsets.resize(size_t(router.nlist()) + 1, 0);
    std::ifstream input(vectors, std::ios::binary);
    header(input, metadata);
    const auto progress_start = std::chrono::steady_clock::now();
    auto       progress = [&](const char* phase, uint32_t done) {
      if (options.base_count >= 1000000)
        std::clog << "[IVF lists] phase=" << phase << " rows=" << done << '/'
                  << options.base_count << " elapsed_s="
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - progress_start)
                         .count()
                  << std::endl;
    };
    {
      const auto buffered_rows = std::min<uint64_t>(
          options.base_count, uint64_t(workers) * options.routing_block_rows);
      const auto cells = mul(buffered_rows, router.dim());
      check(mul(cells, sizeof(T)) <=
                uint64_t(std::numeric_limits<std::streamsize>::max()),
            "IVF list input block exceeds streamsize");
      std::vector<T> buffer(size_t(cells), T{});
      uint64_t       next_report = 1000000;
      std::clog << "[IVF lists] phase=routing_start workers=" << workers
                << " block_rows=" << options.routing_block_rows
                << " buffered_rows=" << buffered_rows << std::endl;
      detail::route_list_blocks(
          options.base_count, options.routing_block_rows, workers,
          [&](uint32_t, uint32_t count) {
            input.read(
                reinterpret_cast<char*>(buffer.data()),
                std::streamsize(uint64_t(count) * router.dim() * sizeof(T)));
            check(bool(input), "Truncated IVF list vector stream");
          },
          [&](uint32_t r, uint32_t local) {
            const auto row = std::span<const T>(buffer).subspan(
                size_t(local) * router.dim(), router.dim());
            const auto label = router.assign(row, options.exact_assignment);
            check(label < router.nlist(), "Invalid IVF list assignment");
            labels[r] = label;
          },
          [&](uint32_t done) {
            if (done >= next_report || done == options.base_count) {
              progress("routing", done);
              next_report = (uint64_t(done) / 1000000 + 1) * 1000000;
            }
          });
    }
    // Joined labels only. Histogram and source-order PQ grouping remain serial,
    // independent of worker completion order and without shared offset writes.
    for (auto label : labels)
      ++result._offsets[size_t(label) + 1];
    for (size_t i = 1; i < result._offsets.size(); ++i)
      result._offsets[i] += result._offsets[i - 1];
    auto cursor = result._offsets;
    result._ids.resize(options.base_count);
    result._codes.resize(size_t(mul(options.base_count, options.chunks)));
    std::ifstream codes(codes_path, std::ios::binary);
    header(codes, {metadata.rows, options.chunks});
    std::vector<uint8_t> code(options.chunks);
    for (uint32_t r = 0; r < options.base_count; ++r) {
      codes.read(reinterpret_cast<char*>(code.data()), code.size());
      check(bool(codes), "Truncated IVF list code stream");
      const size_t position = static_cast<size_t>(cursor[labels[r]]++);
      result._ids[position] = options.first_id + r;
      std::copy(code.begin(), code.end(),
                result._codes.begin() + position * options.chunks);
      if ((r + 1) % 1000000 == 0 || r + 1 == options.base_count)
        progress("pq_grouping", r + 1);
    }
    return result;
  }

  void PostingLists::save(const std::string& path) const {
    static_assert(std::endian::native == std::endian::little);
    check(nlist() > 0 && size() > 0 && _chunks > 0 && _dim > 0 &&
              _codes.size() == uint64_t(size()) * _chunks,
          "Cannot save an empty/moved-from IVF list object");
    claim_artifact(path);
    std::ofstream out(path, std::ios::binary);
    out.write("IVFLST1\0", 8);
    for (uint32_t value : {1u, static_cast<uint32_t>(_type), _dim, nlist(),
                           _chunks, size(), _first_id, _id_capacity})
      write(out, value);
    out.write(reinterpret_cast<const char*>(_offsets.data()),
              _offsets.size() * sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(_ids.data()),
              _ids.size() * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(_codes.data()), _codes.size());
    out.close();
    check(bool(out), "IVF list artifact write failed");
    const auto expected = list_file_bytes(nlist(), size(), _chunks);
    check(std::filesystem::file_size(path) == expected - 8,
          "IVF list artifact framing mismatch");
    const auto    hash = artifact_hash_prefix(path, expected - 8);
    std::ofstream trailer(path, std::ios::binary | std::ios::app);
    write(trailer, hash);
    trailer.close();
    check(bool(trailer), "IVF list checksum write failed");
  }

  PostingLists PostingLists::load(const std::string& path,
                                  uint64_t           workspace_bytes) {
    static_assert(std::endian::native == std::endian::little);
    std::ifstream in(path, std::ios::binary);
    char          magic[8]{};
    in.read(magic, sizeof(magic));
    check(bool(in) && std::equal(magic, magic + 8, "IVFLST1\0"),
          "Invalid IVF list magic");
    check(read<uint32_t>(in) == 1, "Unsupported IVF list version");
    const auto type = read<uint32_t>(in);
    check(type <= static_cast<uint32_t>(DataType::UInt8),
          "Invalid IVF list dtype");
    PostingLists result;
    result._type = static_cast<DataType>(type);
    result._dim = read<uint32_t>(in);
    const auto nlist = read<uint32_t>(in);
    result._chunks = read<uint32_t>(in);
    const auto count = read<uint32_t>(in);
    result._first_id = read<uint32_t>(in);
    result._id_capacity = read<uint32_t>(in);
    check(result._dim > 0 && nlist > 0 && nlist <= INT32_MAX &&
              result._chunks > 0 && result._chunks <= result._dim &&
              count > 0 &&
              uint64_t(result._first_id) + count <= result._id_capacity,
          "Invalid IVF list artifact shape");
    const auto bytes = list_file_bytes(nlist, count, result._chunks);
    check(std::filesystem::file_size(path) == bytes &&
              bytes <= uint64_t(INT64_MAX),
          "Invalid IVF list artifact length");
    uint64_t estimate = bytes;
    add(estimate, (uint64_t(count) + 7) / 8);
    add(estimate, 65536);
    check(estimate <= workspace_bytes && estimate <= SIZE_MAX,
          "IVF list load exceeds workspace allowance");
    in.seekg(static_cast<std::streamoff>(bytes - 8));
    const auto hash = read<uint64_t>(in);
    check(artifact_hash_prefix(path, bytes - 8) == hash,
          "IVF list checksum mismatch");
    in.seekg(list_header_bytes);
    result._offsets.resize(size_t(nlist) + 1);
    result._ids.resize(count);
    result._codes.resize(size_t(mul(count, result._chunks)));
    in.read(reinterpret_cast<char*>(result._offsets.data()),
            result._offsets.size() * sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(result._ids.data()),
            result._ids.size() * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(result._codes.data()),
            result._codes.size());
    check(bool(in), "Truncated IVF list payload");
    check(result._offsets.front() == 0 && result._offsets.back() == count,
          "Invalid IVF list offset endpoints");
    std::vector<uint8_t> seen((uint64_t(count) + 7) / 8, 0);
    for (uint32_t c = 0; c < nlist; ++c) {
      const auto begin = result._offsets[c], end = result._offsets[c + 1];
      check(begin <= end && end <= count, "Invalid IVF list offset order");
      for (uint64_t i = begin; i < end; ++i) {
        const auto id = result._ids[i];
        check(id >= result._first_id &&
                  uint64_t(id) < uint64_t(result._first_id) + count,
              "IVF list ID outside base range");
        check(i == begin || result._ids[i - 1] < id,
              "IVF list IDs not in input order");
        const uint64_t relative = uint64_t(id) - result._first_id;
        const uint8_t  bit = uint8_t(1u << (relative % 8));
        check(!(seen[relative / 8] & bit), "Duplicate IVF list ID");
        seen[relative / 8] |= bit;
      }
    }
    result._estimated_load_bytes = estimate;
    return result;
  }

  std::span<const uint32_t> PostingLists::ids(uint32_t list) const {
    check(list < nlist(), "Invalid IVF list ID");
    return std::span<const uint32_t>(_ids).subspan(
        _offsets[list], _offsets[list + 1] - _offsets[list]);
  }
  std::span<const uint8_t> PostingLists::codes(uint32_t list) const {
    check(list < nlist(), "Invalid IVF list ID");
    return std::span<const uint8_t>(_codes).subspan(
        _offsets[list] * _chunks,
        (_offsets[list + 1] - _offsets[list]) * _chunks);
  }
  ListScan PostingLists::scan(std::span<const uint32_t> lists,
                              std::span<const float> distances, uint32_t budget,
                              uint64_t workspace_bytes) const {
    check(budget > 0 && distances.size() == size_t(_chunks) * 256,
          "Invalid IVF scan buffers/budget");
    check(lists.size() <= nlist(), "Too many IVF scan lists");
    // Heap growth, retained output and selected-list validation. Not a hard
    // allocator/RSS cap; caller-owned lists and ADC table are excluded.
    uint64_t bytes = mul(std::min(budget, size()), 48);
    add(bytes, mul(lists.size(), 4));
    check(bytes <= workspace_bytes, "IVF scan exceeds workspace allowance");
    for (float distance : distances)
      check(std::isfinite(distance) && distance >= 0,
            "Invalid IVF ADC distance table");
    std::vector<uint32_t> selected(lists.begin(), lists.end());
    std::sort(selected.begin(), selected.end());
    check(selected.empty() || selected.back() < nlist(),
          "Invalid IVF scan list ID");
    check(
        std::adjacent_find(selected.begin(), selected.end()) == selected.end(),
        "Duplicate IVF scan list ID");
    std::priority_queue<PQCandidate, std::vector<PQCandidate>,
                        decltype(&nearer)>
             best(&nearer);
    ListScan result;
    for (uint32_t list : lists) {
      const auto list_ids = ids(list);
      const auto list_codes = codes(list);
      for (size_t r = 0; r < list_ids.size(); ++r) {
        float distance = 0;
        for (uint32_t c = 0; c < _chunks; ++c)
          distance += distances[size_t(c) * 256 + list_codes[r * _chunks + c]];
        check(std::isfinite(distance), "IVF ADC sum overflow");
        const PQCandidate candidate{list_ids[r], distance};
        if (best.size() < budget)
          best.push(candidate);
        else if (nearer(candidate, best.top())) {
          best.pop();
          best.push(candidate);
        }
      }
      result.scanned_codes += list_ids.size();
    }
    result.candidates.resize(best.size());
    for (size_t i = best.size(); i > 0; --i) {
      result.candidates[i - 1] = best.top();
      best.pop();
    }
    return result;
  }

  ScanProfile PostingLists::profile_scan(std::span<const uint8_t> codes,
                                         uint32_t                 chunks,
                                         std::span<const float>   distances,
                                         uint32_t budget, uint32_t repeats,
                                         uint64_t workspace_bytes) {
    check(chunks && !codes.empty() && codes.size() % chunks == 0 && repeats &&
              budget,
          "Invalid scan calibration buffers/repeats");
    const auto rows = codes.size() / chunks;
    check(rows <= INT32_MAX, "Scan calibration population too large");
    uint64_t retained = mul(rows, uint64_t(chunks) + 4);
    add(retained, 16);
    uint64_t bytes = retained;
    add(bytes, mul(std::min<size_t>(rows, budget), 48));
    add(bytes, 4);
    check(bytes <= workspace_bytes,
          "Scan calibration exceeds workspace allowance");
    PostingLists lists;
    lists._chunks = chunks;
    lists._offsets = {0, rows};
    lists._codes.assign(codes.begin(), codes.end());
    lists._ids.resize(rows);
    std::iota(lists._ids.begin(), lists._ids.end(), 0);
    const uint32_t selected[]{0};
    // Validate and warm the same scanner once, separately from timed repeats.
    const auto scratch_bytes = workspace_bytes - retained;
    lists.scan(selected, distances, budget, scratch_bytes);
    ScanProfile profile;
    const auto  start = std::chrono::steady_clock::now();
    for (uint32_t r = 0; r < repeats; ++r) {
      const auto result =
          lists.scan(selected, distances, budget, scratch_bytes);
      profile.scanned_codes += result.scanned_codes;
    }
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    profile.per_code_us = us / profile.scanned_codes;
    check(std::isfinite(profile.per_code_us) && profile.per_code_us > 0,
          "Invalid measured scan coefficient");
    return profile;
  }

  template PostingLists PostingLists::build<float>(const std::string&,
                                                   const std::string&,
                                                   const CentroidRouter&,
                                                   const ListBuildOptions&);
  template PostingLists PostingLists::build<int8_t>(const std::string&,
                                                    const std::string&,
                                                    const CentroidRouter&,
                                                    const ListBuildOptions&);
  template PostingLists PostingLists::build<uint8_t>(const std::string&,
                                                     const std::string&,
                                                     const CentroidRouter&,
                                                     const ListBuildOptions&);
}  // namespace diskann::ivf
