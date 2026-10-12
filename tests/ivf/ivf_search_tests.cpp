#include "bufann/ivf_search.h"
#include "bufann/ivf_index.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace {
  using namespace diskann::ivf;
  RouterMode         fixture_mode = RouterMode::Hnswlib;
  constexpr uint32_t dim = 13, chunks = 3, rows = 180, base = 160, first = 5;
  void               require(bool ok, const char* message) {
    if (!ok)
      throw std::runtime_error(message);
  }
  template<typename F>
  void rejects(F&& f) {
    try {
      f();
    } catch (const std::exception&) {
      return;
    } catch (const diskann::ANNException&) {
      return;
    }
    throw std::runtime_error("expected rejection");
  }
  template<typename T>
  void write(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  void pivots(const std::string& path) {
    std::array<uint64_t, 5> offsets{4096, 0, 0, 0, 0};
    offsets[1] = offsets[0] + 8 + 256 * dim * 4;
    offsets[2] = offsets[1] + 8 + dim * 4;
    offsets[3] = offsets[2] + 8 + dim * 4;
    offsets[4] = offsets[3] + 8 + (chunks + 1) * 4;
    std::ofstream out(path, std::ios::binary);
    write(out, uint32_t(5));
    write(out, uint32_t(1));
    for (auto offset : offsets)
      write(out, offset);
    out.seekp(offsets[0]);
    write(out, uint32_t(256));
    write(out, dim);
    for (uint32_t i = 0; i < 256; ++i)
      for (uint32_t d = 0; d < dim; ++d)
        write(out, float(i) * .5f - 50 + d * .25f);
    write(out, dim);
    write(out, uint32_t(1));
    for (uint32_t d = 0; d < dim; ++d)
      write(out, float(d) * .25f);
    write(out, dim);
    write(out, uint32_t(1));
    for (uint32_t d = 0; d < dim; ++d)
      write(out, dim - 1 - d);
    write(out, chunks + 1);
    write(out, uint32_t(1));
    for (uint32_t offset : {0u, 4u, 8u, 13u})
      write(out, offset);
    require(bool(out), "PQ fixture write failed");
  }
  bool nearer(const ExactHit& a, const ExactHit& b) {
    return a.squared_distance < b.squared_distance ||
           (a.squared_distance == b.squared_distance && a.id < b.id);
  }
  template<typename T>
  void exercise(const std::filesystem::path& root, DataType type) {
    std::filesystem::create_directories(root);
    const auto vectors = (root / "vectors.bin").string(),
               codes_path = (root / "codes.bin").string();
    const auto pivot_path = (root / "pivots.bin").string();
    pivots(pivot_path);
    PQCodebook<T> pq(pivot_path, dim);
    rejects([&] { PQCodebook<T> mismatch(pivot_path, dim - 1); });
    rejects([&] { PQCodebook<T> mismatch(pivot_path, 0); });
    rejects([&] { PQCodebook<T> mismatch(pivot_path, dim, chunks + 1); });
    std::vector<T>       data;
    std::vector<uint8_t> codes(rows * chunks);
    {
      std::ofstream out(vectors, std::ios::binary),
          encoded(codes_path, std::ios::binary);
      write(out, rows);
      write(out, dim);
      write(encoded, rows);
      write(encoded, chunks);
      for (uint32_t r = 0; r < rows; ++r) {
        std::array<T, dim> row;
        for (uint32_t d = 0; d < dim; ++d)
          row[d] = T(int(r % 8) * 12 + int((r / 8 + d) % 7) -
                     (std::is_same_v<T, uint8_t> ? 0 : 40));
        for (T x : row)
          write(out, x);
        data.insert(data.end(), row.begin(), row.end());
        pq.encode(row.data(), codes.data() + r * chunks);
        encoded.write(reinterpret_cast<const char*>(codes.data() + r * chunks),
                      chunks);
      }
      require(bool(out) && bool(encoded), "vector/code fixture write failed");
    }
    BuildConfig training;
    training.data_type = type;
    training.dim = dim;
    training.base_count = base;
    training.nlist = 8;
    training.max_reps = 3;
    training.build_threads = 4;
    const auto    trained = train_centroids_hnsw(vectors, training);
    RouterOptions routing;
    routing.native_modes = fixture_mode != RouterMode::Hnswlib;
    routing.mode = fixture_mode;
    auto       router = CentroidRouter::build(dim, trained.centroids, routing);
    const auto lists = PostingLists::build<T>(vectors, codes_path, *router,
                                              {base, first, 200, chunks});
    const HeapOptions options{dim, 200, 4096, 3};
    const auto        heap_path = (root / "vectors.heap").string(),
               meta_path = (root / "vectors.meta").string();
    auto heap = HeapStorage<T>::create(options, heap_path);
    heap->load_vectors(vectors, first, base);
    const std::span<const T> query(data.data() + 73 * dim, dim);
    SearchConfig             config;
    config.nprobe = 8;
    config.recall_at = 10;
    config.rerank_factor = 16;
    rejects([&] {
      search(*router, lists, pq, *heap, query, config);
    });  // Requires frozen generation.
    heap->checkpoint(meta_path);
    require(lists.size() == base && heap->size() == base &&
                !heap->contains(first + base),
            "insertion tail leaked");
    auto exact_distance = [&](uint32_t id) {
      double distance = 0;
      for (uint32_t d = 0; d < dim; ++d) {
        const double delta =
            double(query[d]) - double(data[(id - first) * dim + d]);
        distance += delta * delta;
      }
      return ExactHit{id, distance};
    };
    std::vector<ExactHit> brute;
    for (uint32_t r = 0; r < base; ++r)
      brute.push_back(exact_distance(first + r));
    std::sort(brute.begin(), brute.end(), nearer);
    brute.resize(config.recall_at);
    const auto all = search(*router, lists, pq, *heap, query, config);
    require(all.hits == brute && all.scanned_codes == base &&
                all.reranked_vectors == base && all.backend_batch_calls == 160,
            "all-list/full-budget differs from exact vector oracle");
    heap.reset();
    auto read_options = options;
    read_options.buffer_frames = 256;
    heap = HeapStorage<T>::reopen(read_options, heap_path, meta_path);
    constexpr size_t                      stride = 64 / sizeof(T);
    alignas(64) std::array<T, 3 * stride> copied_coordinates{};
    const std::array<uint32_t, 3>         duplicates{first, first, first + 1};
    // Streaming loads now place adjacent rows on one page, for every fixture
    // dtype. Neither duplicate IDs nor distinct slots should duplicate reads.
    const auto cold_copy =
        heap->copy_coordinates_batch(duplicates, copied_coordinates, stride);
    require(cold_copy.present == std::vector<uint8_t>({1, 1, 1}) &&
                cold_copy.backend_batch_calls == 1 &&
                cold_copy.heap_pages_read == 1,
            "cold duplicate/same-page batch was not deduplicated");
    require(heap->copy_coordinates_batch(duplicates, copied_coordinates, stride)
                    .heap_pages_read == 0,
            "resident page batch reported physical reads");
    // Preserve explicit two-page coverage using the unchanged individual-append
    // allocator, rather than assuming that streamed IDs occupy distinct pages.
    const auto two_path = (root / "two_page.heap").string();
    const auto two_meta = (root / "two_page.meta").string();
    auto       two_pages = HeapStorage<T>::create(options, two_path);
    two_pages->append(first, std::span<const T>(data).first(dim));
    two_pages->append(first + 1, std::span<const T>(data).subspan(dim, dim));
    two_pages->checkpoint(two_meta);
    two_pages.reset();
    two_pages = HeapStorage<T>::reopen(read_options, two_path, two_meta);
    const auto two_copy = two_pages->copy_coordinates_batch(
        duplicates, copied_coordinates, stride);
    require(two_copy.present == std::vector<uint8_t>({1, 1, 1}) &&
                two_copy.backend_batch_calls == 1 &&
                two_copy.heap_pages_read == 2,
            "cold duplicate/two-page batch was not deduplicated");
    for (size_t i = 0; i < duplicates.size(); ++i)
      for (uint32_t d = 0; d < dim; ++d)
        require(copied_coordinates[i * stride + d] ==
                    data[(duplicates[i] - first) * dim + d],
                "two-page batch changed coordinates");
    require(
        two_pages
                ->copy_coordinates_batch(duplicates, copied_coordinates, stride)
                .heap_pages_read == 0,
        "resident two-page batch reported physical reads");
    const auto reopened = search(*router, lists, pq, *heap, query, config);
    require(reopened.hits == all.hits && reopened.backend_batch_calls == 3,
            "reopen/batch split changed exact results");
    require(reopened.logical_fetch_stages == 1 && reopened.routing_us >= 0 &&
                reopened.adc_us >= 0 && reopened.scan_select_us >= 0 &&
                reopened.fetch_us >= 0 && reopened.rerank_us >= 0,
            "query stage accounting is invalid");
    const auto required_query_bytes =
        estimate_query_workspace(*router, lists, pq, *heap, config);
    require(
        search(*router, lists, pq, *heap, query, config, {required_query_bytes})
                .hits == all.hits,
        "estimated query boundary refused");
    rejects([&] {
      search(*router, lists, pq, *heap, query, config,
             {required_query_bytes - 1});
    });
    require(
        search(*router, lists, pq, *heap, query, config).heap_pages_read == 0,
        "warm all-list query reported physical reads");
    // Limited probes and candidate budget: independent direct-code ADC oracle,
    // followed by direct input-vector L2, not by another invocation of scan().
    config.nprobe = 3;
    config.rerank_factor = 1.6;
    const auto limited = search(*router, lists, pq, *heap, query, config);
    const auto routed = router->search(query, 3);
    std::vector<uint32_t> selected;
    for (auto hit : routed)
      selected.push_back(hit.list_id);
    require(selected == limited.selected_lists,
            "query used different routing breadth/order");
    std::vector<float> table;
    pq.populate_distances(query.data(), table);
    std::vector<PQCandidate> adc;
    for (uint32_t list : selected)
      for (uint32_t id : lists.ids(list))
        adc.push_back(
            {id, pq.distance(codes.data() + (id - first) * chunks, table)});
    const uint64_t scanned = adc.size();
    std::sort(adc.begin(), adc.end(), [](PQCandidate a, PQCandidate b) {
      return a.squared_distance < b.squared_distance ||
             (a.squared_distance == b.squared_distance && a.id < b.id);
    });
    adc.resize(std::min(adc.size(), config.candidate_budget(base)));
    std::vector<ExactHit> limited_brute;
    for (auto candidate : adc)
      limited_brute.push_back(exact_distance(candidate.id));
    std::sort(limited_brute.begin(), limited_brute.end(), nearer);
    limited_brute.resize(std::min<size_t>(10, limited_brute.size()));
    require(limited.hits == limited_brute && limited.scanned_codes == scanned &&
                limited.reranked_vectors == adc.size(),
            "limited query disagrees with ADC/rerank oracle");
    std::array<std::exception_ptr, 4> errors{};
    std::vector<std::thread>          readers;
    for (uint32_t t = 0; t < errors.size(); ++t)
      readers.emplace_back([&, t] {
        try {
          for (uint32_t i = 0; i < 5; ++i)
            require(search(*router, lists, pq, *heap, query, config).hits ==
                        limited.hits,
                    "concurrent query changed results");
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    for (auto& reader : readers)
      reader.join();
    for (auto error : errors)
      if (error)
        std::rethrow_exception(error);
    require(search(*router, lists, pq, *heap, query, config,
                   {64ull * 1024 * 1024, true})
                    .hits.size() <= 10,
            "exact routing query failed");
    rejects(
        [&] { search(*router, lists, pq, *heap, query.first(12), config); });
    rejects([&] { search(*router, lists, pq, *heap, query, config, {1}); });
    config.nprobe = 0;
    rejects([&] { search(*router, lists, pq, *heap, query, config); });
    config.nprobe = 9;
    rejects([&] { search(*router, lists, pq, *heap, query, config); });
    config.nprobe = 8;
    config.rerank_factor = 16;
    if constexpr (std::is_same_v<T, float>) {
      auto bad = std::vector<T>(query.begin(), query.end());
      bad[0] = std::numeric_limits<float>::infinity();
      rejects([&] {
        search(*router, lists, pq, *heap, std::span<const T>(bad), config);
      });
    }
    auto missing =
        HeapStorage<T>::create(options, (root / "missing.heap").string());
    missing->load_vectors(vectors, first, base - 1);
    missing->checkpoint((root / "missing.meta").string());
    rejects([&] { search(*router, lists, pq, *missing, query, config); });
    // Probe an actually empty list; do not fabricate filler IDs or scores.
    const float        far_center = std::is_same_v<T, float>
                                        ? 1000
                                        : (std::is_same_v<T, uint8_t> ? 240 : 120);
    std::vector<float> empty_centroids(2 * dim, far_center);
    for (uint32_t d = 0; d < dim; ++d)
      empty_centroids[d] = 0;
    auto empty_router = CentroidRouter::build(dim, empty_centroids, routing);
    ListBuildOptions empty_options{base, first, 200, chunks};
    empty_options.exact_assignment = true;
    auto empty_lists = PostingLists::build<T>(vectors, codes_path,
                                              *empty_router, empty_options);
    require(empty_lists.ids(1).empty(),
            "empty-list fixture unexpectedly occupied");
    std::array<T, dim> far_query;
    far_query.fill(T(far_center));
    config.nprobe = 1;
    const auto empty = search(*empty_router, empty_lists, pq, *heap,
                              std::span<const T>(far_query), config);
    require(empty.hits.empty() && empty.scanned_codes == 0 &&
                empty.backend_batch_calls == 0 &&
                empty.logical_fetch_stages == 0 && empty.fetch_us == 0 &&
                empty.rerank_us == 0,
            "empty probe fabricated results");
    rejects([&] { search(*router, empty_lists, pq, *heap, query, config); });
    // Fewer selected rows than k must produce a short result, never filler IDs.
    const auto small_lists = PostingLists::build<T>(
        vectors, codes_path, *router, {3, first, 200, chunks});
    config.nprobe = 8;
    const auto small = search(*router, small_lists, pq, *heap, query, config);
    require(small.hits.size() == 3 && small.reranked_vectors == 3,
            "short result fabricated entries");
    // Native self-contained bundle: retrain from the same seeded base, persist
    // every component, reopen without source data/PQ files, and rerank exactly.
    const auto prefix = (root / "native bundle").string();
    NativeIndex<T>::build(prefix, vectors, pivot_path, codes_path, training,
                          options, routing, first);
    auto index = NativeIndex<T>::open(prefix, {256});
    require(index->manifest().source_rows == rows &&
                index->manifest().training.base_count == base &&
                index->manifest().first_id == first &&
                index->manifest().chunks == chunks &&
                index->query(query, config).hits == brute,
            "bundle clean reopen differs from exact oracle");
    const auto open_bytes = index->estimated_open_bytes();
    rejects([&] { NativeIndex<T>::open(prefix, {256, open_bytes - 1}); });
    rejects([&] { NativeIndex<T>::open(prefix, {1}); });
    rejects([&] {
      NativeIndex<T>::build(prefix, vectors, pivot_path, codes_path, training,
                            options, {}, first);
    });
    auto wrong_type = training;
    wrong_type.data_type =
        type == DataType::Float ? DataType::Int8 : DataType::Float;
    const auto refused_prefix = (root / "refused").string();
    rejects([&] {
      NativeIndex<T>::build(refused_prefix, vectors, pivot_path, codes_path,
                            wrong_type, options);
    });
    require(!std::filesystem::exists(refused_prefix + ".manifest"),
            "invalid build claimed prefix");
    auto small_budget = training;
    small_budget.memory_budget_gb = .001;
    rejects([&] {
      NativeIndex<T>::build(refused_prefix, vectors, pivot_path, codes_path,
                            small_budget, options, {}, first);
    });
    require(!std::filesystem::exists(refused_prefix + ".manifest"),
            "workspace refusal wrote manifest");
    auto pca_budget = routing;
    pca_budget.native_work_limit = 1;
    auto optimized_training = training;
    optimized_training.trainer_mode = TrainerMode::LVPlus;
    optimized_training.init = CentroidInit::RandomPartition;
    rejects([&] {
      NativeIndex<T>::build(refused_prefix, vectors, pivot_path, codes_path,
                            optimized_training, options, pca_budget);
    });
    require(!std::filesystem::exists(refused_prefix + ".manifest"),
            "impossible PCA allowance claimed manifest");
    if constexpr (std::is_same_v<T, int8_t>)
      rejects([&] { NativeIndex<uint8_t>::open(prefix); });
    const auto collision_prefix = (root / "collision").string();
    {
      std::ofstream out(collision_prefix + ".lists");
      out << "preserve";
    }
    rejects([&] {
      NativeIndex<T>::build(collision_prefix, vectors, pivot_path, codes_path,
                            training, options, {}, first);
    });
    require(!std::filesystem::exists(collision_prefix + ".manifest"),
            "component collision claimed manifest");
    {
      std::ifstream in(collision_prefix + ".lists");
      std::string   text;
      in >> text;
      require(text == "preserve", "build overwrote existing artifact");
    }
    // Relocate all bundle files; no manifest stores absolute or source paths.
    auto copy_bundle = [&](const std::string& to) {
      for (auto suffix : index_component_suffixes)
        std::filesystem::copy_file(prefix + suffix, to + suffix);
      if (routing.native_modes)
        for (auto suffix : native_router_suffixes)
          std::filesystem::copy_file(prefix + suffix, to + suffix);
      std::filesystem::copy_file(prefix + ".manifest", to + ".manifest");
    };
    std::filesystem::create_directory(root / "relocated");
    const auto relocated = (root / "relocated" / "different stem").string();
    copy_bundle(relocated);
    std::filesystem::rename(pivot_path, pivot_path + ".unavailable");
    std::filesystem::rename(vectors, vectors + ".unavailable");
    std::filesystem::rename(codes_path, codes_path + ".unavailable");
    require(NativeIndex<T>::open(relocated, {256, open_bytes})
                    ->query(query, config)
                    .hits == brute,
            "relocated bundle depends on original inputs/paths");
    for (size_t i = 0; i < index_component_suffixes.size(); ++i) {
      const auto damaged = (root / ("damaged_" + std::to_string(i))).string();
      copy_bundle(damaged);
      std::fstream file(damaged + index_component_suffixes[i],
                        std::ios::binary | std::ios::in | std::ios::out);
      char         byte{};
      file.read(&byte, 1);
      byte ^= 1;
      file.seekp(0);
      file.write(&byte, 1);
      file.close();
      rejects([&] { NativeIndex<T>::open(damaged); });
    }
    if (routing.native_modes)
      for (auto suffix : native_router_suffixes) {
        const auto damaged =
            (root / (std::string("damaged_native") + suffix)).string();
        copy_bundle(damaged);
        std::fstream file(damaged + suffix,
                          std::ios::binary | std::ios::in | std::ios::out);
        char         byte{};
        file.read(&byte, 1);
        byte ^= 1;
        file.seekp(0);
        file.write(&byte, 1);
        file.close();
        rejects([&] { NativeIndex<T>::open(damaged); });
      }
    if (routing.native_modes) {
      const auto missing_mode = (root / "missing_mode_component").string();
      copy_bundle(missing_mode);
      std::filesystem::remove(missing_mode + ".pca");
      rejects([&] { NativeIndex<T>::open(missing_mode); });
      const auto unknown = (root / "unknown_version").string();
      copy_bundle(unknown);
      std::ifstream input(unknown + ".manifest");
      std::string   text((std::istreambuf_iterator<char>(input)), {});
      input.close();
      text.replace(0, std::string("IVFINDEX2").size(), "IVFINDEX99");
      {
        std::ofstream out(unknown + ".manifest");
        out << text;
      }
      rejects([&] { NativeIndex<T>::open(unknown); });
      const auto mismatched_pca = (root / "mismatched_pca_dimension").string();
      copy_bundle(mismatched_pca);
      input.open(mismatched_pca + ".manifest");
      text.assign(std::istreambuf_iterator<char>(input), {});
      input.close();
      const auto begin = text.find("native_routing "),
                 end = text.find('\n', begin), last = text.rfind(' ', end);
      text.replace(last + 1, end - last - 1, "0");
      {
        std::ofstream out(mismatched_pca + ".manifest");
        out << text;
      }
      rejects([&] { NativeIndex<T>::open(mismatched_pca); });
    }
    const auto incomplete = (root / "incomplete").string();
    { std::ofstream out(incomplete + ".manifest"); }
    rejects([&] { NativeIndex<T>::open(incomplete); });
    const auto trailing_manifest = (root / "trailing_manifest").string();
    copy_bundle(trailing_manifest);
    {
      std::ofstream out(trailing_manifest + ".manifest", std::ios::app);
      out << "extra";
    }
    rejects([&] { NativeIndex<T>::open(trailing_manifest); });
    const auto negative_manifest = (root / "negative_manifest").string();
    copy_bundle(negative_manifest);
    {
      std::ifstream in(negative_manifest + ".manifest");
      std::string   content((std::istreambuf_iterator<char>(in)), {});
      const auto    begin = content.find('\n') + 1;
      const auto    dim_start = content.find(' ', begin) + 1,
                 dim_end = content.find(' ', dim_start);
      content.replace(dim_start, dim_end - dim_start, "-1");
      std::ofstream out(negative_manifest + ".manifest");
      out << content;
    }
    rejects([&] { NativeIndex<T>::open(negative_manifest); });
    auto update_digest = [&](const std::string& bundle, const char* suffix) {
      std::ifstream      in(bundle + ".manifest");
      std::ostringstream changed;
      std::string        line;
      const auto         digest = artifact_digest(bundle + suffix);
      while (std::getline(in, line)) {
        if (line.starts_with(std::string(suffix) + " "))
          changed << suffix << ' ' << digest.bytes << ' ' << digest.hash
                  << '\n';
        else
          changed << line << '\n';
      }
      in.close();
      std::ofstream out(bundle + ".manifest");
      out << changed.str();
    };
    const auto mismatched_lists = (root / "mismatched_lists").string();
    copy_bundle(mismatched_lists);
    {
      std::fstream file(mismatched_lists + ".lists",
                        std::ios::binary | std::ios::in | std::ios::out);
      uint32_t     other = type == DataType::Float ? 1 : 0;
      file.seekp(12);
      file.write(reinterpret_cast<const char*>(&other), sizeof(other));
      file.close();
      const auto length =
          std::filesystem::file_size(mismatched_lists + ".lists");
      const auto checksum =
          artifact_hash_prefix(mismatched_lists + ".lists", length - 8);
      file.open(mismatched_lists + ".lists",
                std::ios::binary | std::ios::in | std::ios::out);
      file.seekp(length - 8);
      file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    }
    update_digest(mismatched_lists, ".lists");
    // Valid list checksum and manifest digest, incompatible persisted dtype.
    rejects([&] { NativeIndex<T>::open(mismatched_lists); });
    const auto mismatched_router = (root / "mismatched_router").string();
    copy_bundle(mismatched_router);
    {
      std::ifstream            in(mismatched_router + ".router");
      std::vector<std::string> tokens;
      std::string              token;
      while (in >> token)
        tokens.push_back(token);
      in.close();
      require(tokens.size() == (routing.native_modes ? 35u : 13u),
              "unexpected router descriptor fixture");
      tokens[6] = std::to_string(training.nlist + 1000);
      std::ofstream out(mismatched_router + ".router");
      for (const auto& value : tokens)
        out << value << ' ';
      out << '\n';
    }
    update_digest(mismatched_router, ".router");
    rejects([&] { NativeIndex<T>::open(mismatched_router); });
    // Bounded manifest parsing happens before following any artifact framing.
    const auto huge_manifest = (root / "huge_manifest").string();
    {
      std::ofstream out(huge_manifest + ".manifest");
      out << std::string(16385, 'x');
    }
    rejects([&] { NativeIndex<T>::open(huge_manifest); });
    if constexpr (std::is_same_v<T, float>) {
      const auto bad_vectors = (root / "bad_build.bin").string();
      std::filesystem::copy_file(vectors + ".unavailable", bad_vectors);
      {
        std::fstream file(bad_vectors,
                          std::ios::binary | std::ios::in | std::ios::out);
        const float  invalid = std::numeric_limits<float>::infinity();
        file.seekp(8);
        file.write(reinterpret_cast<const char*>(&invalid), sizeof(invalid));
      }
      const auto failed = (root / "failed_build").string();
      rejects([&] {
        NativeIndex<T>::build(failed, bad_vectors, pivot_path + ".unavailable",
                              codes_path + ".unavailable", training, options,
                              {}, first);
      });
      require(std::filesystem::file_size(failed + ".manifest") == 0,
              "failed training published a complete manifest");
      rejects([&] { NativeIndex<T>::open(failed); });
    }
  }
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 2 || argc == 3, "expected fixture directory [native mode]");
    if (argc == 3)
      fixture_mode = parse_router_mode(argv[2]);
    const auto root =
        std::filesystem::path(argv[1]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    exercise<float>(root / "float", DataType::Float);
    exercise<int8_t>(root / "int8", DataType::Int8);
    exercise<uint8_t>(root / "uint8", DataType::UInt8);
    std::cout << "IVF query composition/batched reranking tests passed\n";
    return 0;
  } catch (const diskann::ANNException& error) {
    std::cerr << error.message() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
