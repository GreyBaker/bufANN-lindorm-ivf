#include "bufann/ivf_search.h"
#include "bufann/ivf_numeric_internal.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace diskann::ivf {
  namespace {
    void check(bool ok, const char* message) {
      if (!ok)
        throw std::invalid_argument(message);
    }
    void add_product(uint64_t& bytes, uint64_t count, uint64_t stride) {
      check(!stride || count <= (UINT64_MAX - bytes) / stride,
            "IVF query workspace overflow");
      bytes += count * stride;
    }
    using Clock = std::chrono::steady_clock;
    double elapsed_us(Clock::time_point start) {
      return std::chrono::duration<double, std::micro>(Clock::now() - start)
          .count();
    }
    template<typename T>
    size_t aligned_row_stride(uint32_t aligned_dim) {
      return ((uint64_t(aligned_dim) * sizeof(T) + 63) / 64) * 64 / sizeof(T);
    }
    void accumulate_fetch(QueryResult& result, const BatchCopyResult& copied) {
      result.backend_batch_calls += copied.backend_batch_calls;
      result.heap_pages_read += copied.heap_pages_read;
      result.fetch_adapter_mutex_wait_ns += copied.adapter_mutex_wait_ns;
      result.fetch_backend_pin_ns += copied.backend_pin_wall_ns;
      result.fetch_pin_admission_wait_ns += copied.pin_admission_wait_ns;
    }
  }  // namespace

  template<typename T>
  uint64_t estimate_query_workspace(const CentroidRouter& router,
                                    const PostingLists&   lists,
                                    const PQCodebook<T>&  codebook,
                                    const HeapStorage<T>& heap,
                                    const SearchConfig&   config,
                                    bool                  exact_routing) {
    config.validate(router.nlist());
    check(codebook.dim() == router.dim() && heap.dim() == router.dim() &&
              lists.dim() == router.dim() && lists.nlist() == router.nlist() &&
              lists.chunks() == codebook.chunks() &&
              lists.data_type() ==
                  (std::is_same_v<T, float>
                       ? DataType::Float
                       : (std::is_same_v<T, int8_t> ? DataType::Int8
                                                    : DataType::UInt8)),
          "Incompatible IVF query components");
    check(!heap.failed(), "Cannot query a failed IVF heap build");
    check(heap.read_only(),
          "IVF query requires a clean checkpoint/read-only heap");
    const auto budget = config.candidate_budget(lists.size());
    check(budget > 0 && budget <= UINT32_MAX,
          "Invalid IVF query candidate budget");
    // Retained PQ candidates/exact hits plus scan heap growth, LUT, routing
    // scratch/visited allowance and one bounded, aligned coordinate batch.
    const size_t stride = aligned_row_stride<T>(heap.aligned_dim());
    const size_t batch = std::min<size_t>(budget, heap.copy_batch_limit());
    uint64_t     bytes = 65536;
    add_product(bytes, budget, 96);
    add_product(bytes, codebook.chunks(), 256 * sizeof(float));
    add_product(bytes, config.nprobe, 32);
    add_product(bytes, router.dim(), sizeof(T) + sizeof(float));
    add_product(bytes, router.nlist(), 2);
    // Allow any already-built query mode/breadth, including exact overrides.
    (void) exact_routing;
    add_product(bytes, router.nlist(), router.has_native_modes() ? 128 : 64);
    if (router.has_native_modes())
      add_product(bytes, router.dim(), sizeof(float));
    add_product(bytes, batch, stride * sizeof(T) + sizeof(uint32_t) + 1);
    check(bytes <= SIZE_MAX, "IVF query workspace exceeds size_t");
    return bytes;
  }

  template<typename T>
  QueryResult search(const CentroidRouter& router, const PostingLists& lists,
                     const PQCodebook<T>& codebook, HeapStorage<T>& heap,
                     std::span<const T> query, const SearchConfig& config,
                     const QueryLimits& limits) {
    check(query.size() == router.dim(), "Invalid IVF query dimension");
    for (T coordinate : query)
      check(std::isfinite(double(coordinate)), "Nonfinite IVF query");
    const auto bytes = estimate_query_workspace(router, lists, codebook, heap,
                                                config, limits.exact_routing);
    check(bytes <= limits.workspace_bytes,
          "IVF query exceeds workspace allowance");
    const auto   budget = config.candidate_budget(lists.size());
    const size_t stride = aligned_row_stride<T>(heap.aligned_dim());
    const size_t batch = std::min<size_t>(budget, heap.copy_batch_limit());
    QueryResult  result;
    auto         start = Clock::now();
    const auto   routed =
        router.search(query, config.nprobe, limits.exact_routing,
                      &result.router_stats, limits.routing);
    for (auto hit : routed)
      result.selected_lists.push_back(hit.list_id);
    result.routing_us = elapsed_us(start);
    start = Clock::now();
    std::vector<float> distances;
    codebook.populate_distances(query.data(), distances);
    result.adc_us = elapsed_us(start);
    start = Clock::now();
    auto scanned =
        lists.scan(result.selected_lists, distances,
                   static_cast<uint32_t>(budget), limits.workspace_bytes);
    result.scanned_codes = scanned.scanned_codes;
    result.scan_select_us = elapsed_us(start);
    if (scanned.candidates.empty())
      return result;
    result.logical_fetch_stages = 1;
    const size_t allocation_bytes = batch * stride * sizeof(T);
    std::unique_ptr<void, decltype(&std::free)> scratch(
        std::aligned_alloc(64, allocation_bytes), &std::free);
    if (!scratch)
      throw std::bad_alloc();
    std::vector<uint32_t> ids;
    ids.reserve(batch);
    result.hits.reserve(scanned.candidates.size());
    for (size_t begin = 0; begin < scanned.candidates.size(); begin += batch) {
      const size_t count = std::min(batch, scanned.candidates.size() - begin);
      ids.clear();
      for (size_t i = 0; i < count; ++i)
        ids.push_back(scanned.candidates[begin + i].id);
      start = Clock::now();
      auto copied = heap.copy_coordinates_batch(
          ids, {static_cast<T*>(scratch.get()), count * stride}, stride);
      accumulate_fetch(result, copied);
      result.fetch_us += elapsed_us(start);
      start = Clock::now();
      for (size_t i = 0; i < count; ++i) {
        check(copied.present[i], "IVF posting references a missing heap ID");
        const auto* vector = static_cast<const T*>(scratch.get()) + i * stride;
        const double distance = detail::squared_l2_double_unchecked(
            query.data(), vector, router.dim());
        check(std::isfinite(distance), "Nonfinite IVF exact distance");
        result.hits.push_back({ids[i], distance});
      }
      result.rerank_us += elapsed_us(start);
    }
    result.reranked_vectors = result.hits.size();
    start = Clock::now();
    std::sort(result.hits.begin(), result.hits.end(),
              detail::candidate_nearer<ExactHit>);
    if (result.hits.size() > config.recall_at)
      result.hits.resize(config.recall_at);
    result.rerank_us += elapsed_us(start);
    return result;
  }

  template QueryResult search<float>(const CentroidRouter&, const PostingLists&,
                                     const PQCodebook<float>&,
                                     HeapStorage<float>&,
                                     std::span<const float>,
                                     const SearchConfig&, const QueryLimits&);
  template QueryResult search<int8_t>(const CentroidRouter&,
                                      const PostingLists&,
                                      const PQCodebook<int8_t>&,
                                      HeapStorage<int8_t>&,
                                      std::span<const int8_t>,
                                      const SearchConfig&, const QueryLimits&);
  template QueryResult search<uint8_t>(const CentroidRouter&,
                                       const PostingLists&,
                                       const PQCodebook<uint8_t>&,
                                       HeapStorage<uint8_t>&,
                                       std::span<const uint8_t>,
                                       const SearchConfig&, const QueryLimits&);
  template uint64_t    estimate_query_workspace<float>(const CentroidRouter&,
                                                       const PostingLists&,
                                                       const PQCodebook<float>&,
                                                       const HeapStorage<float>&,
                                                       const SearchConfig&, bool);
  template uint64_t    estimate_query_workspace<int8_t>(const CentroidRouter&,
                                                        const PostingLists&,
                                                        const PQCodebook<int8_t>&,
                                                        const HeapStorage<int8_t>&,
                                                        const SearchConfig&, bool);
  template uint64_t    estimate_query_workspace<uint8_t>(
      const CentroidRouter&, const PostingLists&, const PQCodebook<uint8_t>&,
      const HeapStorage<uint8_t>&, const SearchConfig&, bool);

  template<typename T>
  uint64_t estimate_mutable_query_workspace(const CentroidRouter&      router,
                                            const MutablePostingLists& lists,
                                            const PQCodebook<T>&       codebook,
                                            const HeapStorage<T>&      heap,
                                            const SearchConfig&        config,
                                            bool exact_routing) {
    config.validate(router.nlist());
    check(codebook.dim() == router.dim() && heap.dim() == router.dim() &&
              lists.nlist() == router.nlist() &&
              lists.chunks() == codebook.chunks() &&
              lists.id_capacity() == heap.id_capacity(),
          "Incompatible mutable IVF query components");
    check(heap.mutable_mode() && !heap.failed() && !heap.read_only(),
          "Mutable IVF query requires a healthy mutable heap session");
    // Capacity, not a racing live-count snapshot: concurrent inserts cannot
    // exceed the preflighted scan/rerank workspace. Empty live sets are valid.
    const auto budget =
        config.candidate_budget(lists.id_capacity() - lists.first_id());
    check(budget > 0 && budget <= UINT32_MAX,
          "Invalid mutable IVF query candidate budget");
    const size_t stride = aligned_row_stride<T>(heap.aligned_dim());
    uint64_t bytes = 65536;
    add_product(bytes, budget,
                128);  // Scan/output/filtered candidates, ID validation, hits.
    add_product(bytes, codebook.chunks(), 256 * sizeof(float));
    add_product(bytes, config.nprobe, 32);
    add_product(bytes, router.dim(), sizeof(T) + sizeof(float));
    add_product(bytes, router.nlist(), 2);
    (void) exact_routing;
    add_product(bytes, router.nlist(), router.has_native_modes() ? 128 : 64);
    if (router.has_native_modes())
      add_product(bytes, router.dim(), sizeof(float));
    add_product(bytes, std::min<size_t>(budget, heap.copy_batch_limit()),
                stride * sizeof(T) + 5);
    check(bytes <= SIZE_MAX, "Mutable IVF query workspace exceeds size_t");
    return bytes;
  }

  template<typename T>
  QueryResult rerank_mutable_candidates(
      const CentroidRouter& router, const MutablePostingLists& lists,
      const PQCodebook<T>& codebook, HeapStorage<T>& heap,
      std::span<const T> query, std::span<const PQCandidate> candidates,
      const SearchConfig& config, const QueryLimits& limits) {
    check(query.size() == router.dim(), "Invalid mutable IVF query dimension");
    for (T coordinate : query)
      check(std::isfinite(double(coordinate)), "Nonfinite mutable IVF query");
    check(estimate_mutable_query_workspace(router, lists, codebook, heap,
                                           config, limits.exact_routing) <=
              limits.workspace_bytes,
          "Mutable IVF query exceeds workspace allowance");
    const auto budget =
        config.candidate_budget(lists.id_capacity() - lists.first_id());
    check(candidates.size() <= budget,
          "Mutable IVF candidate set exceeds declared budget");
    std::vector<uint32_t> validation;
    validation.reserve(candidates.size());
    for (auto hit : candidates) {
      check(hit.id >= lists.first_id() && hit.id < lists.id_capacity() &&
                std::isfinite(hit.squared_distance) &&
                hit.squared_distance >= 0,
            "Invalid mutable IVF candidate");
      validation.push_back(hit.id);
    }
    std::sort(validation.begin(), validation.end());
    check(std::adjacent_find(validation.begin(), validation.end()) ==
              validation.end(),
          "Duplicate mutable IVF candidate ID");
    QueryResult              result;
    std::vector<PQCandidate> active;
    active.reserve(candidates.size());
    for (auto hit : candidates) {
      if (lists.contains(hit.id))
        active.push_back(hit);
      else
        ++result.stale_candidates;
    }
    if (active.empty())
      return result;
    result.logical_fetch_stages = 1;
    const size_t stride = aligned_row_stride<T>(heap.aligned_dim());
    const size_t batch =
        std::min<size_t>(active.size(), heap.copy_batch_limit());
    std::unique_ptr<void, decltype(&std::free)> scratch(
        std::aligned_alloc(64, batch * stride * sizeof(T)), &std::free);
    if (!scratch)
      throw std::bad_alloc();
    std::vector<uint32_t> ids;
    ids.reserve(batch);
    result.hits.reserve(active.size());
    for (size_t begin = 0; begin < active.size(); begin += batch) {
      const size_t count = std::min(batch, active.size() - begin);
      ids.clear();
      for (size_t i = 0; i < count; ++i)
        ids.push_back(active[begin + i].id);
      auto       start = Clock::now();
      const auto copied = heap.copy_coordinates_batch(
          ids, {static_cast<T*>(scratch.get()), count * stride}, stride);
      accumulate_fetch(result, copied);
      result.fetch_us += elapsed_us(start);
      start = Clock::now();
      for (size_t i = 0; i < count; ++i) {
        if (!lists.contains(ids[i])) {
          ++result.stale_candidates;
          continue;
        }
        check(copied.present[i],
              "Live mutable IVF posting references a missing heap ID");
        const auto* vector = static_cast<const T*>(scratch.get()) + i * stride;
        const double distance = detail::squared_l2_double_unchecked(
            query.data(), vector, router.dim());
        check(std::isfinite(distance), "Nonfinite mutable IVF exact distance");
        result.hits.push_back({ids[i], distance});
      }
      result.rerank_us += elapsed_us(start);
    }
    result.reranked_vectors = result.hits.size();
    auto start = Clock::now();
    std::sort(result.hits.begin(), result.hits.end(),
              detail::candidate_nearer<ExactHit>);
    std::erase_if(result.hits, [&](auto hit) {
      if (lists.contains(hit.id))
        return false;
      ++result.stale_candidates;
      return true;
    });
    if (result.hits.size() > config.recall_at)
      result.hits.resize(config.recall_at);
    result.rerank_us += elapsed_us(start);
    return result;
  }

  template<typename T>
  QueryResult search_mutable(const CentroidRouter&      router,
                             const MutablePostingLists& lists,
                             const PQCodebook<T>&       codebook,
                             HeapStorage<T>& heap, std::span<const T> query,
                             const SearchConfig& config,
                             const QueryLimits&  limits) {
    check(query.size() == router.dim(), "Invalid mutable IVF query dimension");
    for (T coordinate : query)
      check(std::isfinite(double(coordinate)), "Nonfinite mutable IVF query");
    check(estimate_mutable_query_workspace(router, lists, codebook, heap,
                                           config, limits.exact_routing) <=
              limits.workspace_bytes,
          "Mutable IVF query exceeds workspace allowance");
    auto         start = Clock::now();
    RoutingStats routing_stats;
    const auto   routed =
        router.search(query, config.nprobe, limits.exact_routing,
                      &routing_stats, limits.routing);
    std::vector<uint32_t> selected;
    selected.reserve(routed.size());
    for (auto hit : routed)
      selected.push_back(hit.list_id);
    const double routing = elapsed_us(start);
    start = Clock::now();
    std::vector<float> distances;
    codebook.populate_distances(query.data(), distances);
    const double adc = elapsed_us(start);
    start = Clock::now();
    const auto scanned = lists.scan(
        selected, distances,
        static_cast<uint32_t>(
            config.candidate_budget(lists.id_capacity() - lists.first_id())),
        limits.workspace_bytes);
    const double scan = elapsed_us(start);
    auto         result =
        rerank_mutable_candidates(router, lists, codebook, heap, query,
                                  scanned.candidates, config, limits);
    result.selected_lists = std::move(selected);
    result.scanned_codes = scanned.scanned_codes;
    result.router_stats = routing_stats;
    result.routing_us = routing;
    result.adc_us = adc;
    result.scan_select_us = scan;
    return result;
  }

#define IVF_MUTABLE_SEARCH_INSTANTIATE(T)                                      \
  template uint64_t estimate_mutable_query_workspace<T>(                       \
      const CentroidRouter&, const MutablePostingLists&, const PQCodebook<T>&, \
      const HeapStorage<T>&, const SearchConfig&, bool);                       \
  template QueryResult search_mutable<T>(                                      \
      const CentroidRouter&, const MutablePostingLists&, const PQCodebook<T>&, \
      HeapStorage<T>&, std::span<const T>, const SearchConfig&,                \
      const QueryLimits&);                                                     \
  template QueryResult rerank_mutable_candidates<T>(                           \
      const CentroidRouter&, const MutablePostingLists&, const PQCodebook<T>&, \
      HeapStorage<T>&, std::span<const T>, std::span<const PQCandidate>,       \
      const SearchConfig&, const QueryLimits&);
  IVF_MUTABLE_SEARCH_INSTANTIATE(float)
  IVF_MUTABLE_SEARCH_INSTANTIATE(int8_t)
  IVF_MUTABLE_SEARCH_INSTANTIATE(uint8_t)
#undef IVF_MUTABLE_SEARCH_INSTANTIATE
}  // namespace diskann::ivf
