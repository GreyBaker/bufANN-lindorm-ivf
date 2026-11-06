#include "bufann/ivf_router.h"
#include "bufann/ivf_artifacts.h"
#include "bufann/ivf_io.h"
#include "bufann/ivf_numeric_internal.h"
#include "bufann/ivf_patched_graph.h"
#include "bufann/ivf_pca.h"
#include "ivf_centroids_internal.h"

#include "ivf_faiss_internal.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>

namespace diskann::ivf {
  namespace {
    struct TrainingSearchPolicy {
      uint32_t candidates;
      uint32_t breadth;
    };
    // Returned candidates are independent of traversal breadth for
    // optimized/count trainers. Legacy controls retain breadth-sized double
    // rescoring.
    TrainingSearchPolicy training_search_policy(const BuildConfig& config) {
      const uint32_t candidates =
          trainer_optimized_search(config.trainer_mode)
              ? uint32_t(std::min<long double>(
                    config.nlist, std::ceil(config.training_rerank_factor)))
          : trainer_count_candidates(config.trainer_mode)
              ? 1
              : std::min(config.nlist, config.training_breadth);
      return {candidates,
              std::min(config.nlist,
                       std::max(config.training_breadth, candidates))};
    }
    void check(bool ok, const char* message) {
      if (!ok)
        throw std::runtime_error(message);
    }
    uint64_t estimate(uint32_t dim, uint32_t count,
                      const RouterOptions& options) {
       // External centroids + FAISS Flat storage, levels/offsets/adjacency,
       // construction locks, one caller-thread cached visited table and
       // allocator allowance. Additional workers admit their own scratch.
      const uint64_t per_node =
          uint64_t(dim) * 8 + uint64_t(options.degree) * 16 + 256;
      const uint64_t fixed = 4ull * 1024 * 1024;
      check(count <= (UINT64_MAX - fixed) / per_node,
            "Router memory estimate overflow");
      uint64_t result = uint64_t(count) * per_node + fixed;
      if (options.native_modes) {
        CentroidPCA::preflight_work(dim, count, options.native_work_limit);
        const auto graph =
            result + uint64_t(count) * (uint64_t(options.degree) * 32 + 512);
        const auto pca = CentroidPCA::estimate_bytes(dim, count);
        check(graph <= (UINT64_MAX - result) / 2,
              "Patched router estimate overflow");
        result += graph * 2;
        check(pca <= UINT64_MAX - result, "Native PCA estimate overflow");
        result += pca;
      }
      return result;
    }
    void validate(uint32_t dim, size_t cells, const RouterOptions& options) {
      check(
          dim > 0 && dim <= INT32_MAX && cells > 0 && cells % dim == 0 && cells / dim <= INT32_MAX,
          "Invalid centroid router shape");
      check(
          options.degree >= 2 && options.degree <= 64 &&
              options.construction_breadth >= options.degree &&
              options.construction_breadth <= INT32_MAX &&
              options.query_breadth > 0 && options.query_breadth <= INT32_MAX &&
              options.insert_breadth > 0 && options.insert_breadth <= INT32_MAX,
          "Invalid HNSW settings");
      check(options.mode != RouterMode::Default &&
                (options.native_modes || options.mode == RouterMode::Faiss) &&
                !options.refine_reverse_edges && options.refine_breadth > 0 &&
                options.refine_breadth <= INT32_MAX &&
                options.native_work_limit > 0 &&
                std::isfinite(options.pca_variance) &&
                options.pca_variance > 0 && options.pca_variance <= 1 &&
                std::isfinite(options.router_rerank_factor) &&
                options.router_rerank_factor >= 1,
            "Invalid native router settings");
      check(estimate(dim, static_cast<uint32_t>(cells / dim), options) <=
                options.memory_budget_bytes,
            "Centroid router exceeds preflight memory allowance");
    }
    void validate_values(std::span<const float> values, uint32_t dim) {
      // HNSW uses float squared distances; finite inputs alone do not prevent
      // overflow. Leave a factor-of-two accumulation margin for SIMD rounding.
      const double bound =
          std::sqrt(double(std::numeric_limits<float>::max()) / (8.0 * dim));
      for (float value : values)
        check(std::isfinite(value) && std::abs(double(value)) <= bound,
              "Non-finite or excessive centroid-routing coordinate");
    }
    bool nearer(const CentroidHit& a, const CentroidHit& b) {
      return a.squared_distance != b.squared_distance
                 ? a.squared_distance < b.squared_distance
                 : a.list_id < b.list_id;
    }
    // Bounded local framing validates FAISS adjacency before following edges.
    void validate_graph(const std::string& path, uint32_t dim,
                        std::span<const float> centroids,
                        const RouterOptions&   options) {
      (void) faiss_detail::load(path, dim, centroids,
          {options.degree, options.construction_breadth, options.seed,
           options.memory_budget_bytes, options.native_work_limit});
    }
  }  // namespace

  struct CentroidRouter::Impl {
    explicit Impl(uint32_t) {
    }
    std::unique_ptr<faiss_detail::Index> graph;
    std::unique_ptr<PatchedCentroidGraph>            native, refined;
    std::unique_ptr<CentroidPCA>                     pca;
  };
  RouterMode parse_router_mode(const std::string& value) {
    for (auto mode :
         {RouterMode::Faiss, RouterMode::Exact, RouterMode::FaissFull,
          RouterMode::Refined, RouterMode::PCA, RouterMode::PCARefined})
      if (value == router_mode_name(mode))
        return mode;
    throw std::invalid_argument("Unknown centroid router mode");
  }
  const char* router_mode_name(RouterMode mode) {
    switch (mode) {
      case RouterMode::Faiss:
        return "faiss";
      case RouterMode::Exact:
        return "exact";
      case RouterMode::FaissFull:
        return "faiss-full";
      case RouterMode::Refined:
        return "refined";
      case RouterMode::PCA:
        return "pca";
      case RouterMode::PCARefined:
        return "pca-refined";
      default:
        return "default";
    }
  }
  uint32_t CentroidRouter::projected_dim() const {
    return _impl->pca ? _impl->pca->retained_dim() : _dim;
  }
  double CentroidRouter::retained_variance() const {
    return _impl->pca ? _impl->pca->retained_variance() : 1;
  }
  bool CentroidRouter::pca_degenerate() const {
    return _impl->pca && _impl->pca->degenerate();
  }
  uint32_t CentroidRouter::unreachable_before(bool refined) const {
    return _impl->native ? (refined ? _impl->refined : _impl->native)
                               ->diagnostics()
                               .unreachable_before
                         : 0;
  }
  uint32_t CentroidRouter::reinserted_nodes(bool refined) const {
    return _impl->native ? (refined ? _impl->refined : _impl->native)
                               ->diagnostics()
                               .reinserted_nodes
                         : 0;
  }
  uint32_t CentroidRouter::unreachable_after(bool refined) const {
    return _impl->native ? (refined ? _impl->refined : _impl->native)
                               ->diagnostics()
                               .unreachable_after
                         : 0;
  }
  uint32_t CentroidRouter::zero_indegree(bool refined) const {
    return _impl->native ? (refined ? _impl->refined : _impl->native)
                               ->diagnostics()
                               .zero_indegree
                         : 0;
  }
  CentroidRouter::CentroidRouter(uint32_t dim, std::span<const float> centroids,
                                 const RouterOptions& options)
      : _dim(dim), _options(options) {
    static_assert(std::endian::native == std::endian::little &&
                  sizeof(size_t) == 8 && sizeof(faiss::idx_t) == 8 &&
                  sizeof(int) == 4 && sizeof(float) == 4);
    validate(dim, centroids.size(), options);
    validate_values(centroids, dim);
    _centroids.assign(centroids.begin(), centroids.end());
    _impl = std::make_unique<Impl>(dim);
  }
  CentroidRouter::~CentroidRouter() = default;
  uint64_t CentroidRouter::estimated_bytes() const {
    return estimate(_dim, nlist(), _options);
  }
  uint64_t CentroidRouter::estimate_build_bytes(uint32_t dim, uint32_t count,
                                                const RouterOptions& options) {
    const uint64_t cells = uint64_t(dim) * count;
    check(cells <= SIZE_MAX / sizeof(float), "Router centroid shape overflow");
    validate(dim, static_cast<size_t>(cells), options);
    return estimate(dim, count, options);
  }

  std::unique_ptr<CentroidRouter> CentroidRouter::build(
      uint32_t dim, std::span<const float> centroids,
      const RouterOptions& options) {
    auto router = std::unique_ptr<CentroidRouter>(
        new CentroidRouter(dim, centroids, options));
    router->_impl->graph = faiss_detail::build(dim, centroids,
        {options.degree, options.construction_breadth, options.seed,
         options.memory_budget_bytes, options.native_work_limit});
    if (options.native_modes) {
      PatchedGraphOptions g{options.degree, options.construction_breadth,
                            options.seed, options.memory_budget_bytes,
                            options.native_work_limit};
      router->_impl->native = PatchedCentroidGraph::build(dim, centroids, g);
      router->_impl->refined = PatchedCentroidGraph::build(dim, centroids, g);
      router->_impl->refined->refine(options.refine_breadth);
      router->_impl->pca = std::make_unique<CentroidPCA>(CentroidPCA::fit(
          dim, centroids, options.pca_variance, options.memory_budget_bytes,
          options.native_work_limit));
      router->_impl->native->configure_projection(router->projected_dim());
      router->_impl->refined->configure_projection(router->projected_dim());
    }
    return router;
  }
  std::vector<CentroidHit> CentroidRouter::route(std::span<const float> query,
                                                 uint32_t               count,
                                                 uint32_t breadth, bool exact,
                                                 RoutingStats*  stats,
                                                 RoutingRequest request) const {
    check(query.size() == _dim && count > 0 && count <= nlist(),
          "Invalid centroid query/probe count");
    validate_values(query, _dim);
    using Clock = std::chrono::steady_clock;
    auto us = [](auto start) {
      return std::chrono::duration<double, std::micro>(Clock::now() - start)
          .count();
    };
    if (stats)
      *stats = {};
    const auto mode =
        request.mode == RouterMode::Default ? _options.mode : request.mode;
    check(mode >= RouterMode::Faiss && mode <= RouterMode::PCARefined,
          "Invalid centroid router mode");
    if (request.breadth)
      breadth = request.breadth;
    check(breadth > 0 && breadth <= INT32_MAX,
          "Invalid centroid router breadth");
    const bool native =
        mode != RouterMode::Faiss && mode != RouterMode::Exact;
    check(!native || _options.native_modes,
          "Requested native router mode was not built");
    const double factor = request.rerank_factor == 0
                              ? _options.router_rerank_factor
                              : request.rerank_factor;
    check(std::isfinite(factor) && factor >= 1,
          "Invalid centroid rerank factor");
    auto distance = [&](uint32_t id) {
      return detail::squared_l2_double_unchecked(
          query.data(), _centroids.data() + size_t(id) * _dim, _dim);
    };
    std::vector<CentroidHit> hits;
    if (exact || mode == RouterMode::Exact || count == nlist()) {
      hits.reserve(nlist());
      for (uint32_t id = 0; id < nlist(); ++id)
        hits.push_back({id, distance(id)});
      if (stats) {
        stats->distance_evaluations = nlist();
        stats->distance_count_available = true;
      }
    } else if (native) {
      const bool compressed =
          mode == RouterMode::PCA || mode == RouterMode::PCARefined;
      const bool refined =
          mode == RouterMode::Refined || mode == RouterMode::PCARefined;
      uint32_t candidates = count;
      if (compressed)
        candidates = uint32_t(std::min<long double>(
            nlist(), std::ceil(static_cast<long double>(count) * factor)));
      else
        candidates = std::min(
            nlist(),
            std::max(count,
                     breadth));  // Explicit optimized oversampling/rescoring.
      std::vector<float> projected;
      auto               start = Clock::now();
      if (compressed)
        projected = _impl->pca->project(query);
      if (stats)
        stats->projection_us = compressed ? us(start) : 0;
      start = Clock::now();
      uint64_t    computations = 0;
      const auto& graph = refined ? _impl->refined : _impl->native;
      auto        found = graph->search(
          compressed ? std::span<const float>(projected) : query, candidates,
          std::max(breadth, candidates), {},
          compressed ? _impl->pca->projected() : std::span<const float>{},
          compressed ? projected_dim() : 0, &computations);
      if (stats) {
        stats->graph_us = us(start);
        stats->distance_evaluations = computations;
        stats->distance_count_available = true;
      }
      start = Clock::now();
      for (auto h : found)
        hits.push_back({h.id, distance(h.id)});
      check(hits.size() >= count,
            "Patched HNSW did not reach enough centroids");
      if (stats) {
        stats->centroid_rerank_us = us(start);
        stats->distance_evaluations += found.size();
      }
    } else {
      // Unmodified FAISS HNSW::search with independent per-call efSearch.
      // Return exactly count candidates; full-coordinate output rescoring is
      // timed separately and is not labeled raw FAISS traversal time.
      auto candidates = faiss_detail::search(*_impl->graph, query, count, breadth,
          {}, {}, 0, nullptr, false, stats ? &stats->graph_us : nullptr);
      check(candidates.size() >= count,
              "HNSW did not reach enough distinct centroids");
      hits.reserve(candidates.size());
      const auto rerank = Clock::now();
      for (auto candidate : candidates) {
        const auto id = candidate.id;
        check(id < nlist(), "Invalid HNSW centroid label");
        hits.push_back(
            {static_cast<uint32_t>(id), distance(static_cast<uint32_t>(id))});
      }
      if (stats) stats->centroid_rerank_us = us(rerank);
    }
    const auto sorting = Clock::now();
    std::partial_sort(hits.begin(), hits.begin() + count, hits.end(), nearer);
    if (stats && !exact && mode != RouterMode::Exact &&
        count != nlist())
      stats->centroid_rerank_us += us(sorting);
    hits.resize(count);
    return hits;
  }
  std::vector<CentroidHit> CentroidRouter::search(
      std::span<const float> query, uint32_t nprobe, bool exact,
      RoutingStats* stats, RoutingRequest request) const {
    return route(query, nprobe, _options.query_breadth, exact, stats, request);
  }
  uint32_t CentroidRouter::assign(std::span<const float> vector,
                                  bool                   exact) const {
    return route(vector, 1, _options.insert_breadth, exact).front().list_id;
  }

  void CentroidRouter::save(const std::string& prefix) const {
    check(!prefix.empty(), "Empty router prefix");
    const auto vectors = prefix + ".centroids.bin", graph = prefix + ".hnsw",
               descriptor = prefix + ".router";
    // Check all collisions before creating anything. Exclusive claims also
    // protect against competing creators; partial artifacts remain on failure.
    std::vector<std::string> paths{vectors, graph, descriptor};
    if (_options.native_modes)
      for (auto suffix : {".patched.hnsw", ".refined.hnsw", ".pca"})
        paths.push_back(prefix + suffix);
    for (const auto& path : paths)
      check(!std::filesystem::exists(std::filesystem::symlink_status(path)),
            "Router artifact already exists");
    claim_artifact(vectors);
    claim_artifact(graph);
    claim_artifact(descriptor);
    std::ofstream  output(vectors, std::ios::binary);
    const uint32_t header[]{nlist(), _dim};
    output.write(reinterpret_cast<const char*>(header), sizeof(header));
    output.write(reinterpret_cast<const char*>(_centroids.data()),
                 _centroids.size() * sizeof(float));
    output.close();
    check(bool(output), "Centroid artifact write failed");
    faiss_detail::save(*_impl->graph, graph);
    validate_graph(graph, _dim, _centroids, _options);
    if (_options.native_modes) {
      claim_artifact(prefix + ".patched.hnsw");
      claim_artifact(prefix + ".refined.hnsw");
      claim_artifact(prefix + ".pca");
      _impl->native->save(prefix + ".patched.hnsw");
      _impl->refined->save(prefix + ".refined.hnsw");
      validate_graph(prefix + ".patched.hnsw", _dim, _centroids, _options);
      validate_graph(prefix + ".refined.hnsw", _dim, _centroids, _options);
      std::ofstream pca(prefix + ".pca", std::ios::binary);
      _impl->pca->write(pca);
      pca.close();
      check(bool(pca), "PCA artifact write failed");
    }
    std::ofstream meta(descriptor, std::ios::binary);
    meta.imbue(std::locale::classic());
    meta << std::setprecision(17)
         << (_options.native_modes ? "IVFROUTER4" : "IVFROUTER3") << ' '
         << router_backend_id << ' ' << _dim << ' ' << nlist() << ' '
         << _options.degree << ' ' << _options.construction_breadth << ' '
         << _options.query_breadth << ' ' << _options.insert_breadth << ' '
         << _options.seed << ' ' << std::filesystem::file_size(vectors) << ' '
         << artifact_hash_file(vectors) << ' '
         << std::filesystem::file_size(graph) << ' '
         << artifact_hash_file(graph) << '\n';
    if (_options.native_modes)
      meta << optimized_router_backend_id << ' ' << router_mode_name(_options.mode) << ' ' << _options.refine_breadth
           << ' ' << _options.refine_reverse_edges << ' '
           << _options.pca_variance << ' ' << _options.router_rerank_factor
           << ' ' << _options.native_work_limit << '\n'
           << std::filesystem::file_size(prefix + ".patched.hnsw") << ' '
           << artifact_hash_file(prefix + ".patched.hnsw") << ' '
           << std::filesystem::file_size(prefix + ".refined.hnsw") << ' '
           << artifact_hash_file(prefix + ".refined.hnsw") << ' '
           << std::filesystem::file_size(prefix + ".pca") << ' '
           << artifact_hash_file(prefix + ".pca") << '\n';
    if (_options.native_modes)
      for (const auto* g : {_impl->native.get(), _impl->refined.get()}) {
        const auto s = g->diagnostics();
        meta << s.unreachable_before << ' ' << s.unreachable_after << ' '
             << s.zero_indegree << ' ' << s.reinserted_nodes << ' '
             << s.refined_nodes << '\n';
      }
    meta.close();
    check(bool(meta), "Router descriptor write failed");
  }
  std::unique_ptr<CentroidRouter> CentroidRouter::load(
      const std::string& prefix, uint64_t budget) {
    const auto vectors = prefix + ".centroids.bin", graph = prefix + ".hnsw";
    check(std::filesystem::file_size(prefix + ".router") <= 4096,
          "Router descriptor exceeds framing limit");
    std::ifstream meta(prefix + ".router", std::ios::binary);
    meta.imbue(std::locale::classic());
    std::string   magic, backend;
    uint32_t      dim = 0, count = 0;
    RouterOptions options;
    options.memory_budget_bytes = budget;
    uint64_t vector_bytes = 0, vector_hash = 0, graph_bytes = 0, graph_hash = 0;
    meta >> magic >> backend >> dim >> count >> options.degree >>
        options.construction_breadth >> options.query_breadth >>
        options.insert_breadth >> options.seed >> vector_bytes >> vector_hash >>
        graph_bytes >> graph_hash;
    check(bool(meta) && (magic == "IVFROUTER3" || magic == "IVFROUTER4") &&
              count <= INT32_MAX,
          "Invalid router descriptor");
    check(backend == router_backend_id,
           "Unsupported IVF router producer; rebuild with faiss-1.9.0-cpu1");
    uint64_t native_bytes = 0, native_hash = 0, refined_bytes = 0,
             refined_hash = 0, pca_bytes = 0, pca_hash = 0;
    PatchedGraphDiagnostics base_stats{}, refined_stats{};
    if (magic == "IVFROUTER4") {
      std::string mode, optimized_backend;
      uint32_t    reverse = 0;
      meta >> optimized_backend >> mode >> options.refine_breadth >> reverse >>
          options.pca_variance >> options.router_rerank_factor >>
          options.native_work_limit >> native_bytes >> native_hash >>
          refined_bytes >> refined_hash >> pca_bytes >> pca_hash;
      for (auto* s : {&base_stats, &refined_stats}) {
        meta >> s->unreachable_before >> s->unreachable_after >>
            s->zero_indegree >> s->reinserted_nodes >> s->refined_nodes;
        check(bool(meta) && s->unreachable_before < count &&
                  s->unreachable_after < count && s->zero_indegree <= count &&
                  s->reinserted_nodes <= count,
              "Invalid patched graph diagnostics");
      }
      check(
          base_stats.refined_nodes == 0 && refined_stats.refined_nodes == count,
          "Graph ablation identities mismatch");
      check(bool(meta) && reverse <= 1, "Invalid native router descriptor");
      check(optimized_backend == optimized_router_backend_id,
            "Unsupported optimized FAISS router producer");
      options.native_modes = true;
      options.mode = parse_router_mode(mode);
      options.refine_reverse_edges = reverse != 0;
      check(native_bytes <= budget && refined_bytes <= budget &&
                pca_bytes <= budget &&
                std::filesystem::file_size(prefix + ".patched.hnsw") ==
                    native_bytes &&
                std::filesystem::file_size(prefix + ".refined.hnsw") ==
                    refined_bytes &&
                artifact_hash_file(prefix + ".refined.hnsw") == refined_hash &&
                std::filesystem::file_size(prefix + ".pca") == pca_bytes &&
                artifact_hash_file(prefix + ".patched.hnsw") == native_hash &&
                artifact_hash_file(prefix + ".pca") == pca_hash,
            "Native router artifact size/hash mismatch");
    }
    meta >> std::ws;
    check(meta.eof(), "Trailing router descriptor data");
    const uint64_t cells = uint64_t(dim) * count;
    check(cells <= SIZE_MAX / sizeof(float), "Router centroid shape overflow");
    validate(dim, static_cast<size_t>(cells), options);
    check(vector_bytes == 8 + cells * sizeof(float) &&
              std::filesystem::file_size(vectors) == vector_bytes &&
              graph_bytes <= budget &&
              std::filesystem::file_size(graph) == graph_bytes &&
              artifact_hash_file(vectors) == vector_hash &&
              artifact_hash_file(graph) == graph_hash,
          "Router artifact size/checksum mismatch");
    const auto shape = inspect_vectors(vectors, DataType::Float, dim);
    check(shape.rows == count, "Router centroid count mismatch");
    std::vector<float> centroids(static_cast<size_t>(cells));
    std::ifstream      input(vectors, std::ios::binary);
    input.seekg(8);
    input.read(reinterpret_cast<char*>(centroids.data()),
               cells * sizeof(float));
    check(bool(input), "Truncated centroid artifact");
    validate_values(centroids, dim);
    validate_graph(graph, dim, centroids, options);
    auto router = std::unique_ptr<CentroidRouter>(
        new CentroidRouter(dim, centroids, options));
    // Release the validation table before loading the graph's vector copy;
    // the preflight estimate allows two copies, not three overlapping ones.
    std::vector<float>().swap(centroids);
    router->_impl->graph = faiss_detail::load(graph, dim, router->_centroids,
        {options.degree, options.construction_breadth, options.seed,
         budget, options.native_work_limit});
    if (options.native_modes) {
      PatchedGraphOptions g{options.degree, options.construction_breadth,
                            options.seed, budget, options.native_work_limit};
      validate_graph(prefix + ".patched.hnsw", dim, router->_centroids,
                     options);
      validate_graph(prefix + ".refined.hnsw", dim, router->_centroids,
                     options);
      std::ifstream pca(prefix + ".pca", std::ios::binary);
      router->_impl->native = PatchedCentroidGraph::load(
          prefix + ".patched.hnsw", dim, router->_centroids, g, base_stats);
      router->_impl->refined = PatchedCentroidGraph::load(
          prefix + ".refined.hnsw", dim, router->_centroids, g, refined_stats);
      router->_impl->pca = std::make_unique<CentroidPCA>(
          CentroidPCA::read(pca, dim, router->_centroids, options.pca_variance,
                            budget, options.native_work_limit));
      router->_impl->native->configure_projection(router->projected_dim());
      router->_impl->refined->configure_projection(router->projected_dim());
      check(router->_impl->native->diagnostics().refined_nodes == 0 &&
                router->_impl->refined->diagnostics().refined_nodes == count,
            "Native graph ablation identities mismatch");
      check(pca.peek() == EOF, "Trailing PCA artifact bytes");
    }
    return router;
  }

  CentroidTraining train_centroids_hnsw(const std::string&   path,
                                        const BuildConfig&   config,
                                        const RouterOptions& options,
                                        const TrainerLimits& limits) {
    config.validate();
    const uint64_t workers =
        std::min(config.training_count(), config.build_threads);
    uint64_t workspace =
        CentroidRouter::estimate_build_bytes(config.dim, config.nlist, options);
    // Visited tables and candidate heaps/materialization per simultaneous
    // worker beyond the graph's fixed preflight allowance.
    const uint64_t per_worker =
        2ull * config.nlist +
        64ull * std::min(config.nlist, options.insert_breadth) + 4096;
    check(workers <= (UINT64_MAX - workspace) / per_worker,
          "HNSW trainer workspace overflow");
    workspace += workers * per_worker;
    struct RouterAssignment final : detail::AssignmentIndex {
      explicit RouterAssignment(std::unique_ptr<CentroidRouter> graph)
          : router(std::move(graph)) {
      }
      uint32_t assign(std::span<const float> vector, uint32_t,
                      uint64_t&) const override {
        // The rebuilt training control deliberately retains breadth-sized
        // double rescoring. This is not the raw vanilla query comparison.
        return router->search(vector, std::min(router->nlist(),
                               router->options().insert_breadth), false, nullptr,
                               {RouterMode::Faiss, 0, router->options().insert_breadth})
            .front().list_id;
      }
      std::unique_ptr<CentroidRouter> router;
    };
    const detail::AssignmentFactory factory =
        [&](std::span<const float> centroids) {
          return std::make_unique<RouterAssignment>(
              CentroidRouter::build(config.dim, centroids, options));
        };
    return detail::train_centroids_impl(path, config, limits, &factory,
                                        workspace);
  }

  std::unique_ptr<detail::AssignmentIndex> detail::make_reusable_assignment(
      std::span<const float> centroids, const BuildConfig& config,
      const RouterOptions& options) {
    config.validate();
    check(trainer_reuses_graph(config.trainer_mode) &&
              centroids.size() == uint64_t(config.nlist) * config.dim,
          "Invalid reusable training index shape/mode");
    validate_values(centroids, config.dim);
    struct ReusableAssignment final : detail::AssignmentIndex {
      std::unique_ptr<PatchedCentroidGraph> graph;
      uint32_t                              count, dimension;
      TrainingSearchPolicy                  search_policy;
      std::span<const float>                coordinates;
      BuildConfig                           config;
      RouterOptions                         options;
      std::unique_ptr<CentroidPCA>          pca;
      TrainingPass                          phase;
      ReusableAssignment(uint32_t dim, std::span<const float> centroids,
                         const RouterOptions& opts, const BuildConfig& cfg)
          : graph(PatchedCentroidGraph::build(
                dim, centroids,
                {opts.degree, opts.construction_breadth, opts.seed,
                 opts.memory_budget_bytes, opts.native_work_limit})),
            count(uint32_t(centroids.size() / dim)), dimension(dim),
            search_policy(training_search_policy(cfg)), coordinates(centroids),
            config(cfg), options(opts) {
        if (trainer_optimized_search(config.trainer_mode)) {
          const auto start = std::chrono::steady_clock::now();
          graph->refine(config.training_refine_breadth);
          phase.refinement_seconds = seconds(start);
          phase.initially_refined_nodes = graph->diagnostics().refined_nodes;
          phase.isolated_reinsertions = graph->diagnostics().reinserted_nodes;
          fit_projection();
        }
      }
      static double seconds(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
            .count();
      }
      void fit_projection() {
        const auto start = std::chrono::steady_clock::now();
        pca.reset();  // Never overlap two solver/projection workspaces.
        pca = std::make_unique<CentroidPCA>(CentroidPCA::fit(
            dimension, coordinates, .95, options.memory_budget_bytes,
            options.native_work_limit));
        validate_values(pca->projected(), pca->retained_dim());
        graph->configure_projection(pca->retained_dim());
        phase.pca_seconds = seconds(start);
      }
      void describe_pass(TrainingPass& pass) const override {
        pass.refinement_seconds = phase.refinement_seconds;
        pass.pca_seconds = phase.pca_seconds;
        pass.initially_refined_nodes = phase.initially_refined_nodes;
        pass.isolated_reinsertions = phase.isolated_reinsertions;
        pass.requested_breadth = config.training_breadth;
        pass.candidate_count = search_policy.candidates;
        pass.effective_breadth = search_policy.breadth;
        pass.projected_dim = pca ? pca->retained_dim() : dimension;
        pass.retained_variance = pca ? pca->retained_variance() : 1;
        pass.bottom_unreachable = graph->diagnostics().unreachable_after;
        pass.zero_indegree = graph->diagnostics().zero_indegree;
      }
      uint32_t assign(std::span<const float> vector, uint32_t previous,
                      uint64_t& work) const override {
        detail::AssignmentMeasurements measured;
        const auto result = assign_measured(vector, previous, measured);
        work += measured.graph_work;
        return result;
      }
      uint32_t assign_measured(
          std::span<const float> vector, uint32_t previous,
          detail::AssignmentMeasurements& m) const override {
        const auto entries =
            trainer_seeds_assignment(config.trainer_mode) && previous < count
                ? std::span<const uint32_t>(&previous, 1)
                : std::span<const uint32_t>();
        const bool counted = trainer_count_candidates(config.trainer_mode);
        if (counted)
          validate_values(vector, dimension);
        std::vector<float> projected;
        auto               start = std::chrono::steady_clock::now();
        if (pca) {
          projected = pca->project(vector);
          validate_values(projected, pca->retained_dim());
          m.projection_seconds += seconds(start);
        }
        start = std::chrono::steady_clock::now();
        uint64_t distances = 0;
        auto     hits = graph->search(
            pca ? std::span<const float>(projected) : vector,
            search_policy.candidates, search_policy.breadth, entries,
            pca ? pca->projected() : std::span<const float>{},
            pca ? pca->retained_dim() : 0, &distances,
            trainer_seeds_assignment(config.trainer_mode));
        m.traversal_seconds += seconds(start);
        m.graph_work += distances;
        m.returned_candidates += hits.size();
        check(!hits.empty(), "Training ANN returned no centroid");
        // Count-based full traversal returns nearest-one in float; its double
        // objective distance is computed separately by the shared trainer.
        if (counted && !pca)
          return hits.front().id;
        start = std::chrono::steady_clock::now();
        // Preserve breadth-sized double rescoring for the legacy controls.
        double   best = std::numeric_limits<double>::infinity();
        uint32_t label = UINT32_MAX;
        for (auto hit : hits) {
          const double distance = detail::squared_l2_double_unchecked(
              vector.data(), coordinates.data() + size_t(hit.id) * dimension,
              dimension);
          if (distance < best || (distance == best && hit.id < label)) {
            best = distance;
            label = hit.id;
          }
        }
        m.rerank_distances += hits.size();
        m.rerank_seconds += seconds(start);
        check(label < count, "Training ANN returned no centroid");
        return label;
      }
      void refresh(std::span<const float> centroids) override {
        validate_values(centroids, dimension);
        phase = {};
        graph->refresh(centroids);
        coordinates = centroids;
        if (pca)
          fit_projection();  // Joined boundary, including final assignment.
      }
      uint32_t unreachable() const override {
        return graph->diagnostics().unreachable_after;
      }
    };
    return std::make_unique<ReusableAssignment>(config.dim, centroids, options,
                                                config);
  }

  CentroidTraining train_centroids_ann(const std::string&   path,
                                       const BuildConfig&   config,
                                       const RouterOptions& options,
                                       const TrainerLimits& limits) {
    config.validate();
    if (config.trainer_mode == TrainerMode::Exact)
      return train_centroids(path, config, limits);
    auto training_options = options;
    training_options.native_modes = false;
    training_options.mode = RouterMode::Faiss;
    training_options.insert_breadth = config.training_breadth;
    if (config.trainer_mode == TrainerMode::AnnRebuild)
      return train_centroids_hnsw(path, config, training_options, limits);
    uint64_t workspace = CentroidRouter::estimate_build_bytes(
        config.dim, config.nlist, training_options);
    const uint64_t snapshot =
        uint64_t(config.nlist) * (uint64_t(options.degree) * 32 + 512);
    const uint64_t workers =
        std::min(config.training_count(), config.build_threads);
    const bool     optimized = trainer_optimized_search(config.trainer_mode);
    const auto     search_policy = training_search_policy(config);
    const uint64_t per_worker = 2ull * config.nlist +
                                64ull * search_policy.breadth +
                                uint64_t(config.dim) * sizeof(float) + 4096;
    check(snapshot <= UINT64_MAX - workspace,
          "Training snapshot estimate overflow");
    workspace += snapshot;
    if (optimized) {
      CentroidPCA::preflight_work(config.dim, config.nlist,
                                  options.native_work_limit);
      const uint64_t pca =
          CentroidPCA::estimate_bytes(config.dim, config.nlist);
      const uint64_t refinement_scratch =
          64ull *
          std::min<uint64_t>(config.nlist,
                             uint64_t(config.training_refine_breadth) + 1);
      check(pca <= UINT64_MAX - workspace, "Training PCA estimate overflow");
      workspace += pca;
      check(refinement_scratch <= UINT64_MAX - workspace,
            "Training refinement estimate overflow");
      workspace += refinement_scratch;
    }
    check(workers <= (UINT64_MAX - workspace) / per_worker,
          "Training worker estimate overflow");
    workspace += workers * per_worker;
    const detail::AssignmentFactory factory =
        [&](std::span<const float> centroids) {
          return detail::make_reusable_assignment(centroids, config,
                                                  training_options);
        };
    return detail::train_centroids_impl(
        path, config, limits, &factory, workspace, true,
        trainer_seeds_assignment(config.trainer_mode));
  }
}  // namespace diskann::ivf
