#include "bufann/ivf_router.h"
#include "bufann/ivf_patched_graph.h"
#include "bufann/ivf_pca.h"
#include "../../src/bufann/ivf_centroids_internal.h"
#include <ivf_patched/hnswlib.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace diskann::ivf;
namespace {
  void require(bool ok, const char* why) {
    if (!ok)
      throw std::runtime_error(why);
  }
  template<class F>
  void rejects(F f) {
    bool failed = false;
    try {
      f();
    } catch (const std::exception&) {
      failed = true;
    }
    require(failed, "Expected rejection");
  }
  template<class T>
  void vectors(const std::string& path, uint32_t rows, uint32_t dim,
               const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    uint32_t      header[]{rows, dim};
    out.write(reinterpret_cast<char*>(header), 8);
    out.write(reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(T));
    out.close();
    require(bool(out), "Fixture write failed");
  }
  template<class T>
  void typed(const std::filesystem::path& root, DataType dtype) {
    BuildConfig config;
    config.data_type = dtype;
    config.dim = 13;
    config.base_count = 192;
    config.nlist = 8;
    config.max_reps = 3;
    config.build_threads = 1;
    config.sample_rate = .75;
    config.seed = 41;
    config.training_breadth = 32;
    std::vector<T> values(208 * config.dim);
    for (uint32_t r = 0; r < 208; ++r)
      for (uint32_t d = 0; d < config.dim; ++d)
        values[size_t(r) * config.dim + d] =
            T((r % 8) * 12 + (r / 8) % 7 + d % 3);
    if constexpr (std::is_same_v<T, float>)
      std::fill(values.begin() + size_t(config.base_count) * config.dim,
                values.end(), std::numeric_limits<float>::quiet_NaN());
    auto path = (root / (std::string(data_type_name(dtype)) + ".bin")).string();
    vectors(path, 208, config.dim, values);
    RouterOptions options;
    options.degree = 4;
    options.construction_breadth = 32;
    for (auto init : {CentroidInit::Sample, CentroidInit::RandomPartition}) {
      config.init = init;
      std::vector<uint32_t> ids;
      for (auto mode :
           {TrainerMode::Exact, TrainerMode::AnnRebuild, TrainerMode::AnnReuse,
            TrainerMode::AnnReuseSeeded, TrainerMode::AnnReuseCount,
            TrainerMode::AnnReuseSeededCount, TrainerMode::LV,
            TrainerMode::LVPlus}) {
        if (trainer_optimized_search(mode) &&
            init != CentroidInit::RandomPartition)
          continue;
        config.trainer_mode = mode;
        config.build_threads = 1;
        const auto trained = train_centroids_ann(path, config, options);
        config.build_threads = 4;
        const auto parallel = train_centroids_ann(path, config, options);
        require(trained.centroids == parallel.centroids &&
                    trained.assignments == parallel.assignments &&
                    trained.objective_history == parallel.objective_history,
                "Training depends on assignment worker order");
        require(trained.sampled_ids == parallel.sampled_ids,
                "Sample IDs depend on workers");
        if (ids.empty())
          ids = trained.sampled_ids;
        else
          require(ids == trained.sampled_ids, "Controls sample different IDs");
        require(*std::max_element(ids.begin(), ids.end()) < config.base_count,
                "Tail trained");
        const bool reuse = trainer_reuses_graph(mode);
        require(
            trained.assignment_index_builds == (mode == TrainerMode::Exact ? 0
                                                : reuse                    ? 1
                                                                           : 4),
            "Graph build count");
        require(trained.assignment_index_refreshes == (reuse ? 3 : 0) &&
                    trained.reinserted_centroids == (reuse ? 24 : 0),
                "Graph reuse count");
        require(trained.seeded_searches ==
                        (trainer_seeds_assignment(mode) ? 432 : 0) &&
                    trained.seed_fallbacks ==
                        (trainer_seeds_assignment(mode) ? 144 : 0),
                "Previous assignment/fallback counts");
        if (trainer_count_candidates(mode)) {
          const uint32_t candidates = trainer_optimized_search(mode) ? 3 : 1;
          require(trained.returned_candidates ==
                      trained.assignment_searches * candidates,
                  "Candidate count coupled to breadth");
          require(
              trained.candidate_rerank_distances ==
                  (trainer_optimized_search(mode) ? trained.returned_candidates
                                                  : 0),
              "Rerank count incorrect");
          require(trained.objective_distance_evaluations ==
                      trained.assignment_searches,
                  "Objective work omitted");
          for (size_t p = 0; p < trained.passes.size(); ++p) {
            const auto& pass = trained.passes[p];
            require(pass.candidate_count == candidates &&
                        pass.requested_breadth == 32 &&
                        pass.effective_breadth == 8,
                    "Search contract incorrect");
            if (trainer_optimized_search(mode)) {
              require(pass.projected_dim > 0 &&
                          pass.projected_dim < config.dim &&
                          pass.retained_variance >= .95,
                      "PCA not used during training");
              require(pass.pca_seconds > 0 && pass.initially_refined_nodes ==
                                                  (p ? 0u : config.nlist),
                      "Moving PCA/initial refinement lifecycle");
            }
          }
        }
        require(trained.passes.size() == 4 &&
                    trained.passes.back().mean_seconds == 0,
                "Final pass missing");
        std::vector<uint32_t> occupancy(config.nlist);
        double                assigned = 0, exact_sse = 0;
        for (size_t r = 0; r < ids.size(); ++r) {
          const auto label = trained.assignments[r];
          require(label < config.nlist, "Bad final label");
          ++occupancy[label];
          double nearest = std::numeric_limits<double>::infinity();
          for (uint32_t c = 0; c < config.nlist; ++c) {
            double distance = 0;
            for (uint32_t d = 0; d < config.dim; ++d) {
              const double delta =
                  double(values[size_t(ids[r]) * config.dim + d]) -
                  trained.centroids[size_t(c) * config.dim + d];
              distance += delta * delta;
            }
            nearest = std::min(nearest, distance);
            if (c == label)
              assigned += distance;
          }
          exact_sse += nearest;
        }
        require(occupancy == trained.occupancy &&
                    assigned == trained.final_objective &&
                    exact_sse <= assigned + 1e-9,
                "Final occupancy/SSE does not describe returned centroids");
        require(std::isfinite(assigned), "Nonfinite SSE");
        if (mode == TrainerMode::Exact)
          require(exact_sse == assigned, "Exact oracle mismatch");
      }
    }
    config.trainer_mode = TrainerMode::AnnReuseSeeded;
    rejects([&] { train_centroids_ann(path, config, options, {1, 1, 1}); });
    config.memory_budget_gb = .00001;
    rejects([&] { train_centroids_ann(path, config, options); });
    config.memory_budget_gb = 1;
    config.training_breadth = 0;
    rejects([&] { train_centroids_ann(path, config, options); });
  }
  void raw_seed_and_reinsert() {
    ivf_hnswlib::L2Space                space(2);
    ivf_hnswlib::HierarchicalNSW<float> graph(&space, 3, 2, 8, 10);
    const float                         values[]{0, 0, 10, 10, 20, 20};
    for (uint32_t id = 0; id < 3; ++id)
      graph.addPoint(values + id * 2, id);
    graph.setEf(1);
    graph.replaceBottomIVF(0, {});
    graph.replaceBottomIVF(1, {});
    graph.replaceBottomIVF(2, {});
    graph.enterpoint_node_ = 2;
    graph.maxlevel_ = 0;
    const auto global = graph.searchKnnIVF(values, 1);
    const auto seeded =
        graph.searchKnnIVF(values, 1, nullptr, nullptr, {0}, true);
    require(global.top().second == 2 && seeded.top().second == 0,
            "Seeded traversal still uses global descent");
    require(graph.searchKnnIVF(values, 1, nullptr, nullptr, {}, true)
                    .top()
                    .second == 2,
            "Missing seed fallback");
    rejects(
        [&] { graph.searchKnnIVF(values, 1, nullptr, nullptr, {3}, true); });
    graph.maxlevel_ = *std::max_element(graph.element_levels_.begin(),
                                        graph.element_levels_.begin() + 3);
    for (uint32_t id = 0; id < 3; ++id)
      if (graph.element_levels_[id] == graph.maxlevel_)
        graph.enterpoint_node_ = id;
    auto        levels = graph.element_levels_;
    const float moved[]{20, 20, 0, 0, 10, 10};
    for (uint32_t id = 0; id < 3; ++id)
      graph.replaceVectorIVF(id, moved + id * 2);
    for (uint32_t id = 0; id < 3; ++id)
      graph.reinsertFromIVF(id, {(id + 1) % 3});
    require(levels == graph.element_levels_ && graph.cur_element_count == 3,
            "Reuse changed levels or IDs");
    for (uint32_t id = 0; id < 3; ++id) {
      const auto* data =
          reinterpret_cast<const float*>(graph.getDataByInternalId(id));
      require(data[0] == moved[id * 2] && data[1] == moved[id * 2 + 1] &&
                  graph.getExternalLabel(id) == id,
              "Coordinates/labels stale");
    }
  }
  void joined_refresh(const std::filesystem::path& root) {
    BuildConfig config;
    config.dim = 1;
    config.base_count = 64;
    config.nlist = 2;
    config.max_reps = 3;
    config.build_threads = 4;
    config.init = CentroidInit::RandomPartition;
    const auto path = (root / "joined.bin").string();
    vectors(path, 64, 1, std::vector<float>(64, 1));
    struct Index final : detail::AssignmentIndex {
      mutable std::atomic<uint32_t> active{0};
      uint32_t                      pass = 0;
      uint32_t assign(std::span<const float>, uint32_t previous,
                      uint64_t&) const override {
        active.fetch_add(1);
        require(previous == (pass ? 1u : UINT32_MAX),
                "Previous assignment not supplied");
        active.fetch_sub(1);
        return 1;
      }
      void refresh(std::span<const float>) override {
        require(active.load() == 0, "Refresh overlaps assignment");
        ++pass;
      }
    };
    const detail::AssignmentFactory factory = [](std::span<const float>) {
      return std::make_unique<Index>();
    };
    const auto trained = detail::train_centroids_impl(
        path, config, {1, 10000, 10000}, &factory, 4096, true, true);
    require(trained.empty_recoveries == 3 && trained.seeded_searches == 192,
            "Empty recovery/seed loop");
  }
}  // namespace
void moving_projection() {
  const std::vector<float> rank1{-2, 0, -1, 0, 1, 0, 2, 0},
      rank2{-2, 0, 0, -2, 0, 2, 2, 0};
  auto graph = PatchedCentroidGraph::build(2, rank1, {2, 16, 41});
  graph->refine(16);
  auto pca = CentroidPCA::fit(2, rank1);
  require(pca.retained_dim() == 1, "Rank-one PCA");
  graph->configure_projection(pca.retained_dim());
  graph->refresh(rank2);
  pca = CentroidPCA::fit(2, rank2);
  require(pca.retained_dim() == 2, "Moving PCA dimension unchanged");
  graph->configure_projection(pca.retained_dim());
  const float query[]{0, 2};
  const auto  projected = pca.project(query);
  auto        hits =
      graph->search(projected, 3, 16, {}, pca.projected(), pca.retained_dim());
  require(hits.front().id == 2, "Refreshed projection/coordinates stale");
  graph->refresh(rank1);
  pca = CentroidPCA::fit(2, rank1);
  graph->configure_projection(pca.retained_dim());
  require(pca.retained_dim() == 1, "Projection kernel cannot shrink");
}
void assignment_oracle() {
  BuildConfig c;
  c.dim = 3;
  c.base_count = 64;
  c.nlist = 16;
  c.init = CentroidInit::RandomPartition;
  RouterOptions options;
  options.degree = 4;
  options.construction_breadth = 32;
  for (auto mode : {TrainerMode::AnnReuse, TrainerMode::AnnReuseCount,
                    TrainerMode::LV, TrainerMode::LVPlus})
    for (uint32_t breadth : {1u, 7u, 32u})
      for (double factor : {1., 2.25, 5.}) {
        if (!trainer_optimized_search(mode) && factor != 1.)
          continue;
        c.trainer_mode = mode;
        c.training_breadth = breadth;
        c.training_rerank_factor = trainer_optimized_search(mode) ? factor : 3;
        std::vector<float> coordinates;
        for (uint32_t id = 0; id < c.nlist; ++id)
          coordinates.insert(
              coordinates.end(),
              {float(id * 4), float(id % 3) * .1f, float(id % 2) * .01f});
        auto index = detail::make_reusable_assignment(coordinates, c, options);
        if (trainer_count_candidates(mode))
          rejects([&] {
            const float excessive[]{1e30f, 1e30f, 1e30f};
            uint64_t    work = 0;
            index->assign(excessive, UINT32_MAX, work);
          });
        auto graph = PatchedCentroidGraph::build(
            c.dim, coordinates,
            {options.degree, options.construction_breadth, options.seed});
        if (trainer_optimized_search(mode))
          graph->refine(c.training_refine_breadth);
        for (uint32_t pass = 0; pass < 2; ++pass) {
          std::optional<CentroidPCA> pca;
          if (trainer_optimized_search(mode)) {
            pca = CentroidPCA::fit(c.dim, coordinates);
            graph->configure_projection(pca->retained_dim());
          }
          for (uint32_t q = 0; q < 32; ++q) {
            const float    query[]{float(q * 2), float(q % 3) * .11f,
                                   float(q % 2) * .012f};
            const uint32_t previous = pass ? q % c.nlist : UINT32_MAX;
            const auto     entries =
                trainer_seeds_assignment(mode) && previous < c.nlist
                        ? std::span<const uint32_t>(&previous, 1)
                        : std::span<const uint32_t>{};
            const auto projection =
                pca ? pca->project(query) : std::vector<float>{};
            const uint32_t count =
                pca ? uint32_t(std::min<double>(c.nlist, std::ceil(factor)))
                : trainer_count_candidates(mode) ? 1
                                                 : std::min(c.nlist, breadth);
            uint64_t   computations = 0;
            const auto hits =
                graph->search(pca ? std::span<const float>(projection)
                                  : std::span<const float>(query),
                              count, std::max(breadth, count), entries,
                              pca ? pca->projected() : std::span<const float>{},
                              pca ? pca->retained_dim() : 0, &computations,
                              trainer_seeds_assignment(mode));
            uint32_t expected = hits.front().id;
            double   best = std::numeric_limits<double>::infinity();
            if (pca || !trainer_count_candidates(mode))
              for (const auto hit : hits) {
                double distance = 0;
                for (uint32_t d = 0; d < c.dim; ++d) {
                  const double delta =
                      double(query[d]) - coordinates[hit.id * c.dim + d];
                  distance += delta * delta;
                }
                if (distance < best ||
                    (distance == best && hit.id < expected)) {
                  best = distance;
                  expected = hit.id;
                }
              }
            detail::AssignmentMeasurements m;
            require(index->assign_measured(query, previous, m) == expected,
                    "Independent returned-candidate double-L2 oracle");
            require(
                m.returned_candidates == hits.size() &&
                    m.graph_work == computations &&
                    m.rerank_distances ==
                        (pca || !trainer_count_candidates(mode) ? hits.size()
                                                                : 0),
                "Measured assignment work mismatch");
          }
          // Change rank and centroid positions; both maintain full-coordinate
          // connections, then project exactly the coordinates for the next
          // pass.
          for (uint32_t id = 0; id < c.nlist; ++id) {
            coordinates[id * 3] = float((id % 4) * 8);
            coordinates[id * 3 + 1] = float((id / 4) * 8);
          }
          index->refresh(coordinates);
          graph->refresh(coordinates);
        }
      }
  c.trainer_mode = TrainerMode::LV;
  c.training_rerank_factor = 3;
  options.native_work_limit = 1;
  rejects([&] {
    detail::make_reusable_assignment(std::vector<float>(48, 1), c, options);
  });
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "scratch required");
    const std::filesystem::path root(argv[1]);
    std::filesystem::create_directories(root);
    raw_seed_and_reinsert();
    joined_refresh(root);
    moving_projection();
    assignment_oracle();
    typed<float>(root, DataType::Float);
    typed<int8_t>(root, DataType::Int8);
    typed<uint8_t>(root, DataType::UInt8);
    const auto path = (root / "singleton.bin").string();
    vectors(path, 4, 1, std::vector<float>(4, 2));
    BuildConfig c;
    c.dim = 1;
    c.base_count = 4;
    c.nlist = 1;
    c.max_reps = 2;
    c.trainer_mode = TrainerMode::AnnReuseSeeded;
    c.init = CentroidInit::RandomPartition;
    require(train_centroids_ann(path, c).final_objective == 0,
            "Singleton reuse");
    c.nlist = 4;
    const auto duplicate = train_centroids_ann(path, c);
    require(duplicate.empty_recoveries > 0 && duplicate.final_objective == 0,
            "Duplicate recovery");
    for (auto mode : {TrainerMode::LV, TrainerMode::LVPlus}) {
      c.trainer_mode = mode;
      c.nlist = 1;
      const auto single = train_centroids_ann(path, c);
      require(
          single.final_objective == 0 &&
              single.candidate_rerank_distances == single.assignment_searches,
          "Tiny LV candidate cap");
      c.nlist = 4;
      const auto degenerate = train_centroids_ann(path, c);
      require(
          degenerate.empty_recoveries > 0 && degenerate.final_objective == 0,
          "Degenerate LV recovery");
      c.init = CentroidInit::Sample;
      rejects([&] { train_centroids_ann(path, c); });
      c.init = CentroidInit::RandomPartition;
      c.training_rerank_factor = 0;
      rejects([&] { train_centroids_ann(path, c); });
      c.training_rerank_factor = 3;
    }
    std::cout << "ANN reuse/seeded training fixtures passed (all dtypes)\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
