#include "bufann/ivf_centroids.h"
#include "bufann/ivf_io.h"
#include "bufann/ivf_numeric_internal.h"
#include "ivf_centroids_internal.h"
#include "ivf_random_internal.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <exception>
#include <limits>
#include <numeric>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

namespace diskann::ivf {
  namespace {
    void check(bool ok, const char* message) {
      if (!ok)
        throw std::invalid_argument(message);
    }
    uint64_t mul(uint64_t a, uint64_t b) {
      check(b == 0 || a <= UINT64_MAX / b,
            "Centroid workspace/work count overflow");
      return a * b;
    }
    void add(uint64_t& total, uint64_t amount) {
      check(total <= UINT64_MAX - amount, "Centroid workspace overflow");
      total += amount;
    }
    uint32_t u32(const unsigned char* p) {
      return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 |
             uint32_t(p[3]) << 24;
    }

    template<typename T>
    void sample_base(const std::string& path, const BuildConfig& config,
                     std::vector<float>& samples, std::vector<uint32_t>& ids) {
      const auto meta = inspect_vectors(path, config.data_type, config.dim);
      check(config.base_count <= meta.rows,
            "Centroid base prefix exceeds input rows");
      std::ifstream input(path, std::ios::binary);
      unsigned char header[8]{};
      input.read(reinterpret_cast<char*>(header), sizeof(header));
      check(bool(input) && u32(header) == meta.rows &&
                u32(header + 4) == meta.cols,
            "Centroid input header changed");
      std::vector<T>  row(config.dim);
      std::mt19937_64 rng(config.seed);
      const uint32_t  count = static_cast<uint32_t>(ids.size());
      for (uint32_t r = 0; r < config.base_count; ++r) {
        input.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
        check(bool(input), "Truncated centroid input stream");
        for (T value : row)
          check(std::isfinite(float(value)),
                "Non-finite initial-base coordinate");
        const uint32_t slot =
            r < count ? r : detail::bounded_random(rng, uint64_t(r) + 1);
        if (slot >= count)
          continue;
        ids[slot] = r;
        for (uint32_t d = 0; d < config.dim; ++d)
          samples[size_t(slot) * config.dim + d] = float(row[d]);
      }
    }
  }  // namespace

  CentroidTraining train_centroids(const std::string&   path,
                                   const BuildConfig&   config,
                                   const TrainerLimits& limits) {
    return detail::train_centroids_impl(path, config, limits, nullptr, 0);
  }

  CentroidTraining detail::train_centroids_impl(
      const std::string& path, const BuildConfig& config,
      const TrainerLimits& limits, const AssignmentFactory* factory,
      uint64_t index_workspace_bytes, bool reuse, bool seeded) {
    static_assert(std::endian::native == std::endian::little &&
                  sizeof(float) == 4);
    config.validate();
    check(limits.block_rows > 0 && limits.max_distance_evaluations > 0 &&
              limits.max_assignment_searches > 0,
          "Invalid centroid trainer limits");
    const uint32_t count = config.training_count();
    const uint32_t workers = std::min(count, config.build_threads);
    check(workers <= 256, "Centroid trainer supports at most 256 workers");
    const uint64_t   sample_cells = mul(count, config.dim);
    const uint64_t   centroid_cells = mul(config.nlist, config.dim);
    CentroidTraining result;
    result.dim = config.dim;
    result.nlist = config.nlist;
    result.approximate_assignment = factory != nullptr;
    result.assignment_searches = mul(count, uint64_t(config.max_reps) + 1);
    check(result.assignment_searches <= limits.max_assignment_searches,
          "Centroid assignment exceeds declared search count limit");
    if (!factory) {
      result.distance_evaluations =
          mul(result.assignment_searches, config.nlist);
      check(*result.distance_evaluations <= limits.max_distance_evaluations,
            "Exact centroid assignment exceeds declared work limit");
    }
    result.objective_distance_evaluations =
        factory ? result.assignment_searches : *result.distance_evaluations;
    // Samples, IDs, permutation, assignments, distances; centroids, next means,
    // double sums and occupancy; row buffer, objective history, worker stacks.
    add(result.estimated_bytes, mul(sample_cells, sizeof(float)));
    add(result.estimated_bytes,
        mul(count, 3 * sizeof(uint32_t) + sizeof(double)));
    add(result.estimated_bytes,
        mul(centroid_cells, 2 * sizeof(float) + sizeof(double)));
    add(result.estimated_bytes, mul(config.nlist, sizeof(uint32_t)));
    add(result.estimated_bytes,
        mul(config.dim, element_size(config.data_type)));
    add(result.estimated_bytes, mul(config.max_reps, sizeof(double)));
    add(result.estimated_bytes,
        mul(uint64_t(config.max_reps) + 1, sizeof(TrainingPass)));
    add(result.estimated_bytes, mul(workers, 8ull * 1024 * 1024 + 4096));
    add(result.estimated_bytes, index_workspace_bytes);
    check(result.estimated_bytes <= std::numeric_limits<size_t>::max() &&
              static_cast<long double>(result.estimated_bytes) <=
                  static_cast<long double>(config.memory_budget_gb) * 1024 *
                      1024 * 1024,
          "Centroid trainer exceeds declared memory estimate budget");
    const auto         start = std::chrono::steady_clock::now();
    std::vector<float> samples(static_cast<size_t>(sample_cells));
    result.sampled_ids.resize(count);
    switch (config.data_type) {
      case DataType::Float:
        sample_base<float>(path, config, samples, result.sampled_ids);
        break;
      case DataType::Int8:
        sample_base<int8_t>(path, config, samples, result.sampled_ids);
        break;
      case DataType::UInt8:
        sample_base<uint8_t>(path, config, samples, result.sampled_ids);
        break;
    }
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(config.seed ^ 0x9e3779b97f4a7c15ull);
    result.centroids.resize(static_cast<size_t>(centroid_cells));
    if (config.init == CentroidInit::Sample) {
      for (uint32_t c = 0; c < config.nlist; ++c) {
        std::swap(order[c],
                  order[c + detail::bounded_random(rng, uint64_t(count) - c)]);
        std::copy_n(samples.data() + size_t(order[c]) * config.dim, config.dim,
                    result.centroids.data() + size_t(c) * config.dim);
      }
    } else {
      // Seeded balanced random partitions: shuffle rows, distribute
      // round-robin. Every initial partition is nonempty, including nlist ==
      // sample count.
      for (uint32_t r = 0; r < count; ++r)
        std::swap(order[r],
                  order[r + detail::bounded_random(rng, uint64_t(count) - r)]);
    }
    std::vector<uint32_t> assignment(count, UINT32_MAX);
    std::vector<double>   distances(count);
    std::vector<double>   sums(static_cast<size_t>(centroid_cells));
    std::vector<float>    next(static_cast<size_t>(centroid_cells));
    result.occupancy.resize(config.nlist);
    result.objective_history.reserve(config.max_reps);
    result.passes.reserve(size_t(config.max_reps) + 1);
    if (config.init == CentroidInit::RandomPartition) {
      for (uint32_t r = 0; r < count; ++r) {
        const uint32_t c = r % config.nlist;
        ++result.occupancy[c];
        for (uint32_t d = 0; d < config.dim; ++d)
          sums[size_t(c) * config.dim + d] +=
              samples[size_t(order[r]) * config.dim + d];
      }
      for (uint32_t c = 0; c < config.nlist; ++c)
        for (uint32_t d = 0; d < config.dim; ++d)
          result.centroids[size_t(c) * config.dim + d] =
              float(sums[size_t(c) * config.dim + d] / result.occupancy[c]);
    }
    std::unique_ptr<AssignmentIndex> index;
    auto                             assign = [&] {
      result.passes.emplace_back();
      auto& pass = result.passes.back();
      if (count >= 1000000)
        std::clog << "[IVF training] pass=" << result.passes.size() << '/'
                  << uint64_t(config.max_reps) + 1
                  << " phase=graph_pca rows=" << count << " workers=" << workers
                  << std::endl;
      auto phase = std::chrono::steady_clock::now();
      if (factory) {
        if (reuse && index) {
          index->refresh(result.centroids);
          ++result.assignment_index_refreshes;
          result.reinserted_centroids += config.nlist;
        } else {
          // Release the old graph before allocating its replacement.
          index.reset();
          index = (*factory)(result.centroids);
          ++result.assignment_index_builds;
        }
      }
      check(!factory || bool(index), "Assignment factory returned no index");
      if (index)
        index->describe_pass(pass);
      pass.graph_seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - phase)
                               .count();
      if (count >= 1000000)
        std::clog << "[IVF training] pass=" << result.passes.size()
                  << " phase=assignment graph_s=" << pass.graph_seconds
                  << " pca_s=" << pass.pca_seconds << std::endl;
      phase = std::chrono::steady_clock::now();
      std::atomic<uint64_t> cursor{0};
      std::atomic<uint64_t> graph_work{0}, seed_count{0}, fallback_count{0};
      std::vector<AssignmentMeasurements> measurements(workers);
      std::exception_ptr                  error;
      std::mutex                          error_mutex;
      auto                                work = [&](uint32_t worker) {
        try {
          auto&    measured = measurements[worker];
          uint64_t local_seeded = 0, local_fallback = 0;
          for (;;) {
            const uint64_t begin = cursor.fetch_add(limits.block_rows);
            if (begin >= count)
              break;
            const uint64_t end =
                std::min<uint64_t>(count, begin + limits.block_rows);
            for (uint64_t r = begin; r < end; ++r) {
              double   best = std::numeric_limits<double>::infinity();
              uint32_t label = 0;
              if (index) {
                const uint32_t previous = seeded ? assignment[r] : UINT32_MAX;
                if (seeded) {
                  if (previous < config.nlist)
                    ++local_seeded;
                  else
                    ++local_fallback;
                }
                label = index->assign_measured(
                    std::span<const float>(samples).subspan(r * config.dim,
                                                                                                                       config.dim),
                    previous, measured);
                check(label < config.nlist,
                                                                                 "Assignment index returned an invalid centroid ID");
              }
              const uint32_t first = index ? label : 0;
              const uint32_t last = index ? label + 1 : config.nlist;
              for (uint32_t c = first; c < last; ++c) {
                const double dist = detail::squared_l2_double_unchecked(
                    samples.data() + r * config.dim,
                    result.centroids.data() + size_t(c) * config.dim,
                    config.dim);
                if (dist < best) {
                  best = dist;
                  label = c;
                }
              }
              assignment[r] = label;
              distances[r] = best;
            }
          }
          graph_work.fetch_add(measured.graph_work);
          seed_count.fetch_add(local_seeded);
          fallback_count.fetch_add(local_fallback);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!error)
            error = std::current_exception();
          cursor.store(count);  // Stop assigning additional blocks.
        }
      };
      {
        std::vector<std::jthread> threads;
        for (uint32_t w = 1; w < workers; ++w)
          threads.emplace_back(work, w);
        work(0);  // Destruction joins before error/result inspection.
      }
      if (error)
        std::rethrow_exception(error);
      pass.assignment_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - phase)
                                    .count();
      result.graph_distance_evaluations += graph_work.load();
      for (const auto& m : measurements) {
        pass.returned_candidates += m.returned_candidates;
        pass.candidate_rerank_distances += m.rerank_distances;
        pass.projection_worker_seconds += m.projection_seconds;
        pass.traversal_worker_seconds += m.traversal_seconds;
        pass.rerank_worker_seconds += m.rerank_seconds;
      }
      result.returned_candidates += pass.returned_candidates;
      result.candidate_rerank_distances += pass.candidate_rerank_distances;
      result.seeded_searches += seed_count.load();
      result.seed_fallbacks += fallback_count.load();
      if (count >= 1000000)
        std::clog << "[IVF training] pass=" << result.passes.size()
                  << " phase=assignment_complete assignment_s="
                  << pass.assignment_seconds << std::endl;
    };
    for (uint32_t iteration = 0; iteration < config.max_reps; ++iteration) {
      assign();
      const auto mean_start = std::chrono::steady_clock::now();
      std::fill(sums.begin(), sums.end(), 0);
      std::fill(result.occupancy.begin(), result.occupancy.end(), 0);
      double objective = 0;
      for (uint32_t r = 0; r < count; ++r) {
        const uint32_t c = assignment[r];
        ++result.occupancy[c];
        objective += distances[r];
        for (uint32_t d = 0; d < config.dim; ++d)
          sums[size_t(c) * config.dim + d] +=
              samples[size_t(r) * config.dim + d];
      }
      result.objective_history.push_back(objective);
      result.passes.back().assigned_sse = objective;
      const auto empty_count =
          std::count(result.occupancy.begin(), result.occupancy.end(), 0);
      if (empty_count) {
        std::iota(order.begin(), order.end(), 0);
        // Only the first empty_count rows are consumed. Preserve the exact
        // full-sort prefix/tie order without sorting every training row.
        std::partial_sort(order.begin(), order.begin() + empty_count,
                          order.end(), [&](uint32_t a, uint32_t b) {
                            if (distances[a] != distances[b])
                              return distances[a] > distances[b];
                            return result.sampled_ids[a] <
                                   result.sampled_ids[b];
                          });
      }
      uint32_t recovery = 0;
      for (uint32_t c = 0; c < config.nlist; ++c) {
        if (!result.occupancy[c]) {
          ++result.empty_recoveries;
          std::copy_n(samples.data() + size_t(order[recovery++]) * config.dim,
                      config.dim, next.data() + size_t(c) * config.dim);
        } else {
          for (uint32_t d = 0; d < config.dim; ++d)
            next[size_t(c) * config.dim + d] =
                float(sums[size_t(c) * config.dim + d] / result.occupancy[c]);
        }
      }
      result.centroids.swap(next);
      result.passes.back().mean_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        mean_start)
              .count();
      if (count >= 1000000)
        std::clog << "[IVF training] pass=" << iteration + 1
                  << " phase=means_complete mean_s="
                  << result.passes.back().mean_seconds << std::endl;
    }
    assign();  // Occupancy and objective must describe the returned centroids.
    std::fill(result.occupancy.begin(), result.occupancy.end(), 0);
    for (uint32_t r = 0; r < count; ++r) {
      ++result.occupancy[assignment[r]];
      result.final_objective += distances[r];
    }
    result.passes.back().assigned_sse = result.final_objective;
    result.empty_sample_clusters = uint32_t(
        std::count(result.occupancy.begin(), result.occupancy.end(), 0));
    result.largest_sample_cluster =
        *std::max_element(result.occupancy.begin(), result.occupancy.end());
    if (index)
      result.final_unreachable = index->unreachable();
    result.assignments = std::move(assignment);
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return result;
  }
}  // namespace diskann::ivf
