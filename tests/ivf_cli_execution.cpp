#include "ivf_cli_execution.h"
#include "bufann/ivf_index.h"
#include "ivf_mixed_workload.h"
#include "ivf_training_json.h"
#include "ivf_driver_progress.h"
#include "ivf_insert_workload.h"
#include "ivf_insert_input.h"
#include "ivf_query_warmup.h"
#include <omp.h>
#include <mkl_service.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>
#include <sys/resource.h>

namespace ivf_cli {
  namespace {
    using namespace diskann::ivf;
    using Clock = std::chrono::steady_clock;
    void check(bool ok, const char* message) {
      if (!ok)
        throw std::invalid_argument(message);
    }
    double seconds(Clock::time_point start) {
      return std::chrono::duration<double>(Clock::now() - start).count();
    }
    uint64_t budget_bytes(double gb) {
      const long double bytes = static_cast<long double>(gb) * (1ull << 30);
      check(gb > 0 && bytes < static_cast<long double>(UINT64_MAX),
            "Invalid memory budget");
      return static_cast<uint64_t>(bytes);
    }
    void add(uint64_t& bytes, uint64_t count, uint64_t stride = 1) {
      check(!stride || count <= (UINT64_MAX - bytes) / stride,
            "CLI workspace overflow");
      bytes += count * stride;
      check(bytes <= SIZE_MAX, "CLI workspace exceeds size_t");
    }
    template<typename T>
    void read_values(std::ifstream& in, std::vector<T>& values,
                     uint64_t count) {
      check(count <= SIZE_MAX / sizeof(T) &&
                count <= uint64_t(INT64_MAX) / sizeof(T),
            "Input payload too large");
      values.resize(static_cast<size_t>(count));
      in.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
      check(bool(in), "Input payload read failed");
    }
    double peak_process_rss_mb() {
      rusage usage{};
      check(getrusage(RUSAGE_SELF, &usage) == 0, "Cannot read process RSS");
      return double(usage.ru_maxrss) /
             1024.0;  // Linux KiB; lifetime high-water mark.
    }
    void begin_record(std::ostream& out, const char* phase, DataType type) {
      out << std::setprecision(12) << "{\"baseline\":\"IVF\",\"phase\":\""
          << phase << "\",\"data_type\":\"" << data_type_name(type)
          << "\",\"metric\":\"l2\"";
    }
    void validate_output_paths(const Arguments& args) {
      for (const auto* key : {"--result_file", "--search_result_file"})
        if (args.has(key))
          check(!std::filesystem::exists(
                    std::filesystem::symlink_status(args.require(key))),
                "Output file already exists");
      if (args.has("--result_file") && args.has("--search_result_file"))
        check(
            std::filesystem::absolute(args.require("--result_file"))
                    .lexically_normal() !=
                std::filesystem::absolute(args.require("--search_result_file"))
                    .lexically_normal(),
            "Output paths must differ");
    }
    void publish_record(const Arguments& args, const std::string& record) {
      if (args.has("--result_file")) {
        claim_artifact(args.require("--result_file"));
        std::ofstream out(args.require("--result_file"),
                          std::ios::binary | std::ios::trunc);
        out << record << '\n';
        out.close();
        check(bool(out), "Metrics file write failed");
      }
      std::cout << record << std::endl;
    }
    RouterOptions build_routing_options(const Arguments& args,
                                        uint64_t         budget) {
      RouterOptions routing;
      routing.degree = args.u32("--router_degree", 16);
      routing.construction_breadth = args.u32("--construction_breadth", 200);
      routing.query_breadth = args.u32("--query_breadth", 128);
      routing.insert_breadth = args.u32("--insert_breadth", 64);
      routing.seed = args.u32("--router_seed", 100);
      routing.memory_budget_bytes = budget;
      const auto native = args.u32("--optimized_modes", 0);
      check(native <= 1, "optimized_modes must be 0 or 1");
      routing.native_modes = native != 0;
      routing.mode = parse_router_mode(args.get("--router_mode", "hnswlib"));
      routing.refine_breadth = args.u32("--refine_ef", 128);
      routing.pca_variance = args.real("--pca_variance", .95);
      routing.router_rerank_factor = args.real("--router_rerank_factor", 3);
      routing.native_work_limit =
          args.integer("--routing_work_limit", 1000000000);
      return routing;
    }
    RoutingRequest query_routing_request(const Arguments&     args,
                                         const RouterOptions& defaults) {
      const auto name = args.get("--router_mode", "default");
      // Explicit "default" resolves to the frozen producer's mode, like
      // omission.
      const auto mode =
          name == "default" ? defaults.mode : parse_router_mode(name);
      RoutingRequest routing{
          mode,
          args.real("--router_rerank_factor", defaults.router_rerank_factor),
          args.u32("--router_breadth", defaults.query_breadth)};
      check(routing.rerank_factor >= 1 && routing.breadth > 0 &&
                routing.breadth <= INT32_MAX &&
                (defaults.native_modes || routing.mode == RouterMode::Exact ||
                 routing.mode == RouterMode::Hnswlib),
            "Query router settings invalid or native modes not built");
      return routing;
    }
    template<typename T>
    void build_typed(const Arguments& args, const BuildConfig& config,
                     const BinMetadata& vectors, const PQMetadata& pq) {
      validate_output_paths(args);
      const auto    budget = budget_bytes(config.memory_budget_gb);
      HeapOptions   heap{config.dim, args.u32("--id_capacity", vectors.rows),
                       args.u32("--page_size", 4096),
                       args.u32("--buffer_frames", 64)};
      const auto    routing = build_routing_options(args, budget);
      const auto    start = Clock::now();
      TrainingStats training_stats;
      NativeIndex<T>::build(args.require("--index_prefix"),
                            args.require("--data_file"),
                            args.require("--pq_prefix") + "_pq_pivots.bin",
                            args.require("--pq_prefix") + "_pq_compressed.bin",
                            config, heap, routing, 0, &training_stats);
      auto router =
          CentroidRouter::load(args.require("--index_prefix"), budget);
      const auto         elapsed = seconds(start);
      const auto         peak_rss = peak_process_rss_mb();
      std::ostringstream record;
      begin_record(record, "build", config.data_type);
      bool first = false;
      auto emit = [&](const char* key, const auto& value) {
        bufann_bench::json_kv(record, first, key, value);
      };
      emit("workload", "build");
      emit("index_prefix", args.require("--index_prefix"));
      emit("dim", config.dim);
      emit("source_rows", vectors.rows);
      emit("base_count", config.base_count);
      emit("nlist", config.nlist);
      emit("pq_chunks", pq.chunks);
      emit("build_threads", config.build_threads);
      ListBuildOptions list_options;
      list_options.base_count = config.base_count;
      list_options.build_threads = config.build_threads;
      const auto list_workers = PostingLists::routing_workers(list_options);
      emit("list_routing_workers", list_workers);
      emit("list_routing_block_rows", list_options.routing_block_rows);
      emit("list_routing_buffer_rows",
           std::min<uint64_t>(
               config.base_count,
               uint64_t(list_workers) * list_options.routing_block_rows));
      emit("sample_rate", config.sample_rate);
      emit("max_reps", config.max_reps);
      emit("seed", config.seed);
      emit("id_capacity", heap.id_capacity);
      emit("page_size", heap.page_size);
      emit("buffer_frames", heap.buffer_frames);
      emit("router_degree", routing.degree);
      emit("construction_breadth", routing.construction_breadth);
      emit("query_breadth", routing.query_breadth);
      emit("insert_breadth", routing.insert_breadth);
      emit("router_seed", routing.seed);
      emit("router_mode", router_mode_name(routing.mode));
      emit("assignment_router_mode", router_mode_name(routing.mode));
      emit("optimized_modes_built", routing.native_modes);
      emit("pca_retained_dim", router->projected_dim());
      emit("pca_retained_variance", router->retained_variance());
      emit("pca_degenerate", router->pca_degenerate());
      emit("pca_variance_target", routing.pca_variance);
      emit("router_rerank_factor", routing.router_rerank_factor);
      emit("refine_ef", routing.refine_breadth);
      emit("routing_work_limit", routing.native_work_limit);
      emit("patched_unreachable_before", router->unreachable_before(false));
      emit("refined_unreachable_before", router->unreachable_before(true));
      emit("isolated_reinserted_nodes", router->reinserted_nodes(true));
      if (routing.native_modes) {
        emit("patched_unreachable_after", router->unreachable_after(false));
        emit("refined_unreachable_after", router->unreachable_after(true));
        emit("patched_zero_indegree", router->zero_indegree(false));
        emit("refined_zero_indegree", router->zero_indegree(true));
      }
      emit("build_time_s", elapsed);
      emit("peak_process_rss_mb", peak_rss);
      emit("memory_budget_bytes", budget);
      emit("training_assignment", trainer_mode_name(config.trainer_mode));
      emit("training_sample_count", config.training_count());
      training_settings(record, first, config);
      training_metrics(record, first, training_stats);
      emit("pq_policy", "global_nonresidual_pq8_external_unverified");
      emit("publication", "manifest_last_nonatomic_no_fsync");
      record << "}";
      publish_record(args, record.str());
    }

    struct WorkerStats {
      uint64_t scanned = 0, reranked = 0, stages = 0, batches = 0, pages = 0,
               short_rows = 0;
      double             routing = 0, adc = 0, scan = 0, fetch = 0, rerank = 0;
      double             projection = 0, graph = 0, centroid_rerank = 0;
      double             fetch_adapter_wait = 0, fetch_backend_pin = 0;
      double             fetch_pin_admission_wait = 0;
      uint64_t           router_distances = 0, router_counted_queries = 0;
      std::exception_ptr failure;
    };

    void record_query(WorkerStats& stats, const QueryResult& result,
                      uint32_t recall_at) {
      stats.scanned += result.scanned_codes;
      stats.reranked += result.reranked_vectors;
      stats.stages += result.logical_fetch_stages;
      stats.batches += result.backend_batch_calls;
      stats.pages += result.heap_pages_read;
      stats.short_rows += result.hits.size() < recall_at;
      stats.routing += result.routing_us;
      stats.adc += result.adc_us;
      stats.scan += result.scan_select_us;
      stats.fetch += result.fetch_us;
      stats.fetch_adapter_wait +=
          double(result.fetch_adapter_mutex_wait_ns) / 1000;
      stats.fetch_backend_pin += double(result.fetch_backend_pin_ns) / 1000;
      stats.fetch_pin_admission_wait +=
          double(result.fetch_pin_admission_wait_ns) / 1000;
      stats.rerank += result.rerank_us;
      stats.projection += result.router_stats.projection_us;
      stats.graph += result.router_stats.graph_us;
      stats.centroid_rerank += result.router_stats.centroid_rerank_us;
      stats.router_distances += result.router_stats.distance_evaluations;
      stats.router_counted_queries +=
          result.router_stats.distance_count_available;
    }

    WorkerStats sum_workers(std::span<const WorkerStats> workers) {
      WorkerStats total;
      for (const auto& stats : workers) {
        if (stats.failure)
          std::rethrow_exception(stats.failure);
        total.scanned += stats.scanned;
        total.reranked += stats.reranked;
        total.stages += stats.stages;
        total.batches += stats.batches;
        total.pages += stats.pages;
        total.short_rows += stats.short_rows;
        total.routing += stats.routing;
        total.adc += stats.adc;
        total.scan += stats.scan;
        total.fetch += stats.fetch;
        total.fetch_adapter_wait += stats.fetch_adapter_wait;
        total.fetch_backend_pin += stats.fetch_backend_pin;
        total.fetch_pin_admission_wait += stats.fetch_pin_admission_wait;
        total.rerank += stats.rerank;
        total.projection += stats.projection;
        total.graph += stats.graph;
        total.centroid_rerank += stats.centroid_rerank;
        total.router_distances += stats.router_distances;
        total.router_counted_queries += stats.router_counted_queries;
      }
      return total;
    }

    struct GroundTruth {
      std::vector<uint32_t> ids, row;
      std::vector<float>    distances;
      uint32_t              columns = 0;
    };

    GroundTruth load_truth(const std::string& path, const BinMetadata& shape) {
      GroundTruth truth;
      truth.columns = shape.cols;
      std::ifstream input(path, std::ios::binary);
      input.seekg(8);
      const uint64_t entries = uint64_t(shape.rows) * shape.cols;
      read_values(input, truth.ids, entries);
      read_values(input, truth.distances, entries);
      if (std::filesystem::file_size(path) == 8 + entries * 12) {
        std::array<uint32_t, 4096> tags;
        for (uint64_t offset = 0; offset < entries; offset += tags.size()) {
          const auto count = static_cast<size_t>(
              std::min<uint64_t>(tags.size(), entries - offset));
          input.read(reinterpret_cast<char*>(tags.data()),
                     count * sizeof(uint32_t));
          check(bool(input), "GT tags read failed");
          check(std::equal(tags.begin(), tags.begin() + count,
                           truth.ids.begin() + offset),
                "IVF GT tags must equal source row IDs");
        }
      }
      truth.row.reserve(shape.cols);
      for (uint32_t q = 0; q < shape.rows; ++q) {
        const auto offset = size_t(q) * shape.cols;
        for (uint32_t j = 0; j < shape.cols; ++j) {
          const auto distance = truth.distances[offset + j];
          check(std::isfinite(distance) && distance >= 0 &&
                    (!j || distance >= truth.distances[offset + j - 1]),
                "GT distances must be finite, nonnegative and sorted");
        }
        truth.row.assign(truth.ids.begin() + offset,
                         truth.ids.begin() + offset + shape.cols);
        std::sort(truth.row.begin(), truth.row.end());
        check(std::adjacent_find(truth.row.begin(), truth.row.end()) ==
                  truth.row.end(),
              "Duplicate GT ID in row");
      }
      return truth;
    }

    double score_recall(GroundTruth& truth, std::span<const uint32_t> ids,
                        uint32_t rows, uint32_t recall_at) {
      uint64_t matches = 0;
      for (uint32_t q = 0; q < rows; ++q) {
        const auto offset = size_t(q) * truth.columns;
        uint32_t   end = recall_at;
        while (end < truth.columns &&
               truth.distances[offset + end] ==
                   truth.distances[offset + recall_at - 1])
          ++end;
        truth.row.assign(truth.ids.begin() + offset,
                         truth.ids.begin() + offset + end);
        std::sort(truth.row.begin(), truth.row.end());
        for (uint32_t j = 0; j < recall_at; ++j) {
          const auto id = ids[size_t(q) * recall_at + j];
          if (id != UINT32_MAX &&
              std::binary_search(truth.row.begin(), truth.row.end(), id))
            ++matches;
        }
      }
      return 100.0 * double(matches) / double(ids.size());
    }

    void write_neighbors(const std::string& path, uint32_t rows,
                         uint32_t recall_at, std::span<const uint32_t> ids,
                         std::span<const float> distances) {
      claim_artifact(path);
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<const char*>(&rows), 4);
      out.write(reinterpret_cast<const char*>(&recall_at), 4);
      // SPFresh results interleave row-major ID/distance pairs. GT uses
      // separate planes.
      for (size_t i = 0; i < ids.size(); ++i) {
        out.write(reinterpret_cast<const char*>(&ids[i]), 4);
        out.write(reinterpret_cast<const char*>(&distances[i]), 4);
      }
      out.close();
      check(bool(out), "Result file write failed");
    }

    // Sequential foreground updates, distinct from concurrent mixed scheduling.
    // Full-source identity prevents changing row-ID meaning between build and
    // insertion. Partial failures keep owned files but publish neither a
    // checkpoint nor a success record.
    struct UpdateStats {
      uint64_t       inserted = 0, deleted = 0, missing = 0, bytes = 0;
      double         insert_s = 0, delete_s = 0, checkpoint_s = 0;
      double         source_before_s = 0, input_validation_s = 0;
      double         coordinate_validation_s = 0, source_after_s = 0;
      HeapIOCounters foreground_io, checkpoint_io;
      uint32_t       insert_threads_requested = 0, insert_threads_effective = 0;
      uint32_t insert_peak_active = 0, insert_runtime_threads_before_launch = 0;
      uint64_t insert_worker_allowance_bytes = 0;
      uint64_t insert_list_growth_allowance_bytes = 0;
      uint64_t insert_staged_bytes = 0;
      InsertInputPolicy input_policy = InsertInputPolicy::Streaming;
    };
    void heap_progress(std::ostream& out, const HeapIOCounters& io) {
      out << " heap_page_reads=" << io.page_reads
          << " heap_page_writes=" << io.page_writes
          << " dirty_evictions=" << io.dirty_evictions
          << " heap_flush_bytes=" << io.flush_bytes
          << " heap_read_wait_ns=" << io.read_wait_ns
          << " heap_eviction_write_ns=" << io.eviction_write_ns;
    }
    template<typename T>
    void checkpoint_update(const Arguments& args, MutableIndex<T>& index,
                           UpdateStats& s) {
      const auto     checkpoint_before = index.heap_io_counters();
      DriverProgress checkpoint(
          "checkpoint_flush_copy_verify", std::nullopt, "operations",
          [&](std::ostream& out) {
            heap_progress(out,
                          index.heap_io_counters().since(checkpoint_before));
          });
      const auto start = Clock::now();
      index.checkpoint(args.require("--checkpoint_prefix"),
                       index.estimated_checkpoint_bytes());
      s.checkpoint_s = seconds(start);
      s.checkpoint_io = index.heap_io_counters().since(checkpoint_before);
      checkpoint.finish();
    }
    template<typename T>
    UpdateStats mixed_index(const Arguments& args, MutableIndex<T>& index,
                            uint64_t budget, uint64_t resident_bytes,
                            std::span<const T>  queries,
                            const SearchConfig& config, MixedStats& mixed,
                            RoutingRequest routing) {
      const auto& m = index.origin_manifest();
      args.reject_present({"--pq_prefix"},
                          "mixed updates use frozen bundle PQ");
      const auto     source = args.require("--full_data_file");
      DriverProgress source_before("source_identity_before");
      check(
          inspect_vectors(source, m.training.data_type, m.training.dim).rows ==
                  m.source_rows &&
              artifact_digest(source) == m.sources[0],
          "Mixed full-source identity mismatch");
      const auto     source_before_s = source_before.finish();
      DriverProgress input_validation("update_input_validation");
      const auto     scheduler = args.get("--scheduler", "weighted");
      check(scheduler == "weighted" || scheduler == "fixed_8_8" ||
                scheduler == "update_driven",
            "Unknown mixed scheduler");
      MixedPlan plan;
      plan.fixed_8_8 = scheduler == "fixed_8_8";
      plan.update_driven = scheduler == "update_driven";
      plan.pool_threads = args.u32("--pool_threads", 16);
      if (plan.update_driven)
        args.reject_present(
            {"--query_ratio", "--query_count", "--scheduler_seed"},
            "update_driven queries run until mutations complete");
      plan.seed = args.integer("--scheduler_seed", plan.seed);
      UpdateStats s;
      s.source_before_s = source_before_s;
      s.bytes = resident_bytes;
      for (auto key : {"--insert_ids_file", "--delete_ids_file"})
        add(s.bytes, std::filesystem::file_size(args.require(key)), 16);
      check(s.bytes <= budget, "CLI mixed ID parsing exceeds allowance");
      const auto inserts =
          load_counted_ids(args.require("--insert_ids_file"), m.source_rows);
      const auto deletes =
          load_counted_ids(args.require("--delete_ids_file"), m.source_rows);
      plan.inserts = inserts.size();
      plan.deletes = deletes.size();
      // Validate bounds BEFORE calling the legacy count helper's
      // multiplication.
      check(plan.inserts + plan.deletes <= UINT64_MAX / 99,
            "Mixed query-count overflow");
      if (!plan.update_driven) {
        const auto ratio = args.u32("--query_ratio", 80);
        check(ratio > 0 && ratio < 100, "Mixed query_ratio must be in [1,99]");
        plan.queries = args.integer("--query_count",
                                    ann_bench::mixed_update_query_count(
                                        inserts.size(), deletes.size(), ratio));
      }
      plan.validate();
      auto sorted = deletes;
      std::sort(sorted.begin(), sorted.end());
      for (auto id : inserts)
        check(id >= m.first_id && id < m.heap.id_capacity &&
                  !index.contains(id) &&
                  !std::binary_search(sorted.begin(), sorted.end(), id),
              "Invalid/live/overlapping mixed insert ID");
      for (auto id : deletes)
        check(id >= m.first_id && id < m.heap.id_capacity && index.contains(id),
              "Mixed deletion must refer to a live ID");
      const auto initial_live = index.size();
      const auto retained_before = index.estimated_retained_bytes();
      const auto query_bytes = index.estimated_query_bytes(config),
                 mutation_bytes = index.estimated_mutation_bytes();
      add(s.bytes, plan.inserts * m.training.dim, sizeof(T));
      add(s.bytes, plan.latency_capacity(), 2 * sizeof(double));
      uint64_t worker_bytes = 8ull * 1024 * 1024;
      add(worker_bytes, query_bytes);
      add(worker_bytes, mutation_bytes);
      add(s.bytes, plan.pool_threads, worker_bytes);
      add(s.bytes, index.estimated_checkpoint_bytes());
      add(s.bytes, 1024 * 1024);
      check(s.bytes <= budget, "CLI composed mixed workload exceeds allowance");
      s.input_validation_s = input_validation.finish();
      DriverProgress staging("coordinate_staging", plan.inserts, "staged_rows");
      std::vector<T> coordinates(size_t(plan.inserts) * m.training.dim);
      std::ifstream  input(source, std::ios::binary);
      for (size_t i = 0; i < inserts.size(); ++i) {
        input.seekg(8 + uint64_t(inserts[i]) * m.training.dim * sizeof(T));
        input.read(
            reinterpret_cast<char*>(coordinates.data() + i * m.training.dim),
            size_t(m.training.dim) * sizeof(T));
        check(bool(input), "Mixed source row read failed");
        staging.observe(i + 1);
      }
      for (auto v : coordinates)
        check(std::isfinite(double(v)), "Nonfinite mixed insert coordinate");
      s.coordinate_validation_s = staging.finish();
      const auto query_rows = queries.size() / m.training.dim;
      check(query_rows != 0, "Mixed queries require a nonempty source");
      std::atomic<uint64_t> missing{0};
      const auto            runtime_threads = uint32_t(
          std::distance(std::filesystem::directory_iterator("/proc/self/task"),
                                   std::filesystem::directory_iterator{}));
      const auto     io_before = index.heap_io_counters();
      DriverProgress foreground(
          "mixed_foreground", std::nullopt, "operations",
          [&](std::ostream& out) {
            heap_progress(out, index.heap_io_counters().since(io_before));
          });
      mixed = run_mixed_round(
          plan, {[&](uint64_t i) {
                   index.insert(
                       inserts[size_t(i)],
                       {coordinates.data() + size_t(i) * m.training.dim,
                        m.training.dim},
                       {mutation_bytes, false});
                 },
                 [&](uint64_t i) {
                   if (!index.erase(deletes[size_t(i)]))
                     ++missing;
                 },
                 [&](uint64_t i) {
                   (void) index.query(
                       queries.subspan(size_t(i % query_rows) * m.training.dim,
                                       m.training.dim),
                       config, {query_bytes, false, routing});
                 }});
      mixed.runtime_threads_before_launch = runtime_threads;
      s.foreground_io = index.heap_io_counters().since(io_before);
      foreground.finish();
      check(missing == 0 && index.size() == uint64_t(initial_live) +
                                                plan.inserts - plan.deletes,
            "Mixed round count/live set mismatch");
      for (auto id : inserts)
        check(index.contains(id), "Completed mixed insert missing at barrier");
      for (auto id : deletes)
        check(!index.contains(id), "Completed mixed delete live at barrier");
      DriverProgress source_after("source_identity_after");
      check(artifact_digest(source) == m.sources[0],
            "Mixed source changed during execution");
      s.source_after_s = source_after.finish();
      s.inserted = mixed.completed[0];
      s.deleted = mixed.completed[1];
      s.insert_s = s.delete_s =
          mixed.wall_s;  // Rates use full mixed foreground wall, not summed
                         // callback time.
      add(s.bytes, index.estimated_retained_bytes() - retained_before);
      check(s.bytes <= budget,
            "CLI post-growth mixed workload exceeds allowance");
      checkpoint_update(args, index, s);
      return s;
    }
    void validate_driver_paths(const Arguments& args, bool updating) {
      validate_output_paths(args);
      std::vector<std::string> paths;
      for (auto key : {"--result_file", "--search_result_file"})
        if (args.has(key))
          paths.push_back(args.require(key));
      if (args.has("--private_heap_path")) {
        paths.push_back(args.require("--private_heap_path"));
        paths.push_back(args.require("--private_heap_path") + ".pq.bin");
      }
      if (updating) {
        const auto prefix = args.require("--checkpoint_prefix");
        paths.push_back(prefix + ".manifest");
        for (auto suffix : mutable_component_suffixes)
          paths.push_back(prefix + suffix);
        for (auto suffix : native_router_suffixes)
          paths.push_back(prefix + suffix);
        args.require("--private_heap_path");
      } else
        args.reject_present({"--checkpoint_prefix"},
                            "not used by search-only execution");
      std::vector<std::filesystem::path> normalized;
      for (const auto& path : paths) {
        check(!std::filesystem::exists(std::filesystem::symlink_status(path)),
              "Driver output artifact exists");
        const auto resolved =
            std::filesystem::weakly_canonical(std::filesystem::absolute(path));
        check(std::find(normalized.begin(), normalized.end(), resolved) ==
                  normalized.end(),
              "Driver output paths overlap");
        normalized.push_back(resolved);
      }
    }
    template<typename T>
    std::unique_ptr<MutableIndex<T>> open_mutable(const Arguments& args,
                                                  uint64_t         budget) {
      const MutableIndexOptions options{args.u32("--buffer_frames", 64),
                                        budget};
      const auto format = args.get("--index_format", "immutable");
      check(format == "immutable" || format == "mutable",
            "Unknown index_format");
      return format == "mutable"
                 ? MutableIndex<T>::reopen(args.require("--index_prefix"),
                                           args.require("--private_heap_path"),
                                           options)
                 : MutableIndex<T>::fork(args.require("--index_prefix"),
                                         args.require("--private_heap_path"),
                                         options);
    }
    template<typename T>
    const IndexManifest& cli_manifest(const NativeIndex<T>& index) {
      return index.manifest();
    }
    template<typename T>
    const IndexManifest& cli_manifest(const MutableIndex<T>& index) {
      return index.origin_manifest();
    }
    template<typename T>
    uint64_t cli_retained(const NativeIndex<T>& index) {
      return index.estimated_open_bytes();
    }
    template<typename T>
    uint64_t cli_retained(const MutableIndex<T>& index) {
      return index.estimated_retained_bytes();
    }
    template<typename T>
    UpdateStats update_index(const Arguments& args, MutableIndex<T>& index,
                             uint64_t budget, uint64_t resident_bytes) {
      const auto& m = index.origin_manifest();
      const auto  workload = args.require("--workload");
      const bool  insert = workload != "delete_only",
                 erase = workload != "insert_only";
      args.reject_present({"--pq_prefix"},
                          "frozen bundle PQ is used for updates");
      if (!insert)
        args.reject_present({"--insert_ids_file"}, "not used by delete_only");
      if (!erase)
        args.reject_present({"--delete_ids_file"}, "not used by insert_only");
      const auto     source = args.require("--full_data_file");
      DriverProgress source_before("source_identity_before");
      const auto     shape =
          inspect_vectors(source, m.training.data_type, m.training.dim);
      check(shape.rows == m.source_rows &&
                artifact_digest(source) == m.sources[0],
            "Update full-source identity mismatch");
      const auto     source_before_s = source_before.finish();
      DriverProgress input_validation("update_input_validation");
      // Counted-ID parser temporarily uses an unordered_set as well as the
      // vector. Conservatively budget both ID files before allocating either
      // parser.
      UpdateStats s;
      s.source_before_s = source_before_s;
      s.bytes = resident_bytes;
      const auto retained_before = index.estimated_retained_bytes();
      s.input_policy = parse_insert_input_policy(
          args.get("--insert_input_policy", "streaming"));
      for (auto key : {"--insert_ids_file", "--delete_ids_file"})
        if (args.has(key))
          add(s.bytes, std::filesystem::file_size(args.require(key)), 16);
      add(s.bytes, m.training.dim, sizeof(T));
      add(s.bytes, index.estimated_mutation_bytes());
      add(s.bytes, index.estimated_checkpoint_bytes());
      add(s.bytes, 1024 * 1024);
      check(s.bytes <= budget,
            "CLI composed update exceeds workspace allowance");
      auto inserts = insert
                         ? load_counted_ids(args.require("--insert_ids_file"),
                                            m.source_rows)
                         : std::vector<uint32_t>{};
      auto deletes = erase ? load_counted_ids(args.require("--delete_ids_file"),
                                              m.source_rows)
                           : std::vector<uint32_t>{};
      if (s.input_policy == InsertInputPolicy::Staged) {
        s.insert_staged_bytes =
            insert_staging_bytes(inserts.size(), m.training.dim, sizeof(T));
        add(s.bytes, s.insert_staged_bytes);
        if (s.insert_staged_bytes)
          add(s.bytes, 64);  // Staged allocation metadata/alignment allowance.
        check(s.bytes <= budget,
              "CLI insertion staging exceeds workspace allowance");
      }
      s.insert_threads_requested = insert ? args.u32("--insert_threads", 1) : 0;
      s.insert_threads_effective =
          insert ? effective_insert_workers(inserts.size(),
                                            s.insert_threads_requested)
                 : 0;
      // The original allowance contains one row/mutation workspace. Admit
      // every extra independent reader/row/router/PQ workspace before launch.
      uint64_t per_worker = index.estimated_mutation_bytes();
      add(per_worker, m.training.dim, sizeof(T));
      add(s.insert_worker_allowance_bytes, s.insert_threads_effective, 65536);
      if (s.insert_threads_effective > 1) {
        add(s.insert_worker_allowance_bytes, s.insert_threads_effective - 1,
            per_worker);
        rlimit     stack{};
        const auto stack_bytes =
            getrlimit(RLIMIT_STACK, &stack) == 0 &&
                    stack.rlim_cur != RLIM_INFINITY
                ? std::max<uint64_t>(8ull * 1024 * 1024, stack.rlim_cur)
                : 8ull * 1024 * 1024;
        add(s.insert_worker_allowance_bytes, s.insert_threads_effective,
            stack_bytes);
      }
      add(s.bytes, s.insert_worker_allowance_bytes);
      // Worst-case every input crosses a +16 boundary. Include replacement
      // allocator allowance and old-list storage while growth copies run on
      // distinct locked lists; retained_before conservatively covers all old
      // list buffers, not just one worker's chosen list.
      if (!inserts.empty()) {
        add(s.insert_list_growth_allowance_bytes, inserts.size(),
            16 * (uint64_t(m.chunks) + sizeof(uint32_t)) + 128);
        add(s.insert_list_growth_allowance_bytes, retained_before);
        add(s.bytes, s.insert_list_growth_allowance_bytes);
      }
      check(s.bytes <= budget,
            s.input_policy == InsertInputPolicy::Staged
                ? "CLI staged insertion exceeds workspace allowance"
                : "CLI insertion workers exceed workspace allowance");
      for (auto id : inserts)
        check(
            id >= m.first_id && id < m.heap.id_capacity && !index.contains(id),
            "Insert ID is outside allowed range or already live");
      for (auto id : deletes)
        check(id >= m.first_id && id < m.heap.id_capacity,
              "Delete ID outside allowed range");
      auto sorted_deletes = deletes;
      std::sort(sorted_deletes.begin(), sorted_deletes.end());
      for (auto id : inserts)
        check(!std::binary_search(sorted_deletes.begin(), sorted_deletes.end(),
                                  id),
              "Insert/delete ID sets overlap");
      s.input_validation_s = input_validation.finish();
      const auto    identity = m.sources[0];
      std::ifstream input(source, std::ios::binary);
      auto          read_row = [&](uint32_t id, std::span<T> row) {
        input.seekg(8 + uint64_t(id) * m.training.dim * sizeof(T));
        input.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
        check(bool(input), "Insert source row read failed");
      };
      // Validation/staging is outside the insertion timer. Staged rows are
      // immutable worker inputs; streaming performs its second read inside it.
      DriverProgress coordinates(s.input_policy == InsertInputPolicy::Staged
                                     ? "coordinate_staging"
                                     : "coordinate_validation",
                                 inserts.size(), "validated_rows");
      std::vector<T> staged;
      if (s.input_policy == InsertInputPolicy::Staged) {
        staged = stage_insert_rows<T>(
            inserts, m.training.dim, s.insert_staged_bytes, read_row,
            [&](uint64_t completed) { coordinates.observe(completed); });
      } else {
        std::vector<T> row(m.training.dim);
        uint64_t       validated = 0;
        for (auto id : inserts) {
          read_row(id, row);
          for (auto value : row)
            check(std::isfinite(double(value)), "Nonfinite insert coordinate");
          coordinates.observe(++validated);
        }
      }
      s.coordinate_validation_s = coordinates.finish();
      input.close();
      // Open and allocate all independent inputs before delete/insert side
      // effects; coordinate preflight above covers every requested row.
      const auto readers = s.input_policy == InsertInputPolicy::Streaming
                               ? s.insert_threads_effective
                               : 0;
      std::vector<std::ifstream>  inputs(readers);
      std::vector<std::vector<T>> rows(readers);
      for (uint32_t w = 0; w < readers; ++w) {
        inputs[w].open(source, std::ios::binary);
        check(bool(inputs[w]), "Insertion worker source open failed");
        rows[w].resize(m.training.dim);
      }
      const auto     io_before = index.heap_io_counters();
      DriverProgress deletion(
          "delete", deletes.size(), "processed_delete_ids",
          [&](std::ostream& out) {
            heap_progress(out, index.heap_io_counters().since(io_before));
          });
      auto start = Clock::now();
      for (auto id : deletes) {
        if (index.erase(id))
          ++s.deleted;
        else
          ++s.missing;
        deletion.observe(s.deleted + s.missing);
      }
      s.delete_s = seconds(start);
      deletion.finish();
      const auto     insert_before = index.heap_io_counters();
      DriverProgress insertion(
          "insert", inserts.size(), "published_inserts",
          [&](std::ostream& out) {
            heap_progress(out, index.heap_io_counters().since(insert_before));
          });
      const auto mutation_bytes = index.estimated_mutation_bytes();
      s.insert_runtime_threads_before_launch = uint32_t(
          std::distance(std::filesystem::directory_iterator("/proc/self/task"),
                        std::filesystem::directory_iterator{}));
      start = Clock::now();
      const auto executed = run_insert_round(
          inserts.size(), insert ? s.insert_threads_requested : 1,
          [&](uint32_t w, uint64_t position) {
            const auto id = inserts[size_t(position)];
            if (s.input_policy == InsertInputPolicy::Staged) {
              index.insert(
                  id,
                  std::span<const T>(staged).subspan(
                      size_t(position) * m.training.dim, m.training.dim),
                  {mutation_bytes, false});
              return;
            }
            auto& in = inputs[w];
            auto& values = rows[w];
            in.seekg(8 + uint64_t(id) * m.training.dim * sizeof(T));
            in.read(reinterpret_cast<char*>(values.data()),
                    values.size() * sizeof(T));
            check(bool(in), "Insertion worker source row read failed");
            index.insert(id, values, {mutation_bytes, false});
          },
          [&](uint64_t completed) { insertion.observe(completed); });
      s.inserted = executed.completed;
      s.insert_peak_active = executed.peak_active;
      s.insert_s = seconds(start);
      s.foreground_io = index.heap_io_counters().since(io_before);
      insertion.finish();
      DriverProgress source_after("source_identity_after");
      check(artifact_digest(source) == identity,
            "Update source changed during execution");
      s.source_after_s = source_after.finish();
      // Growth may consume retained list allowance; verify final composed
      // budget too.
      const auto growth = index.estimated_retained_bytes() - retained_before;
      check(growth <= s.insert_list_growth_allowance_bytes,
            "CLI insertion exceeded admitted retained growth allowance");
      check(s.bytes <= budget,
            "CLI post-growth update exceeds workspace allowance");
      checkpoint_update(args, index, s);
      return s;
    }
    template<typename T>
    void emit_update(std::ostream& record, bool& first, const Arguments& args,
                     const MutableIndex<T>& index, const UpdateStats& s,
                     const MixedStats* mixed = nullptr) {
      auto emit = [&](const char* key, const auto& value) {
        bufann_bench::json_kv(record, first, key, value);
      };
      emit("insert_count", s.inserted);
      emit("delete_count", s.deleted);
      emit("missing_delete_count", s.missing);
      emit("insert_time_s", s.insert_s);
      emit("delete_time_s", s.delete_s);
      emit("insert_qps", bufann_bench::safe_div(s.inserted, s.insert_s));
      if (!mixed) {
        const bool staged = s.input_policy == InsertInputPolicy::Staged;
        emit("insert_input_policy", s.insert_threads_requested
                                        ? (staged ? "staged" : "streaming")
                                        : "not_applicable");
        emit("insert_timing_contract_version",
             s.insert_threads_requested ? (staged ? 2 : 1) : 0);
        emit("insert_staged_coordinate_bytes", s.insert_staged_bytes);
        emit("insert_input_preparation_time_s", s.coordinate_validation_s);
        emit("insert_input_preparation_time_scope",
             "alias_coordinate_validation_or_staging_time_not_additive");
        emit("insert_threads_requested", s.insert_threads_requested);
        emit("insert_threads_effective", s.insert_threads_effective);
        emit("insert_peak_active", s.insert_peak_active);
        emit("insert_runtime_threads_before_launch",
             s.insert_runtime_threads_before_launch);
        emit("insert_worker_allowance_bytes", s.insert_worker_allowance_bytes);
        emit("insert_list_growth_allowance_bytes",
             s.insert_list_growth_allowance_bytes);
        emit("insert_worker_policy",
             staged ? "bounded_joined_read_only_staged_coordinates_default_one"
                    : "bounded_joined_independent_readers_default_one");
        emit("insert_time_scope",
             staged
                 ? "completed_rows_mutation_progress_launch_join_no_source_"
                   "reads_no_gt"
                 : "completed_rows_reads_mutation_progress_launch_join_no_gt");
        emit("insert_order_policy",
             "parallel_publication_order_unspecified_distance_then_id_ties");
      }
      emit("delete_qps",
           bufann_bench::safe_div(s.deleted + s.missing, s.delete_s));
      emit("checkpoint_time_s", s.checkpoint_s);
      emit("source_identity_before_time_s", s.source_before_s);
      emit("update_input_validation_time_s", s.input_validation_s);
      emit("coordinate_validation_or_staging_time_s",
           s.coordinate_validation_s);
      emit("source_identity_after_time_s", s.source_after_s);
      emit("foreground_heap_page_reads", s.foreground_io.page_reads);
      emit("foreground_heap_page_writes", s.foreground_io.page_writes);
      emit("foreground_heap_dirty_evictions", s.foreground_io.dirty_evictions);
      emit("foreground_heap_flush_bytes", s.foreground_io.flush_bytes);
      emit("foreground_heap_read_wait_ns", s.foreground_io.read_wait_ns);
      emit("foreground_heap_eviction_write_ns",
           s.foreground_io.eviction_write_ns);
      emit("checkpoint_heap_page_writes", s.checkpoint_io.page_writes);
      emit("checkpoint_heap_flush_bytes", s.checkpoint_io.flush_bytes);
      emit("heap_io_counter_scope", mixed
                                        ? "backend_pool_phase_deltas_includes_"
                                          "concurrent_queries_not_device_io"
                                        : "backend_pool_phase_deltas_excludes_"
                                          "source_reads_not_device_io");
      emit("checkpoint_heap_io_scope",
           "pool_flush_only_excludes_checkpoint_file_copy_hash_metadata");
      emit("mutation_completion_policy",
           "joined_publications_then_quiescent_verified_checkpoint");
      emit("checkpoint_completion_scope",
           "flush_copy_hash_metadata_not_wal_or_crash_durability");
      emit("driver_progress_policy",
           "phase_boundaries_completed_counts_time_and_count_limited_no_"
           "heartbeat_thread");
      emit("driver_progress_timing_scope",
           "periodic_reporting_inside_foreground_boundary_reporting_outside");
      emit("checkpoint_prefix", args.require("--checkpoint_prefix"));
      emit("live_count", index.size());
      emit("retained_heap_records", index.retained_heap_records());
      emit("estimated_update_workspace_bytes", s.bytes);
      if (!mixed) {
        emit("mutation_workers", std::max(s.insert_threads_effective,
                                          s.deleted + s.missing ? 1u : 0u));
        emit("scheduler",
             s.insert_threads_effective > 1
                 ? "serial_delete_then_bounded_parallel_insert_no_query_overlap"
                 : "sequential_delete_then_insert_no_query_overlap");
        emit("mutation_timing_scope",
             "foreground_loops_excludes_open_validation_hash_checkpoint_and_"
             "post_update_query");
      } else {
        const auto fixed = args.get("--scheduler", "weighted") == "fixed_8_8";
        const auto driven =
            args.get("--scheduler", "weighted") == "update_driven";
        const auto threads = args.u32("--pool_threads", 16);
        emit("scheduler", driven  ? "update_driven"
                          : fixed ? "fixed_8_8"
                                  : "weighted_remaining_ops");
        emit("foreground_workers", threads);
        emit("mutation_workers", driven ? threads / 2 : fixed ? 8u : threads);
        emit("concurrent_query_workers", driven  ? threads - threads / 2
                                         : fixed ? 8u
                                                 : threads);
        emit("worker_allocation", driven  ? "dedicated_half_mutation_half_query"
                                  : fixed ? "dedicated_8_mutation_8_query"
                                          : "shared_pool_not_additive");
        if (!driven) {
          emit("scheduler_seed",
               args.integer("--scheduler_seed",
                            ann_bench::WeightedUpdatePool::kDefaultSeed));
          emit("query_ratio", args.u32("--query_ratio", 80));
        }
        emit("mixed_wall_s", mixed->wall_s);
        emit("concurrent_query_count_policy",
             driven ? "until_all_mutations_complete_join_admitted_queries"
             : args.has("--query_count")
                 ? "explicit_finite_count"
                 : "floor_ratio_times_updates_over_100_minus_ratio");
        emit("concurrent_query_count", mixed->completed[2]);
        emit("concurrent_query_qps",
             bufann_bench::safe_div(mixed->completed[2], mixed->wall_s));
        emit("overlapping_query_count", mixed->overlapping_queries);
        emit("peak_active_mutations", mixed->peak_mutations);
        emit("peak_active_queries", mixed->peak_queries);
        emit("insert_callback_us_total", mixed->callback_us[0]);
        emit("delete_callback_us_total", mixed->callback_us[1]);
        emit("concurrent_query_callback_us_total", mixed->callback_us[2]);
        auto latencies = mixed->query_latency_us;
        std::sort(latencies.begin(), latencies.end());
        emit("concurrent_query_lat_avg_us",
             driven ? bufann_bench::safe_div(mixed->callback_us[2],
                                             mixed->completed[2])
                    : std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                          latencies.size());
        emit("concurrent_query_lat_p99_us",
             latencies.empty()
                 ? 0.0
                 : latencies[size_t(0.99 * (latencies.size() - 1))]);
        if (driven) {
          emit("concurrent_query_latency_sample_count", latencies.size());
          emit("concurrent_query_latency_sample_limit",
               MixedPlan::latency_sample_limit);
          emit("concurrent_query_latency_sample_policy",
               "completion_order_reservoir_seeded");
          emit("concurrent_query_latency_sample_seed",
               ann_bench::WeightedUpdatePool::kDefaultSeed);
        }
        emit("mutation_timing_scope",
             "full_mixed_foreground_wall_includes_launch_join_excludes_setup_"
             "checkpoint_post_round_query");
        emit("concurrent_recall_policy",
             "not_scored_weakly_consistent_queries");
        emit("overlap_scope",
             "admitted_operation_intervals_including_monitor_native_lock_io_"
             "and_retirement_not_cpu_simultaneity");
        emit("post_round_barrier",
             "all_foreground_workers_joined_checkpoint_validated_before_"
             "scoring");
        emit("backend_bg_flush_workers", 0);
        emit("openmp_threads", 1);
        emit("mkl_threads", 1);
        emit("runtime_threads_before_mixed_launch",
             mixed->runtime_threads_before_launch);
      }
      emit("publication", "manifest_last_nonatomic_no_fsync");
      const auto& m = index.origin_manifest();
      emit("base_count", m.training.base_count);
      emit("source_rows", m.source_rows);
      emit("assignment_router_mode", router_mode_name(m.routing.mode));
      training_settings(record, first, m.training);
      emit("router_degree", m.routing.degree);
      emit("construction_breadth", m.routing.construction_breadth);
      emit("query_breadth", m.routing.query_breadth);
      emit("insert_breadth", m.routing.insert_breadth);
      emit("router_seed", m.routing.seed);
      emit("input_index_format", args.get("--index_format", "immutable"));
      const auto manifest =
          artifact_digest(args.require("--index_prefix") + ".manifest");
      emit("input_manifest_bytes", manifest.bytes);
      emit("input_manifest_hash_fnv1a", manifest.hash);
      emit("source_bytes", m.sources[0].bytes);
      emit("source_hash_fnv1a", m.sources[0].hash);
      emit("buffer_pool_frames", args.u32("--buffer_frames", 64));
      emit("cache_policy",
           "uncontrolled_after_full_source_and_bundle_hash_and_private_copy");
    }
    template<typename T>
    void mutate_typed(const Arguments& args, DataType type,
                      Clock::time_point workflow_start) {
      args.reject_present(
          {"--query_file", "--gt_file", "--search_result_file",
           "--query_threads", "--recall_at", "--nprobe", "--rerank_factor",
           "--router_mode", "--router_rerank_factor", "--router_breadth"},
          "not used by mutation-only execution; query the saved checkpoint "
          "separately");
      validate_driver_paths(args, true);
      const auto     budget = budget_bytes(args.real("--memory_budget_gb", 1));
      DriverProgress opening("open_private_copy_verify");
      const auto     start = Clock::now();
      auto           index = open_mutable<T>(args, budget);
      const auto     open_time = seconds(start);
      opening.finish();
      const auto& m = index->origin_manifest();
      check(!args.has("--dim") || args.u32("--dim") == m.training.dim,
            "CLI/manifest dimension mismatch");
      check(!args.has("--nlist") || args.u32("--nlist") == m.training.nlist,
            "CLI/manifest nlist mismatch");
      const auto s =
          update_index(args, *index, budget, index->estimated_retained_bytes());
      std::ostringstream record;
      const auto         phase =
          args.require("--workload") == "insert_only" ? "insert" : "delete";
      begin_record(record, phase, type);
      bool first = false;
      bufann_bench::json_kv(record, first, "workload", phase);
      bufann_bench::json_kv(record, first, "dim", m.training.dim);
      bufann_bench::json_kv(record, first, "nlist", m.training.nlist);
      bufann_bench::json_kv(record, first, "pq_chunks", m.chunks);
      bufann_bench::json_kv(record, first, "open_time_s", open_time);
      bufann_bench::json_kv(record, first, "index_prefix",
                            args.require("--index_prefix"));
      bufann_bench::json_kv(record, first, "memory_budget_bytes", budget);
      bufann_bench::json_kv(record, first, "peak_process_rss_mb",
                            peak_process_rss_mb());
      emit_update(record, first, args, *index, s);
      bufann_bench::json_kv(record, first, "driver_workflow_time_s",
                            seconds(workflow_start));
      bufann_bench::json_kv(record, first, "driver_workflow_time_scope",
                            "driver_dispatch_through_completed_workload_before_"
                            "canonical_record_write_not_process_lifetime");
      record << "}";
      publish_record(args, record.str());
    }

    template<typename T, typename Index>
    void search_typed(const Arguments& args, DataType type,
                      std::unique_ptr<Index> index, double open_time,
                      Clock::time_point workflow_start) {
      DriverProgress loading("query_input_validation_load");
      const bool     is_mixed = args.require("--workload") == "mixed";
      const bool     updating =
          args.require("--workload") == "search_after_update" || is_mixed;
      const auto  budget = budget_bytes(args.real("--memory_budget_gb", 1.0));
      const auto& manifest = cli_manifest(*index);
      const auto  dim = manifest.training.dim;
      check(!args.has("--dim") || args.u32("--dim") == dim,
            "CLI/manifest dimension mismatch");
      check(!args.has("--nlist") ||
                args.u32("--nlist") == manifest.training.nlist,
            "CLI/manifest nlist mismatch");
      SearchConfig config{args.u32("--nprobe", 1), args.u32("--recall_at", 10),
                          args.real("--rerank_factor", 2.0),
                          args.u32("--query_threads", 1)};
      config.validate(manifest.training.nlist);
      const auto routing = query_routing_request(args, manifest.routing);
      const auto warmup_requested = args.u32("--warmup_queries", 0);
      validate_warmup_count(warmup_requested);
      const auto queries =
          inspect_vectors(args.require("--query_file"), type, dim);
      const bool  has_truth = args.has("--gt_file");
      BinMetadata truth{};
      if (has_truth)
        truth = inspect_truth(args.require("--gt_file"), queries.rows,
                              config.recall_at);
      const uint32_t workers = std::min(config.query_threads, queries.rows);
      const uint64_t result_count = uint64_t(queries.rows) * config.recall_at;
      auto           query_bytes = index->estimated_query_bytes(config);
      uint64_t       bytes = cli_retained(*index);
      add(bytes, uint64_t(queries.rows) * dim, sizeof(T));
      add(bytes, result_count, sizeof(uint32_t) + sizeof(float));
      add(bytes, queries.rows, sizeof(double));
      add(bytes, uint64_t(truth.rows) * truth.cols,
          sizeof(uint32_t) + sizeof(float));
      add(bytes, truth.cols, sizeof(uint32_t));  // sorted GT row scratch
      add(bytes, workers, query_bytes);
      add(bytes, workers, 8ull * 1024 * 1024 + sizeof(WorkerStats));
      add(bytes,
          1024 * 1024);  // I/O streams, worker handles and control overhead
      uint64_t warmup_row_bytes = 0;
      if (warmup_requested) {
        add(warmup_row_bytes, dim, sizeof(T));
        add(warmup_row_bytes, 64);
        add(bytes, warmup_row_bytes);
      }
      check(bytes <= budget, "CLI composed search exceeds workspace allowance");

      std::ifstream input(args.require("--query_file"), std::ios::binary);
      input.seekg(8);
      std::vector<T> coordinates;
      read_values(input, coordinates, uint64_t(queries.rows) * dim);
      for (T value : coordinates)
        check(std::isfinite(double(value)), "Nonfinite query coordinate");
      GroundTruth gold;
      if (has_truth)
        gold = load_truth(args.require("--gt_file"), truth);
      loading.finish();
      UpdateStats updates;
      MixedStats  mixed;
      if constexpr (std::is_same_v<Index, MutableIndex<T>>) {
        if (updating) {
          updates = is_mixed ? mixed_index(args, *index, budget, bytes,
                                           std::span<const T>(coordinates),
                                           config, mixed, routing)
                             : update_index(args, *index, budget, bytes);
          // Account for additional retained list capacity after insertion.
          bytes = std::max(bytes, updates.bytes);
          const auto post_query_bytes = index->estimated_query_bytes(config);
          if (post_query_bytes > query_bytes)
            add(bytes, workers, post_query_bytes - query_bytes);
          query_bytes = post_query_bytes;
          check(bytes <= budget,
                "CLI post-update search exceeds workspace allowance");
        }
      }
      uint32_t warmup_completed = 0;
      double   warmup_time_s = 0;
      if (warmup_requested) {
        DriverProgress warming("base_centroid_warmup", warmup_requested,
                               "completed_warmup_queries");
        const auto     warmup_start = Clock::now();
        warmup_completed = run_centroid_warmup<T>(
            index->router().centroids(), dim, warmup_requested,
            [&index, &config, &routing,
             query_bytes](std::span<const T> vector) {
              (void) index->query(vector, config,
                                  {query_bytes, false, routing});
            },
            [&](uint64_t completed) { warming.observe(completed); });
        warmup_time_s = seconds(warmup_start);
        warming.finish();
      }
      std::vector<uint32_t>    ids(result_count, UINT32_MAX);
      std::vector<float>       distances(result_count,
                                         std::numeric_limits<float>::infinity());
      std::vector<double>      latency(queries.rows);
      std::vector<WorkerStats> stats(workers);
      std::atomic<uint32_t>    next{0};
      std::atomic_bool         failed{false};
      DriverProgress           query_progress(updating ? "post_update_query"
                                                       : "query_foreground");
      const auto               start = Clock::now();
      {
        // jthread joins even if later thread construction fails. No shared ef
        // setter.
        std::vector<std::jthread> threads;
        threads.reserve(workers);
        for (uint32_t worker = 0; worker < workers; ++worker)
          threads.emplace_back([&, worker] {
            auto& s = stats[worker];
            try {
              while (!failed.load(std::memory_order_relaxed)) {
                const auto q = next.fetch_add(1, std::memory_order_relaxed);
                if (q >= queries.rows)
                  break;
                const auto query_start = Clock::now();
                auto       result =
                    index->query({coordinates.data() + size_t(q) * dim, dim},
                                 config, {query_bytes, false, routing});
                latency[q] = seconds(query_start) * 1e6;
                record_query(s, result, config.recall_at);
                for (size_t j = 0; j < result.hits.size(); ++j) {
                  ids[size_t(q) * config.recall_at + j] = result.hits[j].id;
                  const double distance = result.hits[j].squared_distance;
                  distances[size_t(q) * config.recall_at + j] =
                      distance > std::numeric_limits<float>::max()
                          ? std::numeric_limits<float>::infinity()
                          : static_cast<float>(distance);
                }
              }
            } catch (...) {
              s.failure = std::current_exception();
              failed.store(true, std::memory_order_relaxed);
            }
          });
      }
      const auto query_time = seconds(start);
      const auto total = sum_workers(stats);
      query_progress.finish();
      DriverProgress scoring("recall_scoring_export_statistics");
      const double   recall =
          has_truth ? score_recall(gold, ids, queries.rows, config.recall_at)
                      : 0;
      if (args.has("--search_result_file"))
        write_neighbors(args.require("--search_result_file"), queries.rows,
                        config.recall_at, ids, distances);
      std::sort(latency.begin(), latency.end());
      const double avg_latency =
          std::accumulate(latency.begin(), latency.end(), 0.0) / latency.size();
      const auto peak_rss = peak_process_rss_mb();
      scoring.finish();
      auto percentile = [&](double p) {
        return latency[static_cast<size_t>(p * double(latency.size() - 1))];
      };
      std::ostringstream record;
      begin_record(record, "query", type);
      bool first = false;
      auto emit = [&](const char* key, const auto& value) {
        bufann_bench::json_kv(record, first, key, value);
      };
      emit("workload",
           is_mixed ? "mixed" : (updating ? "search_after_update" : "query"));
      emit("dim", dim);
      emit("nlist", manifest.training.nlist);
      emit("index_prefix", args.require("--index_prefix"));
      emit("query_file", args.require("--query_file"));
      if (!updating)
        emit("base_count", manifest.training.base_count);
      emit("pq_chunks", manifest.chunks);
      emit("nprobe", config.nprobe);
      emit("recall_at", config.recall_at);
      emit("rerank_factor", config.rerank_factor);
      emit("router_mode", router_mode_name(routing.mode));
      emit("router_rerank_factor", routing.rerank_factor);
      emit("router_breadth", routing.breadth);
      // Combined update/query records get these fields from mutation metadata.
      if (!updating) {
        emit("assignment_router_mode", router_mode_name(manifest.routing.mode));
        training_settings(record, first, manifest.training);
      }
      emit("optimized_modes_built", manifest.routing.native_modes);
      emit("pca_retained_dim", index->router().projected_dim());
      emit("pca_variance_target", manifest.routing.pca_variance);
      emit("pca_retained_variance", index->router().retained_variance());
      emit("pca_degenerate", index->router().pca_degenerate());
      emit("refine_ef", manifest.routing.refine_breadth);
      emit("router_projection_us_total", total.projection);
      emit("router_graph_us_total", total.graph);
      emit("router_centroid_rerank_us_total", total.centroid_rerank);
      emit("router_distance_evaluations", total.router_distances);
      emit("router_distance_counted_queries", total.router_counted_queries);
      emit("router_distance_count_scope",
           "patched_graph_plus_full_centroid_rerank_or_exact_enumeration_"
           "hnswlib_graph_unavailable");
      emit("query_threads", config.query_threads);
      emit("active_query_threads", workers);
      emit("query_count", queries.rows);
      if (!updating) {
        emit("query_breadth", manifest.routing.query_breadth);
        emit("buffer_pool_frames", args.u32("--buffer_frames", 64));
      }
      emit("open_time_s", open_time);
      emit("query_time_s", query_time);
      emit("query_qps", bufann_bench::safe_div(queries.rows, query_time));
      emit("query_lat_avg_us", avg_latency);
      emit("query_lat_p50_us", percentile(0.5));
      emit("query_lat_p90_us", percentile(0.9));
      emit("query_lat_p95_us", percentile(0.95));
      emit("query_lat_p99_us", percentile(0.99));
      emit("query_lat_p999_us", percentile(0.999));
      emit("scanned_codes", total.scanned);
      emit("reranked_vectors", total.reranked);
      emit("logical_fetch_stages", total.stages);
      emit("backend_batch_calls", total.batches);
      emit("heap_pages_read", total.pages);
      emit("queries_with_short_results", total.short_rows);
      emit("routing_us_total", total.routing);
      emit("adc_us_total", total.adc);
      emit("scan_select_us_total", total.scan);
      emit("fetch_us_total", total.fetch);
      emit("fetch_adapter_mutex_wait_us_total", total.fetch_adapter_wait);
      emit("fetch_backend_pin_us_total", total.fetch_backend_pin);
      emit("fetch_pin_admission_wait_us_total", total.fetch_pin_admission_wait);
      emit("fetch_pin_admission_policy",
           "fifo_aggregate_worst_shard_bounds_reads_and_writes");
      emit("fetch_substage_timing_scope",
           "subintervals_of_fetch_wall_not_additional_stages_backend_pin_"
           "includes_"
           "frame_lookup_wait_eviction_io_not_pure_device_time");
      emit("rerank_us_total", total.rerank);
      emit("estimated_workspace_bytes", bytes);
      emit("memory_budget_bytes", budget);
      emit("peak_process_rss_mb", peak_rss);
      if (!updating)
        emit("cache_policy", "uncontrolled_after_full_bundle_hash");
      if (!updating)
        emit("driver_progress_policy", "phase_boundaries_no_heartbeat_thread");
      emit("query_timing_scope",
           "workers_start_join_and_result_collection_excludes_gt_and_export");
      emit("stage_timing_scope",
           "summed_per_query_wall_us_not_run_wall_or_cpu");
      emit("latency_percentile_policy", "floor_p_times_n_minus_one");
      emit("heap_pages_read_scope",
           "query_backend_page_reads_not_device_requests");
      emit("result_format", "spfresh_interleaved_uint32_float32_squared_l2");
      if (has_truth) {
        emit("recall", recall);
        emit("recall_policy", "percent_tie_aware_gt_expansion");
        emit("gt_id_policy", "source_row_ids_optional_identity_tags");
      }
      if (args.has("--search_result_file"))
        emit("search_result_file", args.require("--search_result_file"));
      if constexpr (std::is_same_v<Index, MutableIndex<T>>) {
        if (!updating)
          emit("live_count", index->size());
        emit("index_format", "mutable");
        if (updating)
          emit_update(record, first, args, *index, updates,
                      is_mixed ? &mixed : nullptr);
      }
      emit("explicit_warmup_query_count", warmup_completed);
      emit("warmup_query_count_requested", warmup_requested);
      emit("warmup_time_s", warmup_time_s);
      emit("warmup_row_allowance_bytes", warmup_row_bytes);
      emit("warmup_policy",
           warmup_requested ? "frozen_base_centroid_grid_v1" : "none");
      emit("warmup_source_policy",
           "frozen_base_trained_centroids_only_not_final_queries_or_truth");
      emit("warmup_centroid_selection",
           "floor_i_times_nlist_over_requested_round_away_zero_clamp_integer_"
           "dtype");
      emit("warmup_timing_scope",
           "caller_serial_queries_progress_excluded_from_query_time_included_"
           "in_driver_workflow");
      emit("warmup_results_policy",
           "discarded_not_scored_not_exported_not_in_final_query_stats");
      emit("warmup_cache_scope",
           "same_index_object_buffer_pool_router_no_query_worker_tls_"
           "continuity_claim_no_os_cache_control");
      if constexpr (std::is_same_v<Index, MutableIndex<T>>)
        emit(
            "query_cache_process_policy",
            warmup_requested
                ? (updating ? "same_process_live_mutable_session_after_"
                              "checkpoint_no_reopen_frozen_base_centroid_warmup"
                            : "private_mutable_copy_opened_in_this_process_"
                              "frozen_base_centroid_warmup")
            : updating
                ? "same_process_live_mutable_session_after_checkpoint_no_"
                  "reopen_no_explicit_warmup"
                : "private_mutable_copy_opened_in_this_process_no_"
                  "explicit_warmup");
      else
        emit(
            "query_cache_process_policy",
            warmup_requested
                ? "immutable_bundle_opened_in_this_process_frozen_base_"
                  "centroid_warmup"
                : "immutable_bundle_opened_in_this_process_no_explicit_warmup");
      emit("cache_state_claim",
           "open_validation_may_touch_pages_os_cache_not_controlled");
      emit("driver_workflow_time_s", seconds(workflow_start));
      emit("driver_workflow_time_scope",
           "driver_dispatch_through_completed_workload_query_scoring_export_"
           "before_canonical_record_write_not_process_lifetime");
      record << "}";
      publish_record(args, record.str());
    }
    template<typename T>
    void driver_typed(const Arguments& args, DataType type) {
      const auto workflow_start = Clock::now();
      const auto workload = args.require("--workload");
      const bool mixed = workload == "mixed";
      check(workload == "search_only" || workload == "insert_only" ||
                workload == "delete_only" ||
                workload == "search_after_update" || mixed,
            "Unsupported native workload");
      if (args.has("--warmup_queries")) {
        check(workload == "search_only" || workload == "search_after_update" ||
                  mixed,
              "warmup_queries only applies to query-producing workloads");
        validate_warmup_count(args.u32("--warmup_queries"));
      }
      if (args.has("--insert_threads")) {
        check(workload == "insert_only" || workload == "search_after_update",
              "insert_threads only applies to sequential insertion workloads");
        effective_insert_workers(0, args.u32("--insert_threads"));
      }
      if (args.has("--insert_input_policy")) {
        check(workload == "insert_only" || workload == "search_after_update",
              "insert_input_policy only applies to sequential insertion "
              "workloads");
        parse_insert_input_policy(args.require("--insert_input_policy"));
      }
      if (!mixed)
        args.reject_present({"--scheduler", "--pool_threads", "--query_ratio",
                             "--query_count", "--scheduler_seed"},
                            "mixed-only setting");
      else {
        omp_set_num_threads(1);
        mkl_set_num_threads(1);
      }
      if (workload == "insert_only" || workload == "delete_only")
        return mutate_typed<T>(args, type, workflow_start);
      const bool updating = workload == "search_after_update" || mixed;
      validate_driver_paths(args, updating);
      if (!updating)
        args.reject_present({"--pq_prefix", "--full_data_file",
                             "--insert_ids_file", "--delete_ids_file"},
                            "not used by search-only execution");
      const auto format = args.get("--index_format", "immutable");
      check(format == "immutable" || format == "mutable",
            "Unknown index_format");
      const auto     budget = budget_bytes(args.real("--memory_budget_gb", 1));
      DriverProgress opening("open_private_copy_verify");
      const auto     start = Clock::now();
      if (updating || format == "mutable") {
        auto       index = open_mutable<T>(args, budget);
        const auto open_time = seconds(start);
        opening.finish();
        return search_typed<T>(args, type, std::move(index), open_time,
                               workflow_start);
      }
      args.reject_present({"--private_heap_path"},
                          "not used by immutable search");
      auto index =
          NativeIndex<T>::open(args.require("--index_prefix"),
                               {args.u32("--buffer_frames", 64), budget});
      const auto open_time = seconds(start);
      opening.finish();
      return search_typed<T>(args, type, std::move(index), open_time,
                             workflow_start);
    }
  }  // namespace

  void execute_build(const Arguments&                 args,
                     const diskann::ivf::BuildConfig& config,
                     const diskann::ivf::BinMetadata& vectors,
                     const diskann::ivf::PQMetadata&  pq) {
    switch (config.data_type) {
      case DataType::Float:
        return build_typed<float>(args, config, vectors, pq);
      case DataType::Int8:
        return build_typed<int8_t>(args, config, vectors, pq);
      case DataType::UInt8:
        return build_typed<uint8_t>(args, config, vectors, pq);
    }
    throw std::invalid_argument("Unsupported dtype");
  }
  void execute_driver(const Arguments& args) {
    switch (parse_data_type(args.get("--data_type", "float"))) {
      case DataType::Float:
        return driver_typed<float>(args, DataType::Float);
      case DataType::Int8:
        return driver_typed<int8_t>(args, DataType::Int8);
      case DataType::UInt8:
        return driver_typed<uint8_t>(args, DataType::UInt8);
    }
    throw std::invalid_argument("Unsupported dtype");
  }
}  // namespace ivf_cli
