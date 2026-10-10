// Paper §4.1 calibration only: no official queries, heap I/O, or end-to-end
// predictor.
#include "ivf_cli.h"
#include "ivf_training_json.h"
#include "bufann/ivf_artifacts.h"
#include "bufann/ivf_cost_model.h"
#include "bufann/ivf_lists.h"
#include "bufann/ivf_pq.h"
#include "ann_exception.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

using namespace diskann::ivf;
namespace {
  using Clock = std::chrono::steady_clock;
  void check(bool ok, const char* message) {
    if (!ok)
      throw std::invalid_argument(message);
  }
  uint64_t mul(uint64_t a, uint64_t b) {
    check(!b || a <= UINT64_MAX / b, "Cost-model work/size overflow");
    return a * b;
  }
  void add(uint64_t& n, uint64_t amount) {
    check(n <= UINT64_MAX - amount, "Cost-model workspace overflow");
    n += amount;
  }
  template<class T>
  void artifact(const std::filesystem::path& path, uint32_t rows, uint32_t dim,
                const std::vector<T>& values) {
    check(uint64_t(rows) * dim == values.size(),
          "Invalid calibration artifact shape");
    claim_artifact(path.string());
    std::ofstream  out(path, std::ios::binary);
    const uint32_t header[]{rows, dim};
    out.write(reinterpret_cast<const char*>(header), 8);
    out.write(reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(T));
    out.close();
    check(bool(out), "Calibration artifact write failed");
  }
  template<class T>
  std::vector<float> read_rows(const std::string&        path,
                               std::span<const uint32_t> ids, uint32_t dim) {
    std::ifstream input(path, std::ios::binary);
    check(bool(input), "Cannot read calibration source");
    std::vector<T>     row(dim);
    std::vector<float> result(size_t(ids.size()) * dim);
    for (size_t slot = 0; slot < ids.size(); ++slot) {
      input.seekg(8 + uint64_t(ids[slot]) * dim * sizeof(T));
      input.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
      check(bool(input), "Truncated calibration source");
      for (uint32_t d = 0; d < dim; ++d) {
        check(std::isfinite(double(row[d])),
              "Nonfinite calibration coordinate");
        result[slot * dim + d] = float(row[d]);
      }
    }
    return result;
  }
  std::vector<float> rows(const std::string& path, DataType type,
                          std::span<const uint32_t> ids, uint32_t dim) {
    switch (type) {
      case DataType::Float:
        return read_rows<float>(path, ids, dim);
      case DataType::Int8:
        return read_rows<int8_t>(path, ids, dim);
      case DataType::UInt8:
        return read_rows<uint8_t>(path, ids, dim);
    }
    throw std::invalid_argument("Invalid calibration dtype");
  }
  std::vector<uint32_t> grid(const std::string& text) {
    std::vector<uint32_t> result;
    std::stringstream     input(text);
    std::string           part;
    check(!text.empty() && text.back() != ',', "Invalid candidate grid");
    while (std::getline(input, part, ',')) {
      uint32_t   value = 0;
      const auto parsed =
          std::from_chars(part.data(), part.data() + part.size(), value);
      check(parsed.ec == std::errc() &&
                parsed.ptr == part.data() + part.size() && value &&
                value <= INT32_MAX,
            "Invalid candidate nlist");
      check(std::find(result.begin(), result.end(), value) == result.end(),
            "Duplicate candidate nlist");
      result.push_back(value);
    }
    check(!result.empty() && result.size() <= 1024,
          "Invalid/oversized candidate grid");
    return result;
  }
  struct QueryCost {
    ProbeEstimate probe;
    double        t1_us = 0, t2_us = 0;
    uint64_t      routed_sample_codes = 0;
  };
  struct Candidate {
    CandidateCost          cost;
    bool                   eligible = false;
    TrainingStats          training;
    std::vector<QueryCost> queries;
  };
  void identity(std::ostream& out, bool& first, const char* key,
                const ArtifactDigest& value) {
    if (!first)
      out << ',';
    first = false;
    out << '"' << key << "\":{\"bytes\":" << value.bytes
        << ",\"fnv1a64\":" << value.hash << '}';
  }
}  // namespace
int main(int argc, char** argv) {
  try {
    const ivf_cli::Arguments args(argc, argv,
                                  {"--data_file",
                                   "--data_type",
                                   "--dim",
                                   "--pq_prefix",
                                   "--base_count",
                                   "--sample_count",
                                   "--synthetic_queries",
                                   "--nlist_grid",
                                   "--topk",
                                   "--target_recall",
                                   "--seed",
                                   "--output_dir",
                                   "--max_reps",
                                   "--build_threads",
                                   "--memory_budget_gb",
                                   "--trainer_mode",
                                   "--init",
                                   "--training_breadth",
                                   "--router_mode",
                                   "--optimized_modes",
                                   "--router_degree",
                                   "--construction_breadth",
                                   "--query_breadth",
                                   "--insert_breadth",
                                   "--router_seed",
                                   "--refine_ef",
                                   "--pca_variance",
                                   "--router_rerank_factor",
                                   "--routing_work_limit",
                                   "--profile_codes",
                                   "--profile_repeats",
                                   "--routing_repeats",
                                   "--rerank_factor",
                                   "--work_limit",
                                   "--min_sample_per_centroid",
                                   "--training_rerank_factor",
                                   "--training_refine_breadth"});
    if (args.help) {
      std::cout << "Paper T1+T2 calibration: ivf_cost_model --data_file "
                   "BASE_SOURCE --pq_prefix PREFIX\n"
                   "  --data_type float|int8|uint8 --dim N --base_count N "
                   "--sample_count M\n"
                   "  --nlist_grid 1024,10000 --output_dir NEW_DIRECTORY "
                   "[--synthetic_queries 100]\n"
                   "  [--topk 10 --target_recall .90 --profile_codes 100000 "
                   "--work_limit 1000000000]\n"
                   "  [--min_sample_per_centroid 32] Declared project policy, "
                   "not a paper requirement; minimum 1.\n"
                   "  No official query/GT arguments. Fetch/rerank excluded. "
                   "Use benchmark/IVF/cost_model.py.\n";
      return 0;
    }
    const auto path = args.require("--data_file"),
               pq = args.require("--pq_prefix");
    const auto dtype = parse_data_type(args.get("--data_type", "float"));
    const auto dim = args.u32("--dim"), base = args.u32("--base_count"),
               m = args.u32("--sample_count");
    const auto qcount = args.u32("--synthetic_queries", 100),
               topk = args.u32("--topk", 10);
    const auto profile_rows = args.u32("--profile_codes", 100000),
               profile_repeats = args.u32("--profile_repeats", 3);
    const auto routing_repeats = args.u32("--routing_repeats", 3);
    const auto minimum = args.u32("--min_sample_per_centroid", 32);
    check(minimum && minimum <= INT32_MAX,
          "Invalid declared sample adequacy minimum");
    const auto seed = args.integer("--seed"),
               work = args.integer("--work_limit", 1000000000);
    const double recall = args.real("--target_recall", .90),
                 memory = args.real("--memory_budget_gb", 1);
    check(memory > 0 && memory < double(UINT64_MAX) / (1024.0 * 1024 * 1024),
          "Invalid memory estimate budget");
    const uint64_t budget = uint64_t(memory * 1024 * 1024 * 1024);
    const auto     shape = inspect_vectors(path, dtype, dim);
    const auto     pivots = inspect_pq_pivots(pq + "_pq_pivots.bin", dim);
    inspect_pq_codes(pq + "_pq_compressed.bin", shape.rows, pivots.chunks);
    check(base <= shape.rows && base <= INT32_MAX && dim && pivots.dim == dim &&
              m && qcount && uint64_t(m) + qcount <= base && qcount <= 4096 &&
              topk && topk <= m,
          "Invalid base/sample/query/top-k shapes");
    check(profile_rows && profile_rows <= base && profile_repeats &&
              routing_repeats && profile_repeats <= 1024 &&
              routing_repeats <= 1024 && recall > 0 && recall <= 1,
          "Invalid profiling/recall settings");
    auto candidates = grid(args.require("--nlist_grid"));
    check(uint64_t(qcount) * candidates.size() <= 16384,
          "Calibration query/candidate report exceeds bounded size");
    uint64_t evaluations = mul(m, qcount), centroid_work = 0;
    for (auto n : candidates)
      if (uint64_t(n) * minimum <= m)
        add(centroid_work, mul(n, qcount));
    add(evaluations, centroid_work);
    add(evaluations,
        mul(mul(profile_rows, uint64_t(profile_repeats) + 1), qcount));
    check(evaluations <= work,
          "Calibration GT/radius/profile work exceeds declared allowance");
    uint64_t retained = mul(uint64_t(m) + qcount, uint64_t(dim) * 4 + 8);
    add(retained, mul(uint64_t(qcount) * topk, 4));
    add(retained, mul(profile_rows, uint64_t(pivots.chunks) * 2 + 4));
    add(retained, mul(uint64_t(qcount) * candidates.size(), 128));
    add(retained, 65536);
    add(retained,
        mul(candidates.size(), uint64_t(sizeof(Candidate)) +
                                   mul(uint64_t(args.u32("--max_reps", 15)) + 1,
                                       sizeof(TrainingPass) + sizeof(double))));
    add(retained,
        mul(dim,
            256ull * 4 * 4));  // PQ centroid/transpose/LUT storage allowance.
    check(retained < budget && retained <= SIZE_MAX,
          "Calibration retained data exceeds memory allowance");
    const auto source_before = artifact_digest(path),
               pivot_before = artifact_digest(pq + "_pq_pivots.bin"),
               codes_before = artifact_digest(pq + "_pq_compressed.bin");
    const std::filesystem::path root(args.require("--output_dir"));
    check(std::filesystem::create_directory(root),
          "Calibration output exists or cannot be created");
    const auto split = sample_calibration_rows(base, m, qcount, seed, budget);
    const auto samples = rows(path, dtype, split.sample_ids, dim),
               queries = rows(path, dtype, split.query_ids, dim);
    artifact(root / "sample.bin", m, dim, samples);
    artifact(root / "queries.bin", qcount, dim, queries);
    artifact(root / "sample_ids.bin", m, 1, split.sample_ids);
    artifact(root / "query_ids.bin", qcount, 1, split.query_ids);
    const auto truth = calibration_ground_truth(samples, queries, dim, topk,
                                                work, budget - retained);
    artifact(root / "gt_slots.bin", qcount, topk, truth);
    PQCodebook<float>    codebook(pq + "_pq_pivots.bin", dim, pivots.chunks);
    std::vector<uint8_t> codes(size_t(profile_rows) * pivots.chunks);
    std::ifstream        input(pq + "_pq_compressed.bin", std::ios::binary);
    input.seekg(8);
    input.read(reinterpret_cast<char*>(codes.data()), codes.size());
    check(bool(input), "Cannot read scan profile codes");
    SearchConfig scan;
    scan.recall_at = topk;
    scan.rerank_factor = args.real("--rerank_factor", 4);
    const uint32_t candidate_budget =
        static_cast<uint32_t>(scan.candidate_budget(base));
    double             coefficient = 0;
    uint64_t           profiled_codes = 0;
    std::vector<float> lut;
    for (uint32_t q = 0; q < qcount; ++q) {
      codebook.populate_distances(queries.data() + size_t(q) * dim, lut);
      const auto profiled = PostingLists::profile_scan(
          codes, pivots.chunks, lut, candidate_budget, profile_repeats,
          budget - retained);
      coefficient += profiled.per_code_us;
      profiled_codes += profiled.scanned_codes;
    }
    coefficient /= qcount;
    BuildConfig config;
    config.dim = dim;
    config.base_count = m;
    config.sample_rate = 1;
    config.data_type = DataType::Float;
    config.seed = seed;
    config.max_reps = args.u32("--max_reps", 15);
    config.build_threads = args.u32("--build_threads", 1);
    config.memory_budget_gb =
        double(budget - retained) / (1024.0 * 1024 * 1024);
    config.trainer_mode =
        parse_trainer_mode(args.get("--trainer_mode", "ann-reuse-seeded"));
    config.init = parse_centroid_init(args.get("--init", "random_partition"));
    config.training_breadth = args.u32("--training_breadth", 64);
    config.training_rerank_factor = args.real("--training_rerank_factor", 3);
    config.training_refine_breadth = args.u32("--training_refine_breadth", 128);
    RouterOptions routing;
    routing.mode = parse_router_mode(args.get("--router_mode", "hnswlib"));
    const auto optimized = args.u32("--optimized_modes", 0);
    check(optimized <= 1, "optimized_modes must be 0 or 1");
    routing.native_modes = optimized;
    routing.degree = args.u32("--router_degree", 16);
    routing.construction_breadth = args.u32("--construction_breadth", 200);
    routing.query_breadth = args.u32("--query_breadth", 128);
    routing.insert_breadth = args.u32("--insert_breadth", 64);
    routing.seed = args.u32("--router_seed", 100);
    routing.refine_breadth = args.u32("--refine_ef", 128);
    routing.pca_variance = args.real("--pca_variance", .95);
    routing.router_rerank_factor = args.real("--router_rerank_factor", 3);
    routing.native_work_limit =
        args.integer("--routing_work_limit", 1000000000);
    routing.memory_budget_bytes = budget - retained;
    std::vector<Candidate>     records;
    std::vector<CandidateCost> costs;
    for (const auto n : candidates) {
      Candidate record;
      record.cost.nlist = n;
      record.eligible = uint64_t(n) * minimum <= m;
      if (!record.eligible) {
        records.push_back(std::move(record));
        continue;
      }
      if (uint64_t(n) * 32 > m)
        std::cerr << "WARNING: nlist=" << n << " has " << double(m) / n
                  << " sample rows per centroid, below the historical project "
                     "minimum 32; not production qualified\n";
      config.nlist = n;
      config.validate();
      auto trained = train_centroids_ann((root / "sample.bin").string(), config,
                                         routing, {256, work, work});
      check(trained.sampled_ids.size() == m,
            "Calibration trainer omitted sample rows");
      uint64_t final_retained = mul(m, 16);
      add(final_retained, mul(n, uint64_t(dim) * 4 + 16));
      check(final_retained < budget - retained,
            "Final calibration training result exceeds remaining memory "
            "allowance");
      auto final_routing = routing;
      final_routing.memory_budget_bytes = budget - retained - final_retained;
      auto router =
          CentroidRouter::build(dim, trained.centroids, final_routing);
      std::vector<uint32_t> assignments(m), occupancy(n);
      // Match W1's final frozen base/insert router, not the transient training
      // graph.
      for (uint32_t r = 0; r < m; ++r) {
        assignments[r] = router->assign(
            std::span<const float>(samples).subspan(size_t(r) * dim, dim));
        ++occupancy[assignments[r]];
      }
      const auto prefix = root / ("n" + std::to_string(n));
      artifact(prefix.string() + ".assignments.bin", m, 1, assignments);
      artifact(prefix.string() + ".occupancy.bin", n, 1, occupancy);
      router->save(prefix.string());
      std::vector<double> distances(n);
      for (uint32_t q = 0; q < qcount; ++q) {
        const auto query =
            std::span<const float>(queries).subspan(size_t(q) * dim, dim);
        for (uint32_t c = 0; c < n; ++c) {
          double distance = 0;
          for (uint32_t d = 0; d < dim; ++d) {
            const double delta =
                double(query[d]) - trained.centroids[size_t(c) * dim + d];
            distance += delta * delta;
          }
          distances[c] = distance;
        }
        QueryCost qc;
        qc.probe = estimate_probes(
            distances, assignments, occupancy,
            std::span<const uint32_t>(truth).subspan(size_t(q) * topk, topk),
            recall, base);
        // Reconstruct original integer queries outside timing. The
        // router's production integer-to-float adapter remains timed.
        std::vector<uint8_t> query_u8;
        std::vector<int8_t>  query_i8;
        if (dtype == DataType::UInt8)
          query_u8.assign(query.begin(), query.end());
        if (dtype == DataType::Int8)
          query_i8.assign(query.begin(), query.end());
        auto route = [&] {
          RoutingStats stats;
          if (dtype == DataType::UInt8)
            return router->search(std::span<const uint8_t>(query_u8),
                                  qc.probe.nprobe, false, &stats);
          if (dtype == DataType::Int8)
            return router->search(std::span<const int8_t>(query_i8),
                                  qc.probe.nprobe, false, &stats);
          return router->search(query, qc.probe.nprobe, false, &stats);
        };
        route();  // Separate per-query warmup, excluded.
        for (uint32_t repeat = 0; repeat < routing_repeats; ++repeat) {
          const auto start = Clock::now();
          const auto hits = route();
          qc.t1_us +=
              std::chrono::duration<double, std::micro>(Clock::now() - start)
                  .count();
          check(hits.size() == qc.probe.nprobe,
                "Production router returned short probe set");
          if (!repeat)
            for (const auto& hit : hits)
              qc.routed_sample_codes += occupancy[hit.list_id];
        }
        qc.t1_us /= routing_repeats;
        qc.t2_us = qc.probe.scaled_codes * coefficient;
        record.cost.t1_us += qc.t1_us;
        record.cost.t2_us += qc.t2_us;
        record.queries.push_back(qc);
      }
      record.cost.t1_us /= qcount;
      record.cost.t2_us /= qcount;
      record.training = trained;
      costs.push_back(record.cost);
      records.push_back(std::move(record));
      std::cout << "Calibrated nlist=" << n << std::endl;
    }
    check(source_before == artifact_digest(path) &&
              pivot_before == artifact_digest(pq + "_pq_pivots.bin") &&
              codes_before == artifact_digest(pq + "_pq_compressed.bin"),
          "Calibration sources changed during run");
    const uint32_t chosen = costs.empty() ? 0 : choose_nlist(costs);
    claim_artifact((root / "report.json").string());
    std::ofstream out(root / "report.json");
    out << std::setprecision(17) << std::boolalpha << '{';
    bool first = true;
    auto emit = [&](const char* key, const auto& value) {
      bufann_bench::json_kv(out, first, key, value);
    };
    emit("version", 2);
    emit("kind", "ivf_cost_model");
    emit("completed", chosen != 0);
    emit("production_qualified", false);
    emit("min_sample_per_centroid", minimum);
    emit("candidate_eligibility_policy",
         "declared_project_minimum_not_paper_requirement");
    emit("objective", "mean_T1_plus_T2_excludes_fetch_and_exact_rerank");
    emit("selection_source", "cost_model");
    emit("data_type", data_type_name(dtype));
    emit("dim", dim);
    emit("source_rows", shape.rows);
    emit("base_count", base);
    emit("sample_count", m);
    emit("synthetic_query_count", qcount);
    emit("topk", topk);
    emit("target_recall", recall);
    emit("seed", seed);
    emit("gt_policy", "exact_sample_squared_l2_slot_ties_source_order");
    emit("threshold_policy",
         "GT_neighbour_multiplicity_inclusive_centroid_radius");
    emit("sample_assignment_policy", "final_frozen_base_insert_router");
    emit("calibration_query_policy", "disjoint_base_rows_no_official_queries");
    emit("profile_codes", profile_rows);
    emit("profile_repeats", profile_repeats);
    emit("profiled_codes", profiled_codes);
    emit("pq_chunks", pivots.chunks);
    emit("candidate_budget", candidate_budget);
    emit("profile_policy",
         "one_real_PQ_code_list_W4_scan_and_selection_one_untimed_warmup");
    emit("per_code_us", coefficient);
    emit("rerank_factor", scan.rerank_factor);
    emit("routing_repeats", routing_repeats);
    emit("timing_policy",
         "serial_synthetic_queries_mean_of_repeats_one_per_query_warmup_not_"
         "concurrent_QPS");
    emit("t1_scope",
         "production_router_including_integer_adapter_and_routing_stats_"
         "excludes_list_id_copy");
    emit("t2_scope",
         "scan_selection_includes_scanner_validation_and_allocation_excludes_"
         "ADC_table_setup");
    emit("memory_budget_gb", memory);
    emit("retained_estimated_bytes", retained);
    emit("work_limit", work);
    emit("router_mode", router_mode_name(routing.mode));
    emit("optimized_modes", optimized);
    emit("router_degree", routing.degree);
    emit("construction_breadth", routing.construction_breadth);
    emit("query_breadth", routing.query_breadth);
    emit("insert_breadth", routing.insert_breadth);
    emit("router_seed", routing.seed);
    emit("refine_ef", routing.refine_breadth);
    emit("pca_variance", routing.pca_variance);
    emit("router_rerank_factor", routing.router_rerank_factor);
    emit("routing_work_limit", routing.native_work_limit);
    emit("max_reps", config.max_reps);
    emit("build_threads", config.build_threads);
    ivf_cli::training_settings(out, first, config);
    identity(out, first, "source_identity", source_before);
    identity(out, first, "pq_pivots_identity", pivot_before);
    identity(out, first, "pq_codes_identity", codes_before);
    out << ",\"selected_nlist\":";
    if (chosen)
      out << chosen;
    else
      out << "null";
    out << ",\"candidates\":[";
    for (size_t i = 0; i < records.size(); ++i) {
      const auto& record = records[i];
      out << (i ? "," : "") << '{';
      bool f = true;
      bufann_bench::json_kv(out, f, "nlist", record.cost.nlist);
      bufann_bench::json_kv(out, f, "eligible", record.eligible);
      bufann_bench::json_kv(out, f, "sample_rows_per_centroid",
                            double(m) / record.cost.nlist);
      bufann_bench::json_kv(out, f, "below_legacy_32_adequacy",
                            uint64_t(record.cost.nlist) * 32 > m);
      if (!record.eligible)
        bufann_bench::json_kv(
            out, f, "drop_reason",
            minimum == 32 ? "sample_count_per_cluster_below_32"
                          : "sample_count_per_cluster_below_declared_minimum");
      else {
        bufann_bench::json_kv(out, f, "t1_us", record.cost.t1_us);
        bufann_bench::json_kv(out, f, "t2_us", record.cost.t2_us);
        bufann_bench::json_kv(out, f, "total_us",
                              record.cost.t1_us + record.cost.t2_us);
        ivf_cli::training_metrics(out, f, record.training);
        out << ",\"queries\":[";
        for (size_t q = 0; q < record.queries.size(); ++q) {
          const auto& qc = record.queries[q];
          out << (q ? "," : "") << "{\"source_id\":" << split.query_ids[q]
              << ",\"threshold_squared\":" << qc.probe.threshold_squared
              << ",\"nprobe\":" << qc.probe.nprobe
              << ",\"sample_codes\":" << qc.probe.sample_codes
              << ",\"scaled_codes\":" << qc.probe.scaled_codes
              << ",\"routed_sample_codes\":" << qc.routed_sample_codes
              << ",\"t1_us\":" << qc.t1_us << ",\"t2_us\":" << qc.t2_us << '}';
        }
        out << ']';
      }
      out << '}';
    }
    out << "]}\n";
    out.close();
    check(bool(out), "Cost-model report write failed");
    check(chosen != 0,
          "No eligible candidate; exclusions saved in report.json");
    std::cout << "Selected nlist=" << chosen
              << "; report=" << (root / "report.json") << '\n';
  } catch (const diskann::ANNException& e) {
    std::cerr << "ERROR: " << e.message() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
