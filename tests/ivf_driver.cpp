#include "bufann/ivf_config.h"
#include "bufann/ivf_io.h"
#include "ivf_cli.h"
#ifdef IVF_NATIVE_EXECUTION
#include "ivf_cli_execution.h"
#include "ann_exception.h"
#endif

int main(int argc, char** argv) {
  using namespace diskann::ivf;
  try {
    ivf_cli::Arguments args(argc, argv,
                            {"--validate_inputs",
                             "--data_type",
                             "--metric",
                             "--dim",
                             "--index_prefix",
                             "--pq_prefix",
                             "--workload",
                             "--query_file",
                             "--gt_file",
                             "--recall_at",
                             "--query_threads",
                             "--insert_threads",
                             "--insert_input_policy",
                             "--nlist",
                             "--nprobe",
                             "--rerank_factor",
                             "--full_data_file",
                             "--insert_ids_file",
                             "--delete_ids_file",
                             "--memory_budget_gb",
                             "--buffer_frames",
                             "--result_file",
                             "--search_result_file",
                             "--index_format",
                             "--private_heap_path",
                             "--checkpoint_prefix",
                             "--scheduler",
                             "--pool_threads",
                             "--query_ratio",
                             "--query_count",
                             "--scheduler_seed",
                             "--router_mode",
                             "--router_rerank_factor",
                             "--router_breadth"});
    if (args.help) {
      std::cout
          << "IVF driver (--validate_inputs 1: validation only; 0: native "
             "execution in full routing-enabled builds).\n"
             "Usage: ivf_driver --validate_inputs 0|1 --data_type "
             "float|int8|uint8\n"
             "  --index_prefix PATH --dim N --nlist N --query_file PATH\n"
             "  --workload "
             "search_only|insert_only|delete_only|search_after_update|mixed\n"
             "  [--pq_prefix PATH] [--gt_file PATH] [--recall_at N] "
             "[--query_threads N]\n"
             "  [--nprobe N] [--rerank_factor F] [--metric l2]\n"
             "  [--full_data_file PATH --insert_ids_file PATH "
             "--delete_ids_file PATH]\n"
             "Native search: --dim/--nlist are optional assertions; no source "
             "PQ prefix needed.\n"
             "  [--memory_budget_gb F] [--buffer_frames N] [--result_file "
             "JSON_PATH]\n"
             "  [--search_result_file SPFresh_BINARY_PATH]\n";
      std::cout << "Updates: --private_heap_path NEW_PATH --checkpoint_prefix "
                   "NEW_PREFIX\n"
                   "  --full_data_file ORIGINAL_FULL_SOURCE "
                   "--insert_ids_file/--delete_ids_file COUNTED_IDS\n"
                   "  [--index_format immutable|mutable] (default immutable; "
                   "mutable requires a new private heap)\n"
                   "insert_only/delete_only need no queries. "
                   "search_after_update deletes then inserts,\n"
                   "checkpoints and queries the final live set; no concurrent "
                   "mixed scheduling.\n";
      std::cout
          << "Sequential insertion: [--insert_threads 1..256] (default 1);\n"
             "  only insert_only/search_after_update; delete phase stays "
             "serial.\n";
      std::cout
          << "  [--insert_input_policy streaming|staged] (default streaming);\n"
             "  staged coordinates load before insertion timing and require "
             "memory admission.\n";
      std::cout << "Mixed: --workload mixed --scheduler "
                   "update_driven|weighted|fixed_8_8 "
                   "(default weighted)\n"
                   "  [--pool_threads N] [--query_ratio 1..99] [--query_count "
                   "N] [--scheduler_seed N]\n"
                   "update_driven: dedicated half mutation/half query workers; "
                   "queries run until updates finish.\n"
                   "  Requires at least 2 pool workers; query ratio/count/seed "
                   "overrides are not used.\n"
                   "One quiescent round per call; GT/results apply only to the "
                   "post-round query pass.\n";
      std::cout << "Query-only overrides among persisted modes (assignments "
                   "stay frozen):\n"
                   "  [--router_mode "
                   "hnswlib|exact|patched-full|refined|pca|pca-refined]\n"
                   "  [--router_rerank_factor F] [--router_breadth N]\n";
      return 0;
    }
    const bool validation = args.validation_mode();
#ifndef IVF_NATIVE_EXECUTION
    args.require_validation_mode();
#endif
    if (args.get("--metric", "l2") != "l2")
      throw std::invalid_argument("IVF currently supports only metric l2");
#ifdef IVF_NATIVE_EXECUTION
    if (!validation) {
      ivf_cli::execute_driver(args);
      return 0;
    }
#endif
    args.reject_present(
        {"--memory_budget_gb", "--buffer_frames", "--result_file",
         "--search_result_file", "--index_format", "--private_heap_path",
         "--checkpoint_prefix", "--scheduler", "--pool_threads",
         "--query_ratio", "--query_count", "--scheduler_seed", "--router_mode",
         "--router_rerank_factor", "--router_breadth", "--insert_threads",
         "--insert_input_policy"},
        "execution-only setting");
    const auto     type = parse_data_type(args.get("--data_type", "float"));
    const uint32_t dim = args.u32("--dim");
    if (dim == 0)
      throw std::invalid_argument("dim must be positive");
    SearchConfig config;
    config.nprobe = args.u32("--nprobe", 1);
    config.recall_at = args.u32("--recall_at", 10);
    config.query_threads = args.u32("--query_threads", 1);
    config.rerank_factor = args.real("--rerank_factor", 2.0);
    config.validate(args.u32("--nlist"));
    const std::string prefix = args.require("--index_prefix");
    const std::string workload = args.require("--workload");
    if (workload != "search_only" && workload != "insert_only" &&
        workload != "delete_only" && workload != "search_after_update" &&
        workload != "mixed") {
      throw std::invalid_argument("unsupported IVF workload");
    }
    const auto queries =
        inspect_vectors(args.require("--query_file"), type, dim);
    if (!args.get("--gt_file").empty())
      inspect_truth(args.get("--gt_file"), queries.rows, config.recall_at);
    const auto pq = inspect_pq_pivots(
        args.get("--pq_prefix", prefix) + "_pq_pivots.bin", dim);
    if (pq.dim != dim)
      throw std::invalid_argument("PQ dimension mismatch");
    if (workload != "search_only" || !args.get("--full_data_file").empty()) {
      const auto vectors =
          inspect_vectors(args.require("--full_data_file"), type, dim);
      if (workload == "insert_only" || workload == "search_after_update" ||
          workload == "mixed") {
        load_counted_ids(args.require("--insert_ids_file"), vectors.rows);
      }
      if (workload == "delete_only" || workload == "search_after_update" ||
          workload == "mixed") {
        load_counted_ids(args.require("--delete_ids_file"), vectors.rows);
      }
    }
    ivf_cli::validation_record(workload.c_str(), data_type_name(type), dim,
                               queries.rows, pq.chunks);
    return 0;
#ifdef IVF_NATIVE_EXECUTION
  } catch (const diskann::ANNException& error) {
    std::cerr << "ERROR: " << error.message() << std::endl;
    return 1;
#endif
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 1;
  }
}
