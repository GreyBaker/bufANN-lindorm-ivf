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
                             "--data_file",
                             "--index_prefix",
                             "--pq_prefix",
                             "--base_count",
                             "--nlist",
                             "--sample_rate",
                             "--max_reps",
                             "--seed",
                             "--build_threads",
                             "--memory_budget_gb",
                             "--id_capacity",
                             "--page_size",
                             "--buffer_frames",
                             "--router_degree",
                             "--construction_breadth",
                             "--query_breadth",
                             "--insert_breadth",
                             "--router_seed",
                             "--result_file",
                             "--optimized_modes",
                             "--router_mode",
                             "--refine_ef",
                             "--pca_variance",
                             "--router_rerank_factor",
                             "--routing_work_limit",
                             "--trainer_mode",
                             "--init",
                             "--training_breadth",
                             "--training_rerank_factor",
                             "--training_refine_breadth"});
    if (args.help) {
      std::cout
          << "IVF build (--validate_inputs 1: validation only; 0: native "
             "execution in full routing-enabled builds).\n"
             "Usage: ivf_build --validate_inputs 0|1 --data_type "
             "float|int8|uint8\n"
             "  --data_file PATH --index_prefix PATH --pq_prefix PATH --dim N\n"
             "  --base_count N --nlist N [--sample_rate F] [--max_reps N]\n"
             "  [--seed N] [--build_threads N] [--memory_budget_gb F] "
             "[--metric l2]\n"
             "Execution settings: [--id_capacity N] [--page_size N] "
             "[--buffer_frames N]\n"
             "  [--router_degree N] [--construction_breadth N] "
             "[--query_breadth N]\n"
             "  [--insert_breadth N] [--router_seed N] [--result_file "
             "JSON_PATH]\n";
      std::cout << "Optimized routing: --optimized_modes 1 builds the patched "
                   "hnswlib/PCA ablations.\n"
                   "  [--router_mode "
                   "hnswlib|exact|patched-full|refined|pca|pca-refined]\n"
                   "  --router_mode is the frozen base/insert assignment "
                   "default, not a query-only ablation.\n"
                   "  [--refine_ef N] [--pca_variance F]\n"
                   "  [--router_rerank_factor F] [--routing_work_limit N] "
                   "(PCA/refinement work allowance)\n";
      std::cout
          << "Training: [--trainer_mode "
             "exact|ann-rebuild|ann-reuse|ann-reuse-seeded|ann-reuse-count|ann-"
             "reuse-seeded-count|lv|lv-plus]\n"
             "  LV/LV+: --training_rerank_factor 3 --training_refine_breadth "
             "128; PCA95 refit each pass.\n"
             "  [--init sample|random_partition] [--training_breadth N]\n"
             "  Defaults: ann-reuse-seeded, random_partition, breadth 64.\n"
             "  Training breadth is independent of frozen query/insert router "
             "breadth.\n";
      return 0;
    }
    const bool validation = args.validation_mode();
#ifndef IVF_NATIVE_EXECUTION
    args.require_validation_mode();
#endif
    if (validation)
      args.reject_present(
          {"--id_capacity", "--page_size", "--buffer_frames", "--router_degree",
           "--construction_breadth", "--query_breadth", "--insert_breadth",
           "--router_seed", "--result_file"},
          "execution-only setting");
    if (args.get("--metric", "l2") != "l2")
      throw std::invalid_argument("IVF currently supports only metric l2");
    args.require("--index_prefix");
    BuildConfig config;
    config.data_type = parse_data_type(args.get("--data_type", "float"));
    config.dim = args.u32("--dim");
    config.base_count = args.u32("--base_count");
    config.nlist = args.u32("--nlist");
    config.sample_rate = args.real("--sample_rate", 1.0);
    config.max_reps = args.u32("--max_reps", 15);
    config.build_threads = args.u32("--build_threads", 1);
    config.seed = args.integer("--seed");
    config.memory_budget_gb = args.real("--memory_budget_gb", 1.0);
    config.trainer_mode =
        parse_trainer_mode(args.get("--trainer_mode", "ann-reuse-seeded"));
    config.init = parse_centroid_init(args.get("--init", "random_partition"));
    config.training_breadth = args.u32("--training_breadth", 64);
    config.training_rerank_factor = args.real("--training_rerank_factor", 3);
    config.training_refine_breadth = args.u32("--training_refine_breadth", 128);
    config.validate();
    const auto vectors = inspect_vectors(args.require("--data_file"),
                                         config.data_type, config.dim);
    if (config.base_count > vectors.rows)
      throw std::invalid_argument("base_count exceeds vector file rows");
    const std::string pq_prefix = args.require("--pq_prefix");
    const auto pq = inspect_pq_pivots(pq_prefix + "_pq_pivots.bin", config.dim);
    if (pq.dim != config.dim)
      throw std::invalid_argument("PQ dimension mismatch");
    inspect_pq_codes(pq_prefix + "_pq_compressed.bin", vectors.rows, pq.chunks);
#ifdef IVF_NATIVE_EXECUTION
    if (!validation) {
      ivf_cli::execute_build(args, config, vectors, pq);
      return 0;
    }
#endif
    args.reject_present(
        {"--optimized_modes", "--router_mode", "--refine_ef", "--pca_variance",
         "--router_rerank_factor", "--routing_work_limit"},
        "routing execution-only setting");
    ivf_cli::validation_record("build", data_type_name(config.data_type),
                               config.dim, config.base_count, pq.chunks);
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
