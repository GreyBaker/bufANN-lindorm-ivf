#include "bufann/ivf_config.h"
#include "bufann/ivf_io.h"
#include "ivf_cli.h"

int main(int argc, char** argv) {
    using namespace diskann::ivf;
    try {
        ivf_cli::Arguments args(argc, argv, {
            "--validate_inputs", "--data_type", "--metric", "--dim", "--data_file",
            "--index_prefix", "--pq_prefix", "--base_count", "--nlist", "--sample_rate",
            "--max_reps", "--seed", "--build_threads", "--memory_budget_gb"});
        if (args.help) {
            std::cout << "IVF build scaffold (input validation only; no index is written).\n"
                "Usage: ivf_build --validate_inputs 1 --data_type float|int8|uint8\n"
                "  --data_file PATH --index_prefix PATH --pq_prefix PATH --dim N\n"
                "  --base_count N --nlist N [--sample_rate F] [--max_reps N]\n"
                "  [--seed N] [--build_threads N] [--memory_budget_gb F] [--metric l2]\n";
            return 0;
        }
        args.require_validation_mode();
        if (args.get("--metric", "l2") != "l2") throw std::invalid_argument("IVF currently supports only metric l2");
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
        config.validate();
        const auto vectors = inspect_vectors(args.require("--data_file"), config.data_type, config.dim);
        if (config.base_count > vectors.rows) throw std::invalid_argument("base_count exceeds vector file rows");
        const std::string pq_prefix = args.require("--pq_prefix");
        const auto pq = inspect_pq_pivots(pq_prefix + "_pq_pivots.bin");
        if (pq.dim != config.dim) throw std::invalid_argument("PQ dimension mismatch");
        inspect_pq_codes(pq_prefix + "_pq_compressed.bin", config.base_count, pq.chunks);
        ivf_cli::validation_record("build", data_type_name(config.data_type), config.dim, config.base_count, pq.chunks);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
