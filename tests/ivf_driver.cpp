#include "bufann/ivf_config.h"
#include "bufann/ivf_io.h"
#include "ivf_cli.h"

int main(int argc, char** argv) {
    using namespace diskann::ivf;
    try {
        ivf_cli::Arguments args(argc, argv, {
            "--validate_inputs", "--data_type", "--metric", "--dim", "--index_prefix",
            "--pq_prefix", "--workload", "--query_file", "--gt_file", "--recall_at",
            "--query_threads", "--nlist", "--nprobe", "--rerank_factor",
            "--full_data_file", "--insert_ids_file", "--delete_ids_file"});
        if (args.help) {
            std::cout << "IVF driver scaffold (input validation only; no queries/updates execute).\n"
                "Usage: ivf_driver --validate_inputs 1 --data_type float|int8|uint8\n"
                "  --index_prefix PATH --dim N --nlist N --query_file PATH\n"
                "  --workload search_only|insert_only|delete_only|search_after_update\n"
                "  [--pq_prefix PATH] [--gt_file PATH] [--recall_at N] [--query_threads N]\n"
                "  [--nprobe N] [--rerank_factor F] [--metric l2]\n"
                "  [--full_data_file PATH --insert_ids_file PATH --delete_ids_file PATH]\n";
            return 0;
        }
        args.require_validation_mode();
        if (args.get("--metric", "l2") != "l2") throw std::invalid_argument("IVF currently supports only metric l2");
        const auto type = parse_data_type(args.get("--data_type", "float"));
        const uint32_t dim = args.u32("--dim");
        if (dim == 0) throw std::invalid_argument("dim must be positive");
        SearchConfig config;
        config.nprobe = args.u32("--nprobe", 1);
        config.recall_at = args.u32("--recall_at", 10);
        config.query_threads = args.u32("--query_threads", 1);
        config.rerank_factor = args.real("--rerank_factor", 2.0);
        config.validate(args.u32("--nlist"));
        const std::string prefix = args.require("--index_prefix");
        const std::string workload = args.require("--workload");
        if (workload != "search_only" && workload != "insert_only" &&
            workload != "delete_only" && workload != "search_after_update") {
            throw std::invalid_argument("unsupported IVF workload");
        }
        const auto queries = inspect_vectors(args.require("--query_file"), type, dim);
        if (!args.get("--gt_file").empty()) inspect_truth(args.get("--gt_file"), queries.rows, config.recall_at);
        const auto pq = inspect_pq_pivots(args.get("--pq_prefix", prefix) + "_pq_pivots.bin");
        if (pq.dim != dim) throw std::invalid_argument("PQ dimension mismatch");
        if (workload != "search_only" || !args.get("--full_data_file").empty()) {
            const auto vectors = inspect_vectors(args.require("--full_data_file"), type, dim);
            if (workload == "insert_only" || workload == "search_after_update") {
                load_counted_ids(args.require("--insert_ids_file"), vectors.rows);
            }
            if (workload == "delete_only" || workload == "search_after_update") {
                load_counted_ids(args.require("--delete_ids_file"), vectors.rows);
            }
        }
        ivf_cli::validation_record(workload.c_str(), data_type_name(type), dim, queries.rows, pq.chunks);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
