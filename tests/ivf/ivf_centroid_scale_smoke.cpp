// Explicitly invoked synthetic cardinality/resource smoke, NOT an IVF benchmark.
#include "bufann/ivf_router.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected smoke fixture directory");
        const auto root = std::filesystem::path(argv[1]) /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(root);
        const auto data = (root / "synthetic.bin").string();
        constexpr uint32_t rows = 100000, dim = 13, nlist = 50000;
        {
            std::ofstream out(data, std::ios::binary);
            const uint32_t header[]{rows, dim};
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            std::mt19937 rng(12345);
            for (uint32_t r = 0; r < rows; ++r) {
                float coordinates[dim];
                for (auto& x : coordinates) x = float(int(rng() % 100001) - 50000) / 500;
                out.write(reinterpret_cast<const char*>(coordinates), sizeof(coordinates));
            }
            out.close();
            if (!out) throw std::runtime_error("scale fixture write failed");
        }
        diskann::ivf::BuildConfig config;
        config.dim = dim; config.base_count = rows; config.nlist = nlist;
        config.max_reps = 2; config.build_threads = 4; config.seed = 12345;
        config.memory_budget_gb = 1;
        diskann::ivf::RouterOptions options;
        options.degree = 16; options.construction_breadth = 200; options.insert_breadth = 32;
        options.query_breadth = 128; options.seed = 100;
        options.memory_budget_bytes = 256ull * 1024 * 1024;
        // The exact reference must refuse 15 billion vector distances at its
        // default one-billion limit, before attempting any such assignment.
        bool refused = false;
        try { diskann::ivf::train_centroids(data, config); }
        catch (const std::invalid_argument& error) {
            refused = std::string(error.what()) == "Exact centroid assignment exceeds declared work limit";
        }
        if (!refused) throw std::runtime_error("large exact assignment was not refused");
        auto trained = diskann::ivf::train_centroids_hnsw(data, config, options);
        const auto start = std::chrono::steady_clock::now();
        auto router = diskann::ivf::CentroidRouter::build(dim, trained.centroids, options);
        const double router_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto prefix = (root / "centroids").string(); router->save(prefix);
        router.reset();
        auto reopened = diskann::ivf::CentroidRouter::load(prefix, options.memory_budget_bytes);
        float query[dim]{};
        const auto selected = reopened->search(query, 20);
        if (selected.size() != 20) throw std::runtime_error("scale router reopen failed");
        uint32_t empty = 0; uint64_t assigned = 0;
        for (auto occupancy : trained.occupancy) { empty += occupancy == 0; assigned += occupancy; }
        if (assigned != rows || trained.centroids.size() != size_t(dim) * nlist ||
            trained.assignment_index_builds != 3 || trained.distance_evaluations)
            throw std::runtime_error("scale training diagnostics invalid");
        std::cout << "{\"kind\":\"synthetic_centroid_scale_smoke\",\"rows\":" << rows
                  << ",\"dim\":" << dim << ",\"nlist\":" << nlist
                  << ",\"iterations\":2,\"workers\":4,\"trainer_budget_gb\":1"
                  << ",\"training_seconds\":" << trained.elapsed_seconds
                  << ",\"router_build_seconds\":" << router_seconds
                  << ",\"estimated_training_bytes\":" << trained.estimated_bytes
                  << ",\"assignment_searches\":" << trained.assignment_searches
                  << ",\"assignment_index_builds\":" << trained.assignment_index_builds
                  << ",\"graph_distance_evaluations\":null,\"final_sample_sse\":" << trained.final_objective
                  << ",\"empty_sample_clusters\":" << empty << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
