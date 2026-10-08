#include "bufann/ivf_router.h"
#include "bufann/ivf_centroids.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<typename F> void rejects(F&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "expected router rejection");
}
void verify_hits(const std::vector<diskann::ivf::CentroidHit>& hits, uint32_t count, uint32_t nlist) {
    require(hits.size() == count, "probe count mismatch");
    std::set<uint32_t> ids;
    double last = -1;
    for (auto hit : hits) {
        require(hit.list_id < nlist && ids.insert(hit.list_id).second &&
                std::isfinite(hit.squared_distance) && hit.squared_distance >= last,
                "invalid/duplicate/unsorted centroid results");
        last = hit.squared_distance;
    }
}
template<typename T>
void training_comparison(const std::filesystem::path& root, diskann::ivf::DataType type) {
    using namespace diskann::ivf;
    const auto path = (root / ("ann_training_" + std::string(data_type_name(type)) + ".bin")).string();
    const uint32_t header[]{256, 13};
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(header), sizeof(header));
        for (uint32_t r = 0; r < header[0]; ++r)
            for (uint32_t d = 0; d < header[1]; ++d) {
                T value = T(int(r % 8) * 12 + int((r + d) % 5) - (std::is_same_v<T, uint8_t> ? 0 : 40));
                if constexpr (std::is_same_v<T, float>)
                    if (r >= 240) value = std::numeric_limits<float>::infinity();
                out.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
    }
    BuildConfig config; config.data_type = type; config.dim = 13; config.base_count = 240;
    config.nlist = 8; config.max_reps = 5; config.sample_rate = .8; config.seed = 777;
    RouterOptions options; options.insert_breadth = 8;
    auto exact = train_centroids(path, config);
    auto ann = train_centroids_hnsw(path, config, options);
    config.build_threads = 4;
    const auto threaded = train_centroids_hnsw(path, config, options, {3, 1000000});
    require(ann.centroids == threaded.centroids && ann.sampled_ids == threaded.sampled_ids &&
            ann.occupancy == threaded.occupancy && ann.objective_history == threaded.objective_history &&
            ann.final_objective == threaded.final_objective, "HNSW trainer changed across workers/blocks");
    require(ann.centroids == exact.centroids && ann.final_objective == exact.final_objective &&
            ann.occupancy == exact.occupancy, "high-breadth tiny training diverged from exact assignment");
    require(ann.approximate_assignment && !ann.distance_evaluations &&
            ann.assignment_searches == 192 * 6 && ann.assignment_index_builds == 6 &&
            ann.estimated_bytes > exact.estimated_bytes, "incorrect ANN trainer work/memory diagnostics");
    // HNSW is not constrained by the exact all-centroid distance budget.
    require(train_centroids_hnsw(path, config, options, {16, 1, 2000}).assignment_searches == 192 * 6,
            "ANN trainer reused exact work cap");
    rejects([&] { train_centroids_hnsw(path, config, options, {1, 1000000, 10}); });
    config.build_threads = 1;
    // Combined sample/trainer/graph allowance must fit, not just either part.
    config.memory_budget_gb = double(exact.estimated_bytes) / (1024 * 1024 * 1024);
    rejects([&] { train_centroids_hnsw(path, config, options); });
}
void exercise(const std::filesystem::path& root) {
    using namespace diskann::ivf;
    constexpr uint32_t dim = 13, nlist = 1024;
    std::mt19937 rng(12345);
    auto random_value = [&] { return float(int(rng() % 10001) - 5000) / 100; };
    std::vector<float> centroids(size_t(dim) * nlist);
    for (auto& x : centroids) x = random_value();
    RouterOptions options;
    options.query_breadth = 192; options.insert_breadth = 96;
    auto router = CentroidRouter::build(dim, centroids, options);
    require(router->dim() == dim && router->nlist() == nlist && router->estimated_bytes() < options.memory_budget_bytes,
            "router shape/estimate mismatch");
    std::array<float, dim> query{};
    uint32_t found = 0, total = 0;
    for (uint32_t q = 0; q < 40; ++q) {
        for (auto& x : query) x = random_value();
        const auto approx = router->search(query, 10);
        const auto exact = router->search(query, 10, true);
        verify_hits(approx, 10, nlist);
        verify_hits(exact, 10, nlist);
        for (auto hit : approx) {
            found += std::any_of(exact.begin(), exact.end(), [&](auto truth) { return hit.list_id == truth.list_id; });
            ++total;
        }
    }
    const double recall = double(found) / total;
    std::cout << "Synthetic centroid recall@10: " << recall << " (40 queries, 1024 centroids)\n";
    require(recall >= .95, "synthetic router recall below fixture gate");
    const auto all = router->search(query, nlist);
    require(all == router->search(query, nlist, true), "all-list enumeration differs from exact");
    verify_hits(all, nlist, nlist);
    require(router->assign(query, true) == all.front().list_id, "exact insertion route mismatch");
    const auto self = std::span<const float>(centroids).subspan(123 * dim, dim);
    require(router->assign(self) == 123 && router->search(self, 1).front().squared_distance == 0,
            "centroid labels changed");
    std::array<int8_t, dim> signed_query{};
    std::array<uint8_t, dim> unsigned_query{};
    signed_query.fill(-12); unsigned_query.fill(12);
    std::array<float, dim> converted{}; converted.fill(-12);
    require(router->search(std::span<const int8_t>(signed_query), 5) == router->search(converted, 5),
            "int8 routing conversion mismatch");
    converted.fill(12);
    require(router->assign(std::span<const uint8_t>(unsigned_query)) == router->assign(converted),
            "uint8 insertion conversion mismatch");
    const auto prefix = (root / "router").string();
    router->save(prefix);
    auto loaded = CentroidRouter::load(prefix);
    require(loaded->search(query, 20) == router->search(query, 20) &&
            loaded->assign(query) == router->assign(query) && loaded->options().insert_breadth == 96,
            "router persistence changed results/settings");
    // Serial build with identical seed/order must persist the same graph.
    auto duplicate = CentroidRouter::build(dim, centroids, options);
    const auto other = (root / "duplicate").string(); duplicate->save(other);
    auto bytes = [](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    };
    require(bytes(prefix + ".hnsw") == bytes(other + ".hnsw"), "serial HNSW build is nondeterministic");
    const auto expected = loaded->search(query, 17);
    const auto expected_insert = loaded->assign(query);
    std::array<std::exception_ptr, 8> errors{};
    std::vector<std::thread> readers;
    for (size_t t = 0; t < errors.size(); ++t) readers.emplace_back([&, t] {
        try {
            for (uint32_t i = 0; i < 40; ++i)
                require(loaded->search(query, 17) == expected && loaded->assign(query) == expected_insert,
                        "concurrent query/insert routing changed results");
        } catch (...) { errors[t] = std::current_exception(); }
    });
    for (auto& thread : readers) thread.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    // Exercise framing/edge validation independently of the checksum guard.
    // These files are deliberately rehashed; FNV is not authentication.
    auto rehash_graph = [&](const std::string& target) {
        auto content = bytes(target + ".hnsw");
        uint64_t hash = 14695981039346656037ull;
        for (unsigned char byte : content) { hash ^= byte; hash *= 1099511628211ull; }
        std::istringstream input(bytes(target + ".router"));
        std::vector<std::string> tokens;
        for (std::string token; input >> token;) tokens.push_back(token);
        require(tokens.size() == 13, "unexpected descriptor framing");
        tokens[11] = std::to_string(content.size()); tokens[12] = std::to_string(hash);
        std::ofstream output(target + ".router", std::ios::binary);
        for (auto& token : tokens) output << token << ' ';
    };
    for (uint32_t fault = 0; fault < 4; ++fault) {
        const auto damaged = (root / ("rehashed_" + std::to_string(fault))).string();
        for (const auto& suffix : {".router", ".hnsw", ".centroids.bin"})
            std::filesystem::copy_file(prefix + suffix, damaged + suffix);
        std::fstream output(damaged + ".hnsw", std::ios::binary | std::ios::in | std::ios::out);
        if (fault == 0) { // Invalid allocation stride in native header.
            const uint64_t huge = UINT64_MAX; output.seekp(24);
            output.write(reinterpret_cast<const char*>(&huge), sizeof(huge));
        } else if (fault == 1) { // Out-of-range level-0 target.
            const uint32_t one = 1, invalid_id = nlist;
            output.seekp(96); output.write(reinterpret_cast<const char*>(&one), sizeof(one));
            output.write(reinterpret_cast<const char*>(&invalid_id), sizeof(invalid_id));
        } else if (fault == 2) { // Invalid label despite unchanged centroid table.
            const uint64_t invalid_label = nlist;
            output.seekp(96 + 4 + 8 * options.degree + 4 * dim);
            output.write(reinterpret_cast<const char*>(&invalid_label), sizeof(invalid_label));
        } else { // Implausible upper-layer allocation length.
            const uint32_t huge = UINT32_MAX;
            output.seekp(96 + uint64_t(nlist) * (4 + 8 * options.degree + 4 * dim + 8));
            output.write(reinterpret_cast<const char*>(&huge), sizeof(huge));
        }
        output.close(); rehash_graph(damaged);
        rejects([&] { CentroidRouter::load(damaged); });
    }
    rejects([&] { router->search(query, 0); });
    rejects([&] { router->search(query, nlist + 1); });
    rejects([&] { router->search(std::span<const float>(query).first(12), 1); });
    rejects([&] { router->assign(std::span<const int8_t>(signed_query).first(12)); });
    query[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { router->search(query, 1); });
    query[0] = std::numeric_limits<float>::max();
    rejects([&] { router->assign(query); });
    auto invalid = options; invalid.degree = 1;
    rejects([&] { CentroidRouter::build(dim, centroids, invalid); });
    invalid = options; invalid.construction_breadth = 1;
    rejects([&] { CentroidRouter::build(dim, centroids, invalid); });
    invalid = options; invalid.memory_budget_bytes = 1;
    rejects([&] { CentroidRouter::build(dim, centroids, invalid); });
    rejects([&] { CentroidRouter::build(0, centroids); });
    centroids[0] = std::numeric_limits<float>::infinity();
    rejects([&] { CentroidRouter::build(dim, centroids); });
    rejects([&] { CentroidRouter::load(prefix, 1); });
    rejects([&] { router->save(prefix); });
    const auto collision = (root / "collision").string();
    { std::ofstream out(collision + ".router"); out << "preserve"; }
    rejects([&] { router->save(collision); });
    require(bytes(collision + ".router") == "preserve" &&
            !std::filesystem::exists(collision + ".centroids.bin"), "save overwrote/created conflicting artifacts");
    // Same-size corruption must be rejected before invoking the native loader.
    { std::fstream out(prefix + ".hnsw", std::ios::binary | std::ios::in | std::ios::out); out.put('x'); }
    rejects([&] { CentroidRouter::load(prefix); });
    { std::ofstream out(other + ".centroids.bin", std::ios::binary | std::ios::app); out.put('x'); }
    rejects([&] { CentroidRouter::load(other); });
    std::vector<float> identical(13 * 7, 1);
    auto ties = CentroidRouter::build(13, identical);
    std::array<float, 13> ones{}; ones.fill(1);
    auto tied = ties->search(ones, 7);
    for (uint32_t i = 0; i < 7; ++i)
        require(tied[i].list_id == i && tied[i].squared_distance == 0, "all-list tie order invalid");
    auto singleton = CentroidRouter::build(13, std::span<const float>(ones));
    const auto single_prefix = (root / "singleton").string(); singleton->save(single_prefix);
    require(CentroidRouter::load(single_prefix)->assign(ones) == 0, "singleton graph failed");
    // End-to-end component seam: base-only trainer output feeds the router.
    const auto data = (root / "training.bin").string();
    { std::ofstream out(data, std::ios::binary);
      const uint32_t header[]{32, 13}; out.write(reinterpret_cast<const char*>(header), sizeof(header));
      for (uint32_t r = 0; r < 32; ++r) {
          ones.fill(float(r % 4) * 10); out.write(reinterpret_cast<const char*>(ones.data()), sizeof(ones));
      } }
    BuildConfig config; config.dim = 13; config.base_count = 32; config.nlist = 4;
    auto trained = train_centroids(data, config);
    auto from_training = CentroidRouter::build(config.dim, trained.centroids);
    require(from_training->search(ones, 4) == from_training->search(ones, 4, true), "trainer/router seam failed");
    const auto same_data = (root / "ann_identical.bin").string();
    {
        std::ofstream out(same_data, std::ios::binary);
        const uint32_t shape[]{32, dim}; out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        ones.fill(7);
        for (uint32_t r = 0; r < 32; ++r) out.write(reinterpret_cast<const char*>(ones.data()), sizeof(ones));
    }
    const auto identical_ann = train_centroids_hnsw(same_data, config);
    require(identical_ann.final_objective == 0 && identical_ann.empty_recoveries == 3 * config.max_reps &&
            identical_ann.occupancy == std::vector<uint32_t>({32, 0, 0, 0}), "ANN empty-cluster recovery failed");
    training_comparison<float>(root, DataType::Float);
    training_comparison<int8_t>(root, DataType::Int8);
    training_comparison<uint8_t>(root, DataType::UInt8);
    // Nontrivial low-breadth fixture: compare final assigned SSE, not just
    // router recall. The same sample and initialization are used in both modes.
    const auto many = (root / "ann_quality.bin").string();
    {
        std::ofstream out(many, std::ios::binary);
        const uint32_t shape[]{1024, dim}; out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        for (uint32_t r = 0; r < shape[0]; ++r) {
            for (auto& x : query) x = random_value();
            out.write(reinterpret_cast<const char*>(query.data()), sizeof(query));
        }
    }
    config.base_count = 1024; config.nlist = 64; config.max_reps = 4; config.build_threads = 4;
    options.insert_breadth = 8;
    const auto quality_exact = train_centroids(many, config);
    const auto quality_ann = train_centroids_hnsw(many, config, options);
    const double ratio = quality_ann.final_objective / quality_exact.final_objective;
    std::cout << "Synthetic HNSW/exact final sample SSE ratio: " << ratio << '\n';
    require(std::isfinite(ratio) && ratio <= 1.05, "ANN training quality exceeded fixture tolerance");
    require(std::accumulate(quality_ann.occupancy.begin(), quality_ann.occupancy.end(), uint32_t(0)) == 1024,
            "ANN final sample occupancy lost rows");
}
} // namespace
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected fixture directory");
        const auto root = std::filesystem::path(argv[1]) /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(root);
        exercise(root);
        std::cout << "IVF centroid router tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
