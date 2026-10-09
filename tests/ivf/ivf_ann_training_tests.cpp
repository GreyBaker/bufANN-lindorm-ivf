#include "bufann/ivf_router.h"
#include "bufann/ivf_patched_graph.h"
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
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f) { bool failed = false; try { f(); } catch (const std::exception&) { failed = true; } require(failed, "Expected rejection"); }
template<class T> void vectors(const std::string& path, uint32_t rows, uint32_t dim, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary); uint32_t header[]{rows, dim};
    out.write(reinterpret_cast<char*>(header), 8);
    out.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(T));
    out.close(); require(bool(out), "Fixture write failed");
}
template<class T> void typed(const std::filesystem::path& root, DataType dtype) {
    BuildConfig config; config.data_type = dtype; config.dim = 13; config.base_count = 192;
    config.nlist = 8; config.max_reps = 3; config.build_threads = 1; config.sample_rate = .75; config.seed = 41;
    config.training_breadth = 32;
    std::vector<T> values(208*config.dim);
    for (uint32_t r = 0; r < 208; ++r) for (uint32_t d = 0; d < config.dim; ++d)
        values[size_t(r)*config.dim+d] = T((r%8)*12+(r/8)%7+d%3);
    if constexpr (std::is_same_v<T,float>)
        std::fill(values.begin()+size_t(config.base_count)*config.dim, values.end(), std::numeric_limits<float>::quiet_NaN());
    auto path = (root/(std::string(data_type_name(dtype))+".bin")).string(); vectors(path, 208, config.dim, values);
    RouterOptions options; options.degree = 4; options.construction_breadth = 32;
    for (auto init : {CentroidInit::Sample, CentroidInit::RandomPartition}) {
        config.init = init; std::vector<uint32_t> ids;
        for (auto mode : {TrainerMode::Exact, TrainerMode::AnnRebuild, TrainerMode::AnnReuse, TrainerMode::AnnReuseSeeded}) {
            config.trainer_mode = mode; config.build_threads = 1;
            const auto trained = train_centroids_ann(path, config, options);
            config.build_threads = 4;
            const auto parallel = train_centroids_ann(path, config, options);
            require(trained.centroids == parallel.centroids && trained.assignments == parallel.assignments &&
                trained.objective_history == parallel.objective_history, "Training depends on assignment worker order");
            require(trained.sampled_ids == parallel.sampled_ids, "Sample IDs depend on workers");
            if (ids.empty()) ids = trained.sampled_ids; else require(ids == trained.sampled_ids, "Controls sample different IDs");
            require(*std::max_element(ids.begin(),ids.end()) < config.base_count, "Tail trained");
            const bool reuse = mode == TrainerMode::AnnReuse || mode == TrainerMode::AnnReuseSeeded;
            require(trained.assignment_index_builds == (mode == TrainerMode::Exact ? 0 : reuse ? 1 : 4), "Graph build count");
            require(trained.assignment_index_refreshes == (reuse ? 3 : 0) && trained.reinserted_centroids == (reuse ? 24 : 0), "Graph reuse count");
            require(trained.seeded_searches == (mode == TrainerMode::AnnReuseSeeded ? 432 : 0) &&
                trained.seed_fallbacks == (mode == TrainerMode::AnnReuseSeeded ? 144 : 0), "Previous assignment/fallback counts");
            require(trained.passes.size() == 4 && trained.passes.back().mean_seconds == 0, "Final pass missing");
            std::vector<uint32_t> occupancy(config.nlist); double assigned = 0, exact_sse = 0;
            for (size_t r = 0; r < ids.size(); ++r) {
                const auto label = trained.assignments[r]; require(label < config.nlist, "Bad final label"); ++occupancy[label];
                double nearest = std::numeric_limits<double>::infinity();
                for (uint32_t c = 0; c < config.nlist; ++c) {
                    double distance = 0;
                    for (uint32_t d = 0; d < config.dim; ++d) {
                        const double delta = double(values[size_t(ids[r])*config.dim+d])-trained.centroids[size_t(c)*config.dim+d];
                        distance += delta*delta;
                    }
                    nearest = std::min(nearest,distance); if (c == label) assigned += distance;
                }
                exact_sse += nearest;
            }
            require(occupancy == trained.occupancy && assigned == trained.final_objective && exact_sse <= assigned+1e-9,
                "Final occupancy/SSE does not describe returned centroids");
            require(std::isfinite(assigned), "Nonfinite SSE");
            if (mode == TrainerMode::Exact) require(exact_sse == assigned, "Exact oracle mismatch");
        }
    }
    config.trainer_mode = TrainerMode::AnnReuseSeeded;
    rejects([&] { train_centroids_ann(path, config, options, {1,1,1}); });
    config.memory_budget_gb = .00001; rejects([&] { train_centroids_ann(path, config, options); });
    config.memory_budget_gb = 1; config.training_breadth = 0; rejects([&] { train_centroids_ann(path, config, options); });
}
void raw_seed_and_reinsert() {
    ivf_hnswlib::L2Space space(2);
    ivf_hnswlib::HierarchicalNSW<float> graph(&space, 3, 2, 8, 10);
    const float values[]{0,0, 10,10, 20,20};
    for (uint32_t id = 0; id < 3; ++id) graph.addPoint(values+id*2,id);
    graph.setEf(1);
    graph.replaceBottomIVF(0,{}); graph.replaceBottomIVF(1,{}); graph.replaceBottomIVF(2,{});
    graph.enterpoint_node_ = 2; graph.maxlevel_ = 0;
    const auto global = graph.searchKnnIVF(values,1);
    const auto seeded = graph.searchKnnIVF(values,1,nullptr,nullptr,{0},true);
    require(global.top().second == 2 && seeded.top().second == 0, "Seeded traversal still uses global descent");
    require(graph.searchKnnIVF(values,1,nullptr,nullptr,{},true).top().second == 2, "Missing seed fallback");
    rejects([&] { graph.searchKnnIVF(values,1,nullptr,nullptr,{3},true); });
    graph.maxlevel_ = *std::max_element(graph.element_levels_.begin(),graph.element_levels_.begin()+3);
    for (uint32_t id = 0; id < 3; ++id) if (graph.element_levels_[id] == graph.maxlevel_) graph.enterpoint_node_ = id;
    auto levels = graph.element_levels_;
    const float moved[]{20,20, 0,0, 10,10};
    for (uint32_t id = 0; id < 3; ++id) graph.replaceVectorIVF(id,moved+id*2);
    for (uint32_t id = 0; id < 3; ++id) graph.reinsertFromIVF(id,{(id+1)%3});
    require(levels == graph.element_levels_ && graph.cur_element_count == 3, "Reuse changed levels or IDs");
    for (uint32_t id = 0; id < 3; ++id) {
        const auto* data = reinterpret_cast<const float*>(graph.getDataByInternalId(id));
        require(data[0] == moved[id*2] && data[1] == moved[id*2+1] && graph.getExternalLabel(id) == id, "Coordinates/labels stale");
    }
}
void joined_refresh(const std::filesystem::path& root) {
    BuildConfig config; config.dim = 1; config.base_count = 64; config.nlist = 2; config.max_reps = 3; config.build_threads = 4;
    config.init = CentroidInit::RandomPartition;
    const auto path = (root/"joined.bin").string(); vectors(path,64,1,std::vector<float>(64,1));
    struct Index final : detail::AssignmentIndex {
        mutable std::atomic<uint32_t> active{0}; uint32_t pass = 0;
        uint32_t assign(std::span<const float>, uint32_t previous, uint64_t&) const override {
            active.fetch_add(1);
            require(previous == (pass ? 1u : UINT32_MAX), "Previous assignment not supplied");
            active.fetch_sub(1); return 1;
        }
        void refresh(std::span<const float>) override { require(active.load() == 0, "Refresh overlaps assignment"); ++pass; }
    };
    const detail::AssignmentFactory factory = [](std::span<const float>) { return std::make_unique<Index>(); };
    const auto trained = detail::train_centroids_impl(path,config,{1,10000,10000},&factory,4096,true,true);
    require(trained.empty_recoveries == 3 && trained.seeded_searches == 192, "Empty recovery/seed loop");
}
}
int main(int argc,char** argv) {
    try {
        require(argc == 2,"scratch required"); const std::filesystem::path root(argv[1]); std::filesystem::create_directories(root);
        raw_seed_and_reinsert(); joined_refresh(root);
        typed<float>(root,DataType::Float); typed<int8_t>(root,DataType::Int8); typed<uint8_t>(root,DataType::UInt8);
        const auto path=(root/"singleton.bin").string(); vectors(path,4,1,std::vector<float>(4,2));
        BuildConfig c; c.dim=1; c.base_count=4; c.nlist=1; c.max_reps=2; c.trainer_mode=TrainerMode::AnnReuseSeeded; c.init=CentroidInit::RandomPartition;
        require(train_centroids_ann(path,c).final_objective == 0,"Singleton reuse");
        c.nlist=4; const auto duplicate=train_centroids_ann(path,c);
        require(duplicate.empty_recoveries > 0 && duplicate.final_objective == 0,"Duplicate recovery");
        std::cout << "ANN reuse/seeded training fixtures passed (all dtypes)\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
