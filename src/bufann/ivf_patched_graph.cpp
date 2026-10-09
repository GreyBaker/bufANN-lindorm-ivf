#include "bufann/ivf_patched_graph.h"
#include <ivf_patched/hnswlib.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace diskann::ivf {
namespace {
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
using Graph = ivf_hnswlib::HierarchicalNSW<float>;
}
struct PatchedCentroidGraph::Impl {
    explicit Impl(uint32_t dim) : space(dim) {}
    ivf_hnswlib::L2Space space;
    std::unique_ptr<ivf_hnswlib::L2Space> projection;
    uint32_t projection_dim = 0;
    std::unique_ptr<Graph> graph;
};
PatchedCentroidGraph::PatchedCentroidGraph(uint32_t dim, uint32_t count, PatchedGraphOptions options)
    : dim_(dim), count_(count), options_(options), impl_(std::make_unique<Impl>(dim)) {}
PatchedCentroidGraph::~PatchedCentroidGraph() = default;
void PatchedCentroidGraph::configure_projection(uint32_t dim) {
    check(dim > 0 && dim <= dim_, "Invalid frozen projection dimension");
    impl_->projection = std::make_unique<ivf_hnswlib::L2Space>(dim); impl_->projection_dim = dim;
}
std::unique_ptr<PatchedCentroidGraph> PatchedCentroidGraph::build(uint32_t dim, std::span<const float> vectors, PatchedGraphOptions options) {
    check(dim && !vectors.empty() && vectors.size() % dim == 0, "Invalid patched graph shape");
    auto result = std::unique_ptr<PatchedCentroidGraph>(new PatchedCentroidGraph(dim, uint32_t(vectors.size()/dim), options));
    result->impl_->graph = std::make_unique<Graph>(&result->impl_->space, result->count_, options.degree, options.breadth, options.seed);
    for (uint32_t id = 0; id < result->count_; ++id) result->impl_->graph->addPoint(vectors.data() + size_t(id)*dim, id);
    result->impl_->graph->setEf(1); result->measure(); result->stats_.unreachable_before = result->stats_.unreachable_after;
    return result;
}
std::unique_ptr<PatchedCentroidGraph> PatchedCentroidGraph::load(const std::string& path, uint32_t dim,
    std::span<const float> vectors, PatchedGraphOptions options, PatchedGraphDiagnostics diagnostics) {
    auto result = std::unique_ptr<PatchedCentroidGraph>(new PatchedCentroidGraph(dim, uint32_t(vectors.size()/dim), options));
    result->impl_->graph = std::make_unique<Graph>(&result->impl_->space, path, false, result->count_);
    result->impl_->graph->setEf(1); result->measure();
    check(result->stats_.unreachable_after == diagnostics.unreachable_after && result->stats_.zero_indegree == diagnostics.zero_indegree,
        "Patched graph connectivity diagnostics mismatch");
    result->stats_ = diagnostics; return result;
}
void PatchedCentroidGraph::measure() {
    auto& g = *impl_->graph; std::vector<uint32_t> indegree(count_); std::vector<uint8_t> visited(count_);
    for (uint32_t id = 0; id < count_; ++id) {
        auto* links = g.get_linklist0(id);
        for (uint32_t j = 0; j < g.getListCount(links); ++j) ++indegree[reinterpret_cast<uint32_t*>(links + 1)[j]];
    }
    stats_.zero_indegree = uint32_t(std::count(indegree.begin(), indegree.end(), 0));
    std::vector<uint32_t> queue{g.enterpoint_node_}; visited[queue[0]] = 1;
    for (size_t at = 0; at < queue.size(); ++at) {
        auto* links = g.get_linklist0(queue[at]);
        for (uint32_t j = 0; j < g.getListCount(links); ++j) {
            auto id = reinterpret_cast<uint32_t*>(links + 1)[j]; if (!visited[id]) { visited[id] = 1; queue.push_back(id); }
        }
    }
    stats_.unreachable_after = count_ - uint32_t(queue.size());
}
std::vector<PatchedGraphHit> PatchedCentroidGraph::search(std::span<const float> query, uint32_t count, uint32_t breadth,
    std::span<const uint32_t> entries, std::span<const float> projected, uint32_t projected_dim, uint64_t* computations, bool seeded) const {
    check(count && count <= count_ && breadth && breadth <= INT32_MAX && entries.size() <= count_, "Invalid patched graph search");
    check(projected.empty() ? query.size() == dim_ : projected_dim && query.size() == projected_dim && projected.size() == uint64_t(count_)*projected_dim,
        "Invalid projected traversal shape");
    for (auto entry : entries) check(entry < count_, "Invalid patched graph entry");
    uint64_t work = 0; Graph::IVFTraversal traversal;
    traversal.computations = &work;
    if (!projected.empty()) {
        check(impl_->projection && impl_->projection_dim == projected_dim, "Projection distance kernel not configured");
        traversal.vectors = reinterpret_cast<const char*>(projected.data()); traversal.stride = projected_dim*sizeof(float);
        traversal.distance = impl_->projection->get_dist_func(); traversal.parameter = impl_->projection->get_dist_func_param();
    }
    // k supplies per-call ef without mutating the shared graph's ef_.
    auto hits = impl_->graph->searchKnnIVF(query.data(), std::min(count_, std::max(count, breadth)), nullptr, &traversal,
        std::vector<uint32_t>(entries.begin(), entries.end()), seeded);
    if (computations) *computations = work;
    std::vector<PatchedGraphHit> result;
    while (!hits.empty()) { result.push_back({uint32_t(hits.top().second), hits.top().first}); hits.pop(); }
    std::sort(result.begin(), result.end(), [](auto a, auto b) { return a.distance < b.distance || (a.distance == b.distance && a.id < b.id); });
    if (result.size() > count) result.resize(count);
    return result;
}
void PatchedCentroidGraph::refresh(std::span<const float> vectors) {
    check(!impl_->projection && vectors.size() == uint64_t(count_)*dim_, "Invalid training coordinate refresh");
    for (float value : vectors) check(std::isfinite(value), "Nonfinite training centroid");
    auto& g = *impl_->graph;
    // This allowance is separately included in the trainer's preflight estimate.
    const uint64_t allowance = uint64_t(count_) * (uint64_t(options_.degree)*16 + 256);
    uint64_t bytes = uint64_t(count_)*sizeof(std::vector<uint32_t>);
    std::vector<std::vector<uint32_t>> old(count_);
    for (uint32_t id = 0; id < count_; ++id) {
        auto& entries = old[id];
        size_t size = 0;
        for (int level = 0; level <= g.element_levels_[id]; ++level)
            size += g.getListCount(g.get_linklist_at_level(id, level));
        check(bytes <= allowance && size*sizeof(uint32_t) <= allowance-bytes, "Training neighbor snapshot exceeds allowance");
        bytes += size*sizeof(uint32_t); entries.reserve(size);
        for (int level = 0; level <= g.element_levels_[id]; ++level) {
            auto* links = g.get_linklist_at_level(id, level);
            auto* ids = reinterpret_cast<uint32_t*>(links + 1);
            entries.insert(entries.end(), ids, ids + g.getListCount(links));
        }
    }
    // No traversal sees a mixture of old/new centroid coordinates.
    for (uint32_t id = 0; id < count_; ++id) g.replaceVectorIVF(id, vectors.data() + size_t(id)*dim_);
    for (uint32_t id = 0; id < count_; ++id) g.reinsertFromIVF(id, old[id]);
    measure();
}
void PatchedCentroidGraph::refine(uint32_t breadth) {
    check(breadth > 0 && breadth <= INT32_MAX, "Invalid refinement breadth");
    auto& g = *impl_->graph;
    std::vector<std::vector<uint32_t>> proposed(count_); uint64_t work = 0;
    for (uint32_t id = 0; id < count_; ++id) {
        uint64_t distances = 0;
        auto hits = search({reinterpret_cast<const float*>(g.getDataByInternalId(id)), dim_}, std::min(count_, breadth + 1), breadth, {}, {}, 0, &distances);
        check(distances <= options_.work_limit - work, "Refinement work allowance exceeded"); work += distances;
        std::priority_queue<std::pair<float,uint32_t>, std::vector<std::pair<float,uint32_t>>, Graph::CompareByFirst> candidates;
        for (auto hit : hits) if (hit.id != id) candidates.emplace(float(hit.distance), hit.id);
        g.getNeighborsByHeuristic2(candidates, g.maxM0_);
        while (!candidates.empty()) { proposed[id].push_back(candidates.top().second); candidates.pop(); }
    }
    // Every search/prune above reads the frozen pre-refinement graph. Out-edge
    // replacement adds no reciprocal links; reinsertion below is a separate step.
    for (uint32_t id = 0; id < count_; ++id) g.replaceBottomIVF(id, proposed[id]);
    stats_.refined_nodes = count_; measure(); stats_.unreachable_before = stats_.unreachable_after;
    std::vector<uint32_t> indegree(count_);
    for (const auto& links : proposed) for (auto id : links) ++indegree[id];
    for (uint32_t id = 0; id < count_; ++id) if (!indegree[id] && count_ > 1) {
        g.reinsertIVF(id); ++stats_.reinserted_nodes;
    }
    measure(); // A measurement, never an enforced connectedness gate.
}
void PatchedCentroidGraph::save(const std::string& path) const { impl_->graph->saveIndex(path); }
} // namespace diskann::ivf
