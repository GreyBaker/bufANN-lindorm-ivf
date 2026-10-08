#include "bufann/ivf_router.h"
#include "bufann/ivf_io.h"
#include "ivf_centroids_internal.h"

#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>

namespace diskann::ivf {
namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
uint64_t estimate(uint32_t dim, uint32_t count, const RouterOptions& options) {
    // Two centroid copies; level-0 links, a generous upper-level allowance,
    // maps/locks/allocator overhead; fixed label locks and one visited table.
    const uint64_t per_node = uint64_t(dim) * 8 + uint64_t(options.degree) * 16 + 256;
    const uint64_t fixed = 4ull * 1024 * 1024;
    check(count <= (UINT64_MAX - fixed) / per_node, "Router memory estimate overflow");
    return uint64_t(count) * per_node + fixed;
}
void validate(uint32_t dim, size_t cells, const RouterOptions& options) {
    check(dim > 0 && cells > 0 && cells % dim == 0 && cells / dim <= INT32_MAX,
          "Invalid centroid router shape");
    check(options.degree >= 2 && options.degree <= 64 &&
          options.construction_breadth >= options.degree &&
          options.construction_breadth <= INT32_MAX && options.query_breadth > 0 &&
          options.query_breadth <= INT32_MAX && options.insert_breadth > 0 &&
          options.insert_breadth <= INT32_MAX, "Invalid HNSW settings");
    check(estimate(dim, static_cast<uint32_t>(cells / dim), options) <= options.memory_budget_bytes,
          "Centroid router exceeds preflight memory allowance");
}
void validate_values(std::span<const float> values, uint32_t dim) {
    // HNSW uses float squared distances; finite inputs alone do not prevent
    // overflow. Leave a factor-of-two accumulation margin for SIMD rounding.
    const double bound = std::sqrt(double(std::numeric_limits<float>::max()) / (8.0 * dim));
    for (float value : values)
        check(std::isfinite(value) && std::abs(double(value)) <= bound,
              "Non-finite or excessive centroid-routing coordinate");
}
bool nearer(const CentroidHit& a, const CentroidHit& b) {
    return a.squared_distance != b.squared_distance ?
        a.squared_distance < b.squared_distance : a.list_id < b.list_id;
}
void claim(const std::string& path) {
    auto* file = std::fopen(path.c_str(), "wbx");
    check(file != nullptr, "Router artifact exists or cannot be created");
    check(std::fclose(file) == 0, "Router artifact close failed");
}
uint64_t hash_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    check(bool(input), "Cannot read router artifact");
    uint64_t hash = 14695981039346656037ull;
    char bytes[8192];
    while (input.read(bytes, sizeof(bytes)) || input.gcount())
        for (std::streamsize i = 0; i < input.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(bytes[i]);
            hash *= 1099511628211ull;
        }
    check(input.eof() && !input.bad(), "Router artifact read failed");
    return hash;
}
template<typename T>
T read_pod(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    check(bool(input), "Truncated HNSW artifact");
    return value;
}
template<typename T>
T packed(const char* bytes) {
    T value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

// Check native v0.8.0 framing/identity/edges BEFORE its legacy loader allocates
// or follows graph offsets. Sequential labels are guaranteed by serial build.
void validate_graph(const std::string& path, uint32_t dim, std::span<const float> centroids,
                    const RouterOptions& options) {
    std::ifstream input(path, std::ios::binary);
    const size_t count = centroids.size() / dim;
    const size_t base_links = 4 + 8 * options.degree;
    const size_t upper_links = 4 + 4 * options.degree;
    const size_t stride = base_links + size_t(dim) * 4 + 8;
    const size_t offset = read_pod<size_t>(input);
    const size_t capacity = read_pod<size_t>(input);
    const size_t stored_count = read_pod<size_t>(input);
    const size_t stored_stride = read_pod<size_t>(input);
    const size_t label_offset = read_pod<size_t>(input);
    const size_t data_offset = read_pod<size_t>(input);
    const int max_level = read_pod<int>(input);
    const uint32_t entry = read_pod<uint32_t>(input);
    const size_t max_m = read_pod<size_t>(input);
    const size_t max_m0 = read_pod<size_t>(input);
    const size_t m = read_pod<size_t>(input);
    const double mult = read_pod<double>(input);
    const size_t ef = read_pod<size_t>(input);
    check(offset == 0 && capacity == count && stored_count == count && stored_stride == stride &&
          label_offset == base_links + size_t(dim) * 4 && data_offset == base_links &&
          max_level >= 0 && max_level <= 64 && entry < count && max_m == options.degree &&
          max_m0 == 2 * options.degree && m == options.degree &&
          mult == 1.0 / std::log(double(options.degree)) && ef == options.construction_breadth,
          "Incompatible HNSW artifact header");
    auto validate_links = [&](const char* bytes, size_t maximum, int level,
                              const std::vector<uint32_t>* levels) {
        const uint32_t header = packed<uint32_t>(bytes);
        check(header <= maximum, "Invalid/deleted HNSW neighbor list");
        for (uint32_t j = 0; j < header; ++j) {
            const uint32_t target = packed<uint32_t>(bytes + 4 + size_t(j) * 4);
            check(target < count && (!levels || (*levels)[target] >= uint32_t(level)),
                  "Invalid HNSW neighbor ID/level");
        }
    };
    std::vector<char> buffer(stride);
    for (size_t id = 0; id < count; ++id) {
        input.read(buffer.data(), buffer.size());
        check(bool(input) && packed<uint64_t>(buffer.data() + label_offset) == id &&
              std::memcmp(buffer.data() + base_links, centroids.data() + id * dim, size_t(dim) * 4) == 0,
              "HNSW centroid/label mismatch");
        validate_links(buffer.data(), 2 * options.degree, 0, nullptr);
    }
    std::vector<uint32_t> levels(count);
    std::vector<std::streampos> offsets(count);
    const uint64_t file_size = std::filesystem::file_size(path);
    uint32_t observed_max = 0;
    for (size_t id = 0; id < count; ++id) {
        const uint32_t bytes = read_pod<uint32_t>(input);
        check(bytes % upper_links == 0 && bytes / upper_links <= uint32_t(max_level),
              "Invalid HNSW upper-layer length");
        levels[id] = static_cast<uint32_t>(bytes / upper_links);
        observed_max = std::max(observed_max, levels[id]);
        offsets[id] = input.tellg();
        check(uint64_t(offsets[id]) <= file_size && bytes <= file_size - uint64_t(offsets[id]),
              "Truncated HNSW upper layer");
        input.seekg(bytes, std::ios::cur);
    }
    check(uint64_t(input.tellg()) == file_size && observed_max == uint32_t(max_level) &&
          levels[entry] == observed_max, "Trailing/inconsistent HNSW layers");
    buffer.resize(upper_links);
    for (size_t id = 0; id < count; ++id) {
        input.seekg(offsets[id]);
        for (uint32_t level = 1; level <= levels[id]; ++level) {
            input.read(buffer.data(), buffer.size());
            check(bool(input), "Truncated HNSW links");
            validate_links(buffer.data(), options.degree, int(level), &levels);
        }
    }
}
} // namespace

struct CentroidRouter::Impl {
    explicit Impl(uint32_t dim) : space(dim) {}
    hnswlib::L2Space space; // Must outlive the index's distance parameters.
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;
};
CentroidRouter::CentroidRouter(uint32_t dim, std::span<const float> centroids,
                             const RouterOptions& options) : _dim(dim), _options(options) {
    static_assert(std::endian::native == std::endian::little && sizeof(size_t) == 8 &&
                  sizeof(hnswlib::labeltype) == 8 && sizeof(int) == 4 && sizeof(float) == 4);
    validate(dim, centroids.size(), options);
    validate_values(centroids, dim);
    _centroids.assign(centroids.begin(), centroids.end());
    _impl = std::make_unique<Impl>(dim);
}
CentroidRouter::~CentroidRouter() = default;
uint64_t CentroidRouter::estimated_bytes() const { return estimate(_dim, nlist(), _options); }
uint64_t CentroidRouter::estimate_build_bytes(uint32_t dim, uint32_t count, const RouterOptions& options) {
    const uint64_t cells = uint64_t(dim) * count;
    check(cells <= SIZE_MAX / sizeof(float), "Router centroid shape overflow");
    validate(dim, static_cast<size_t>(cells), options);
    return estimate(dim, count, options);
}

std::unique_ptr<CentroidRouter> CentroidRouter::build(uint32_t dim,
    std::span<const float> centroids, const RouterOptions& options) {
    auto router = std::unique_ptr<CentroidRouter>(new CentroidRouter(dim, centroids, options));
    router->_impl->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        &router->_impl->space, router->nlist(), options.degree,
        options.construction_breadth, options.seed);
    for (uint32_t id = 0; id < router->nlist(); ++id)
        router->_impl->graph->addPoint(router->_centroids.data() + size_t(id) * dim, id);
    router->_impl->graph->setEf(1); // Immutable. Per-operation breadth via candidate oversampling.
    return router;
}
std::vector<CentroidHit> CentroidRouter::route(std::span<const float> query, uint32_t count,
                                            uint32_t breadth, bool exact) const {
    check(query.size() == _dim && count > 0 && count <= nlist(), "Invalid centroid query/probe count");
    validate_values(query, _dim);
    auto distance = [&](uint32_t id) {
        double result = 0;
        for (uint32_t d = 0; d < _dim; ++d) {
            const double diff = double(query[d]) - _centroids[size_t(id) * _dim + d];
            result += diff * diff;
        }
        return result;
    };
    std::vector<CentroidHit> hits;
    if (exact || count == nlist()) {
        hits.reserve(nlist());
        for (uint32_t id = 0; id < nlist(); ++id) hits.push_back({id, distance(id)});
    } else {
        // Unmodified searchKnn uses max(ef_, k). Oversampling permits separate
        // query/insert breadth without racing setEf or duplicating the graph.
        auto candidates = _impl->graph->searchKnn(query.data(), std::min(nlist(), std::max(count, breadth)));
        check(candidates.size() >= count, "HNSW did not reach enough distinct centroids");
        hits.reserve(candidates.size());
        while (!candidates.empty()) {
            const auto id = candidates.top().second;
            check(id < nlist(), "Invalid HNSW centroid label");
            hits.push_back({static_cast<uint32_t>(id), distance(static_cast<uint32_t>(id))});
            candidates.pop();
        }
    }
    std::partial_sort(hits.begin(), hits.begin() + count, hits.end(), nearer);
    hits.resize(count);
    return hits;
}
std::vector<CentroidHit> CentroidRouter::search(std::span<const float> query, uint32_t nprobe,
                                             bool exact) const {
    return route(query, nprobe, _options.query_breadth, exact);
}
uint32_t CentroidRouter::assign(std::span<const float> vector, bool exact) const {
    return route(vector, 1, _options.insert_breadth, exact).front().list_id;
}

void CentroidRouter::save(const std::string& prefix) const {
    check(!prefix.empty(), "Empty router prefix");
    const auto vectors = prefix + ".centroids.bin", graph = prefix + ".hnsw", descriptor = prefix + ".router";
    // Check all collisions before creating anything. Exclusive claims also
    // protect against competing creators; partial artifacts remain on failure.
    for (const auto& path : {vectors, graph, descriptor})
        check(!std::filesystem::exists(std::filesystem::symlink_status(path)), "Router artifact already exists");
    claim(vectors); claim(graph); claim(descriptor);
    std::ofstream output(vectors, std::ios::binary);
    const uint32_t header[]{nlist(), _dim};
    output.write(reinterpret_cast<const char*>(header), sizeof(header));
    output.write(reinterpret_cast<const char*>(_centroids.data()), _centroids.size() * sizeof(float));
    output.close();
    check(bool(output), "Centroid artifact write failed");
    _impl->graph->saveIndex(graph);
    validate_graph(graph, _dim, _centroids, _options);
    std::ofstream meta(descriptor, std::ios::binary);
    meta.imbue(std::locale::classic());
    meta << "IVFROUTER1 hnswlib-0.8.0-nomv " << _dim << ' ' << nlist() << ' ' << _options.degree << ' '
         << _options.construction_breadth << ' ' << _options.query_breadth << ' ' << _options.insert_breadth
         << ' ' << _options.seed << ' ' << std::filesystem::file_size(vectors) << ' ' << hash_file(vectors)
         << ' ' << std::filesystem::file_size(graph) << ' ' << hash_file(graph) << '\n';
    meta.close();
    check(bool(meta), "Router descriptor write failed");
}
std::unique_ptr<CentroidRouter> CentroidRouter::load(const std::string& prefix, uint64_t budget) {
    const auto vectors = prefix + ".centroids.bin", graph = prefix + ".hnsw";
    std::ifstream meta(prefix + ".router", std::ios::binary);
    meta.imbue(std::locale::classic());
    std::string magic, backend;
    uint32_t dim = 0, count = 0;
    RouterOptions options;
    options.memory_budget_bytes = budget;
    uint64_t vector_bytes = 0, vector_hash = 0, graph_bytes = 0, graph_hash = 0;
    meta >> magic >> backend >> dim >> count >> options.degree >> options.construction_breadth
         >> options.query_breadth >> options.insert_breadth >> options.seed
         >> vector_bytes >> vector_hash >> graph_bytes >> graph_hash;
    check(bool(meta) && magic == "IVFROUTER1" && backend == "hnswlib-0.8.0-nomv" && count <= INT32_MAX,
          "Invalid router descriptor");
    meta >> std::ws;
    check(meta.eof(), "Trailing router descriptor data");
    const uint64_t cells = uint64_t(dim) * count;
    check(cells <= SIZE_MAX / sizeof(float), "Router centroid shape overflow");
    validate(dim, static_cast<size_t>(cells), options);
    check(vector_bytes == 8 + cells * sizeof(float) &&
          std::filesystem::file_size(vectors) == vector_bytes && graph_bytes <= budget &&
          std::filesystem::file_size(graph) == graph_bytes &&
          hash_file(vectors) == vector_hash && hash_file(graph) == graph_hash,
          "Router artifact size/checksum mismatch");
    const auto shape = inspect_vectors(vectors, DataType::Float, dim);
    check(shape.rows == count, "Router centroid count mismatch");
    std::vector<float> centroids(static_cast<size_t>(cells));
    std::ifstream input(vectors, std::ios::binary);
    input.seekg(8);
    input.read(reinterpret_cast<char*>(centroids.data()), cells * sizeof(float));
    check(bool(input), "Truncated centroid artifact");
    validate_values(centroids, dim);
    validate_graph(graph, dim, centroids, options);
    auto router = std::unique_ptr<CentroidRouter>(new CentroidRouter(dim, centroids, options));
    // Release the validation table before loading the graph's vector copy;
    // the preflight estimate allows two copies, not three overlapping ones.
    std::vector<float>().swap(centroids);
    router->_impl->graph = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        &router->_impl->space, graph, false, count);
    router->_impl->graph->setEf(1);
    return router;
}

CentroidTraining train_centroids_hnsw(const std::string& path, const BuildConfig& config,
                                    const RouterOptions& options, const TrainerLimits& limits) {
    config.validate();
    const uint64_t workers = std::min(config.training_count(), config.build_threads);
    uint64_t workspace = CentroidRouter::estimate_build_bytes(config.dim, config.nlist, options);
    // Visited tables and candidate heaps/materialization per simultaneous
    // worker beyond the graph's fixed preflight allowance.
    const uint64_t per_worker = 2ull * config.nlist +
        64ull * std::min(config.nlist, options.insert_breadth) + 4096;
    check(workers <= (UINT64_MAX - workspace) / per_worker, "HNSW trainer workspace overflow");
    workspace += workers * per_worker;
    struct RouterAssignment final : detail::AssignmentIndex {
        explicit RouterAssignment(std::unique_ptr<CentroidRouter> graph) : router(std::move(graph)) {}
        uint32_t assign(std::span<const float> vector) const override { return router->assign(vector); }
        std::unique_ptr<CentroidRouter> router;
    };
    const detail::AssignmentFactory factory = [&](std::span<const float> centroids) {
        return std::make_unique<RouterAssignment>(CentroidRouter::build(config.dim, centroids, options));
    };
    return detail::train_centroids_impl(path, config, limits, &factory, workspace);
}
} // namespace diskann::ivf
