#include "bufann/ivf_index.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace diskann::ivf {
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<typename T> constexpr DataType type() {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
    return std::is_same_v<T, float> ? DataType::Float : (std::is_same_v<T, int8_t> ? DataType::Int8 : DataType::UInt8);
}
void add(uint64_t& bytes, uint64_t count, uint64_t stride = 1) {
    check(!stride || count <= (UINT64_MAX - bytes) / stride, "IVF index workspace overflow");
    bytes += count * stride;
}
template<typename U> U integer(std::istream& in) {
    std::string token; in >> token; U value{};
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    check(bool(in) && !token.empty() && parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size(),
          "Invalid IVF manifest integer"); return value;
}
void word(std::istream& in, const char* expected) {
    std::string value; in >> value; check(bool(in) && value == expected, "Invalid IVF manifest field");
}
uint64_t budget_bytes(const BuildConfig& config) {
    const long double bytes = static_cast<long double>(config.memory_budget_gb) * 1024 * 1024 * 1024;
    check(bytes > 0 && bytes < UINT64_MAX, "Invalid IVF index build allowance");
    return static_cast<uint64_t>(bytes);
}
void validate(const IndexManifest& m) {
    m.training.validate();
    check(m.source_rows >= m.training.base_count && m.first_id < m.heap.id_capacity &&
          uint64_t(m.first_id) + m.training.base_count <= m.heap.id_capacity &&
          m.heap.id_capacity < inplace::INVALID_NODE && m.heap.dim == m.training.dim &&
          m.heap.dim <= UINT32_MAX - 7 && m.heap.buffer_frames >= 2 &&
          m.heap.page_size >= 512 && m.heap.page_size <= 65536 && m.heap.page_size % 512 == 0 &&
          m.chunks > 0 && m.chunks <= m.training.dim, "Invalid IVF manifest shape/layout");
    const uint64_t aligned = (uint64_t(m.heap.dim) + 7) / 8 * 8;
    check(17 + aligned * element_size(m.training.data_type) <= m.heap.page_size, "IVF heap slot exceeds page size");
    uint64_t vector_bytes = 8, code_bytes = 8;
    add(vector_bytes, uint64_t(m.source_rows) * m.training.dim, element_size(m.training.data_type));
    add(code_bytes, m.source_rows, m.chunks);
    check(m.sources[0].bytes == vector_bytes && m.sources[2].bytes == code_bytes &&
          m.sources[1] == m.components[7], "IVF source/artifact framing or PQ identity mismatch");
    CentroidRouter::estimate_build_bytes(m.training.dim, m.training.nlist, m.routing);
}
uint64_t heap_allowance(const IndexManifest& m, uint32_t frames, uint64_t pages) {
    uint64_t bytes = 16ull * 1024 * 1024; // Residency directory, allocator and I/O allowance.
    add(bytes, m.heap.id_capacity, 64); // Locators + snapshot-load temporary arrays.
    add(bytes, pages, 128); // Page directory/space set and residency segments.
    add(bytes, frames, uint64_t(m.heap.page_size) + 4096);
    return bytes;
}
uint64_t list_allowance(const IndexManifest& m) {
    uint64_t bytes = 65536 + 48;
    add(bytes, uint64_t(m.training.nlist) + 1, 8);
    add(bytes, m.training.base_count, uint64_t(m.chunks) + 4);
    add(bytes, (uint64_t(m.training.base_count) + 7) / 8);
    return bytes;
}
uint64_t pq_allowance(const IndexManifest& m) {
    uint64_t bytes = 65536;
    // Existing ADC-only table plus transposed copy, mean/permutation/offsets.
    add(bytes, m.training.dim, 256 * 8 + 8); add(bytes, uint64_t(m.chunks) + 1, 4);
    return bytes;
}
IndexManifest read_manifest(const std::string& prefix) {
    const auto path = prefix + ".manifest";
    check(std::filesystem::file_size(path) <= 16384, "IVF manifest exceeds framing limit");
    std::ifstream in(path, std::ios::binary); in.imbue(std::locale::classic());
    word(in, "IVFINDEX1"); word(in, "l2"); word(in, "global_nonresidual_pq8_external_unverified");
    word(in, "hnswlib-0.8.0-nomv");
    IndexManifest m;
    std::string dtype; in >> dtype; m.training.data_type = parse_data_type(dtype);
    m.training.dim = integer<uint32_t>(in); m.training.nlist = integer<uint32_t>(in);
    m.chunks = integer<uint32_t>(in); m.source_rows = integer<uint32_t>(in);
    m.training.base_count = integer<uint32_t>(in); m.first_id = integer<uint32_t>(in);
    m.heap.dim = m.training.dim; m.heap.id_capacity = integer<uint32_t>(in);
    m.heap.page_size = integer<uint32_t>(in); m.heap.buffer_frames = integer<uint32_t>(in);
    word(in, "training");
    in >> m.training.sample_rate; m.training.max_reps = integer<uint32_t>(in);
    m.training.seed = integer<uint64_t>(in); m.training.build_threads = integer<uint32_t>(in);
    in >> m.training.memory_budget_gb; check(bool(in), "Invalid IVF training metadata");
    word(in, "routing");
    m.routing.degree = integer<uint32_t>(in); m.routing.construction_breadth = integer<uint32_t>(in);
    m.routing.query_breadth = integer<uint32_t>(in); m.routing.insert_breadth = integer<uint32_t>(in);
    m.routing.seed = integer<uint32_t>(in); m.routing.memory_budget_bytes = integer<uint64_t>(in);
    for (auto& digest : m.sources) { word(in, "source"); digest = {integer<uint64_t>(in), integer<uint64_t>(in)}; }
    for (size_t i = 0; i < m.components.size(); ++i) {
        word(in, index_component_suffixes[i]); m.components[i] = {integer<uint64_t>(in), integer<uint64_t>(in)};
    }
    in >> std::ws; check(in.eof(), "Trailing IVF manifest data"); validate(m); return m;
}
void write_manifest(const std::string& prefix, const IndexManifest& m) {
    validate(m);
    std::ofstream out(prefix + ".manifest", std::ios::binary); out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "IVFINDEX1 l2 global_nonresidual_pq8_external_unverified hnswlib-0.8.0-nomv\n"
        << data_type_name(m.training.data_type) << ' ' << m.training.dim << ' ' << m.training.nlist << ' '
        << m.chunks << ' ' << m.source_rows << ' ' << m.training.base_count << ' ' << m.first_id << ' '
        << m.heap.id_capacity << ' ' << m.heap.page_size << ' ' << m.heap.buffer_frames << '\n'
        << "training " << m.training.sample_rate << ' ' << m.training.max_reps << ' ' << m.training.seed << ' '
        << m.training.build_threads << ' ' << m.training.memory_budget_gb << '\n'
        << "routing " << m.routing.degree << ' ' << m.routing.construction_breadth << ' '
        << m.routing.query_breadth << ' ' << m.routing.insert_breadth << ' ' << m.routing.seed << ' '
        << m.routing.memory_budget_bytes << '\n';
    for (auto digest : m.sources) out << "source " << digest.bytes << ' ' << digest.hash << '\n';
    for (size_t i = 0; i < m.components.size(); ++i)
        out << index_component_suffixes[i] << ' ' << m.components[i].bytes << ' ' << m.components[i].hash << '\n';
    out.close(); check(bool(out), "IVF manifest write failed");
}
void copy_pivots(const std::string& from, const std::string& to) {
    claim_artifact(to);
    std::ifstream input(from, std::ios::binary); std::ofstream output(to, std::ios::binary);
    check(bool(input) && bool(output), "Cannot copy IVF PQ pivots");
    char buffer[8192];
    while (input.read(buffer, sizeof(buffer)) || input.gcount()) output.write(buffer, input.gcount());
    check(input.eof() && !input.bad(), "IVF PQ pivot copy read failed");
    output.close(); check(bool(output), "IVF PQ pivot copy write failed");
}
bool same_routing(const RouterOptions& a, const RouterOptions& b) {
    return a.degree == b.degree && a.construction_breadth == b.construction_breadth &&
        a.query_breadth == b.query_breadth && a.insert_breadth == b.insert_breadth && a.seed == b.seed;
}
} // namespace

template<typename T>
void NativeIndex<T>::build(const std::string& prefix, const std::string& vectors, const std::string& pivots,
                          const std::string& codes, const BuildConfig& training, const HeapOptions& heap,
                          const RouterOptions& routing, uint32_t first_id) {
    check(!prefix.empty(), "Empty IVF index prefix"); training.validate();
    check(training.data_type == type<T>(), "IVF index build dtype mismatch");
    IndexManifest m; m.training = training; m.heap = heap; m.routing = routing; m.first_id = first_id;
    m.source_rows = inspect_vectors(vectors, training.data_type, training.dim).rows;
    const auto pq = inspect_pq_pivots(pivots, training.dim);
    m.chunks = pq.chunks; inspect_pq_codes(codes, m.source_rows, m.chunks);
    m.sources = {artifact_digest(vectors), artifact_digest(pivots), artifact_digest(codes)};
    m.components[7] = m.sources[1]; validate(m);
    const auto budget = budget_bytes(training);
    // Conservative reserved-page upper bound using a lower bound on slots/page.
    const uint64_t slot_bytes = 8 + ((uint64_t(heap.dim) + 7) / 8 * 8) * sizeof(T);
    const auto slots = (heap.page_size - 16) / (slot_bytes + 1);
    check(slots > 0, "IVF heap page cannot hold a slot");
    const uint64_t pages = ((uint64_t(training.base_count) + 256 * slots - 1) / (256 * slots)) * 256;
    const auto heap_bytes = heap_allowance(m, heap.buffer_frames, pages);
    const auto router_bytes = CentroidRouter::estimate_build_bytes(training.dim, training.nlist, routing);
    uint64_t retained = router_bytes; add(retained, heap_bytes); add(retained, pq_allowance(m));
    // Build lists need row labels and a cursor in addition to retained storage.
    uint64_t list_build = list_allowance(m); add(list_build, training.base_count, 4);
    add(list_build, uint64_t(training.nlist) + 1, 8); add(list_build, training.nlist, 66);
    add(list_build, training.dim, sizeof(T) + sizeof(float)); add(list_build, m.chunks);
    uint64_t peak = retained; add(peak, list_build);
    uint64_t router_peak = router_bytes; add(router_peak, uint64_t(training.dim) * training.nlist, sizeof(float));
    check(peak <= budget && router_peak <= budget, "IVF composed build exceeds workspace allowance");
    // Refuse every existing artifact before claiming anything. Keep partial
    // owned files on failure; an empty/partial manifest is not a loadable index.
    check(!std::filesystem::exists(std::filesystem::symlink_status(prefix + ".manifest")), "IVF index manifest exists");
    for (auto suffix : index_component_suffixes)
        check(!std::filesystem::exists(std::filesystem::symlink_status(prefix + suffix)), "IVF index component exists");
    claim_artifact(prefix + ".manifest"); // Reserves the prefix against another builder.
    std::vector<float> centroids;
    {
        auto trained = train_centroids_hnsw(vectors, training, routing);
        centroids = std::move(trained.centroids);
    }
    auto router = CentroidRouter::build(training.dim, centroids, routing);
    std::vector<float>().swap(centroids);
    router->save(prefix);
    ListBuildOptions list_options{training.base_count, first_id, heap.id_capacity, m.chunks, budget - retained};
    auto lists = PostingLists::build<T>(vectors, codes, *router, list_options);
    lists.save(prefix + ".lists");
    auto storage = HeapStorage<T>::create(heap, prefix + ".heap");
    storage->load_vectors(vectors, first_id, training.base_count);
    storage->checkpoint(prefix + ".heap.meta");
    copy_pivots(pivots, prefix + ".pq.bin");
    check(m.sources == std::array<ArtifactDigest, 3>{artifact_digest(vectors), artifact_digest(pivots), artifact_digest(codes)},
          "IVF source changed during build");
    for (size_t i = 0; i < m.components.size(); ++i) m.components[i] = artifact_digest(prefix + index_component_suffixes[i]);
    write_manifest(prefix, m); // Last successful operation; not an atomic/fsync commit.
}

template<typename T>
std::unique_ptr<NativeIndex<T>> NativeIndex<T>::open(const std::string& prefix, const IndexOpenOptions& options) {
    check(!prefix.empty(), "Empty IVF index prefix");
    auto m = read_manifest(prefix);
    check(m.training.data_type == type<T>() && options.buffer_frames >= 2, "IVF index open dtype/pool mismatch");
    const auto router_bytes = CentroidRouter::estimate_build_bytes(m.training.dim, m.training.nlist, m.routing);
    check(m.components[4].bytes % m.heap.page_size == 0, "IVF heap artifact length mismatch");
    uint64_t bytes = router_bytes;
    add(bytes, heap_allowance(m, options.buffer_frames, m.components[4].bytes / m.heap.page_size));
    add(bytes, list_allowance(m)); add(bytes, pq_allowance(m));
    check(bytes <= options.memory_budget_bytes && bytes <= SIZE_MAX, "IVF composed open exceeds workspace allowance");
    for (size_t i = 0; i < m.components.size(); ++i)
        check(artifact_digest(prefix + index_component_suffixes[i]) == m.components[i], "IVF index component integrity mismatch");
    auto result = std::unique_ptr<NativeIndex>(new NativeIndex);
    result->_manifest = m; result->_estimated_open_bytes = bytes;
    result->_router = CentroidRouter::load(prefix, router_bytes);
    check(result->_router->dim() == m.training.dim && result->_router->nlist() == m.training.nlist &&
          same_routing(result->_router->options(), m.routing), "IVF manifest/router mismatch");
    result->_lists = std::make_unique<PostingLists>(PostingLists::load(prefix + ".lists", list_allowance(m)));
    const auto& lists = *result->_lists;
    check(lists.data_type() == type<T>() && lists.dim() == m.training.dim && lists.nlist() == m.training.nlist &&
          lists.chunks() == m.chunks && lists.size() == m.training.base_count && lists.first_id() == m.first_id &&
          lists.id_capacity() == m.heap.id_capacity, "IVF manifest/lists mismatch");
    result->_pq = std::make_unique<PQCodebook<T>>(prefix + ".pq.bin", m.training.dim, m.chunks);
    check(result->_pq->chunks() == m.chunks, "IVF manifest/PQ mismatch");
    auto runtime_heap = m.heap; runtime_heap.buffer_frames = options.buffer_frames;
    result->_heap = HeapStorage<T>::reopen(runtime_heap, prefix + ".heap", prefix + ".heap.meta");
    check(result->_heap->size() == lists.size(), "IVF heap/list population mismatch");
    for (uint32_t c = 0; c < lists.nlist(); ++c) for (auto id : lists.ids(c))
        check(result->_heap->contains(id), "IVF list references missing heap ID");
    return result;
}

template<typename T>
uint64_t NativeIndex<T>::estimated_query_bytes(const SearchConfig& config, bool exact_routing) const {
    return estimate_query_workspace(*_router, *_lists, *_pq, *_heap, config, exact_routing);
}
template<typename T>
QueryResult NativeIndex<T>::query(std::span<const T> vector, const SearchConfig& config, const QueryLimits& limits) const {
    return search(*_router, *_lists, *_pq, *_heap, vector, config, limits);
}
template class NativeIndex<float>;
template class NativeIndex<int8_t>;
template class NativeIndex<uint8_t>;
} // namespace diskann::ivf
