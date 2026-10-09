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
    check(m.format_version >= 1 && m.format_version <= 3 &&
          (m.format_version == 3 || m.routing.native_modes == (m.format_version == 2)), "Invalid IVF manifest family");
    check(m.format_version == 3 || (m.training.trainer_mode == TrainerMode::AnnRebuild &&
          m.training.init == CentroidInit::Sample && m.training.training_breadth == m.routing.insert_breadth),
          "Legacy manifest cannot encode new trainer settings");
    check(m.router_backend == "hnswlib-0.8.0-prefetch1" || (!m.routing.native_modes && m.router_backend == "hnswlib-0.8.0-nomv"), "Invalid IVF router producer identity");
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
IndexManifest read_manifest_file(const std::string& path) {
    check(std::filesystem::file_size(path) <= 16384, "IVF manifest exceeds framing limit");
    std::ifstream in(path, std::ios::binary); in.imbue(std::locale::classic());
    std::string version; in >> version;
    check(version == "IVFINDEX1" || version == "IVFINDEX2" || version == "IVFINDEX3", "Unsupported IVF index version");
    word(in, "l2"); word(in, "global_nonresidual_pq8_external_unverified");
    std::string backend; in >> backend;
    check(backend == "hnswlib-0.8.0-prefetch1" || (version == "IVFINDEX1" && backend == "hnswlib-0.8.0-nomv"), "IVF backend mismatch");
    IndexManifest m;
    m.format_version = uint32_t(version.back()-'0');
    m.router_backend = backend;
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
    if (m.format_version == 3) {
        word(in, "trainer"); std::string mode, init; in >> mode >> init;
        m.training.trainer_mode = parse_trainer_mode(mode); m.training.init = parse_centroid_init(init);
        m.training.training_breadth = integer<uint32_t>(in);
    }
    word(in, "routing");
    m.routing.degree = integer<uint32_t>(in); m.routing.construction_breadth = integer<uint32_t>(in);
    m.routing.query_breadth = integer<uint32_t>(in); m.routing.insert_breadth = integer<uint32_t>(in);
    m.routing.seed = integer<uint32_t>(in); m.routing.memory_budget_bytes = integer<uint64_t>(in);
    if (m.format_version < 3) m.training.training_breadth = m.routing.insert_breadth;
    bool native = version == "IVFINDEX2";
    if (m.format_version == 3) {
        word(in, "optimized_modes"); auto value = integer<uint32_t>(in);
        check(value <= 1, "Invalid optimized modes flag"); native = value != 0;
    }
    if (native) {
        word(in, "native_routing"); std::string mode; in >> mode; m.routing.mode = parse_router_mode(mode); m.routing.native_modes = true;
        m.routing.refine_breadth = integer<uint32_t>(in); const auto reverse = integer<uint32_t>(in);
        check(reverse <= 1, "Invalid native reverse-edge flag"); m.routing.refine_reverse_edges = reverse != 0;
        in >> m.routing.pca_variance >> m.routing.router_rerank_factor; check(bool(in), "Invalid native PCA/rerank metadata");
        m.routing.native_work_limit = integer<uint64_t>(in);
        m.pca_retained_dim = integer<uint32_t>(in);
        check(m.pca_retained_dim > 0 && m.pca_retained_dim <= m.training.dim, "Invalid manifest PCA retained dimension");
    }
    for (auto& digest : m.sources) { word(in, "source"); digest = {integer<uint64_t>(in), integer<uint64_t>(in)}; }
    for (size_t i = 0; i < m.components.size(); ++i) {
        word(in, index_component_suffixes[i]); m.components[i] = {integer<uint64_t>(in), integer<uint64_t>(in)};
    }
    if (m.routing.native_modes) for (size_t i = 0; i < m.native_components.size(); ++i) {
        word(in, native_router_suffixes[i]); m.native_components[i] = {integer<uint64_t>(in), integer<uint64_t>(in)};
    }
    in >> std::ws; check(in.eof(), "Trailing IVF manifest data"); validate(m); return m;
}
IndexManifest read_manifest(const std::string& prefix) { return read_manifest_file(prefix + ".manifest"); }
void write_manifest_file(const std::string& path, const IndexManifest& m) {
    check(!m.routing.native_modes || (m.pca_retained_dim > 0 && m.pca_retained_dim <= m.training.dim), "Missing fitted PCA manifest dimension");
    validate(m);
    std::ofstream out(path, std::ios::binary); out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "IVFINDEX" << m.format_version << " l2 global_nonresidual_pq8_external_unverified " << m.router_backend << '\n'
        << data_type_name(m.training.data_type) << ' ' << m.training.dim << ' ' << m.training.nlist << ' '
        << m.chunks << ' ' << m.source_rows << ' ' << m.training.base_count << ' ' << m.first_id << ' '
        << m.heap.id_capacity << ' ' << m.heap.page_size << ' ' << m.heap.buffer_frames << '\n'
        << "training " << m.training.sample_rate << ' ' << m.training.max_reps << ' ' << m.training.seed << ' '
        << m.training.build_threads << ' ' << m.training.memory_budget_gb << '\n';
    if (m.format_version == 3) out << "trainer " << trainer_mode_name(m.training.trainer_mode) << ' '
        << centroid_init_name(m.training.init) << ' ' << m.training.training_breadth << '\n';
    out << "routing " << m.routing.degree << ' ' << m.routing.construction_breadth << ' '
        << m.routing.query_breadth << ' ' << m.routing.insert_breadth << ' ' << m.routing.seed << ' '
        << m.routing.memory_budget_bytes << '\n';
    if (m.format_version == 3) out << "optimized_modes " << m.routing.native_modes << '\n';
    if (m.routing.native_modes) out << "native_routing " << router_mode_name(m.routing.mode) << ' ' << m.routing.refine_breadth << ' '
        << m.routing.refine_reverse_edges << ' ' << m.routing.pca_variance << ' ' << m.routing.router_rerank_factor << ' '
        << m.routing.native_work_limit << ' ' << m.pca_retained_dim << '\n';
    for (auto digest : m.sources) out << "source " << digest.bytes << ' ' << digest.hash << '\n';
    for (size_t i = 0; i < m.components.size(); ++i)
        out << index_component_suffixes[i] << ' ' << m.components[i].bytes << ' ' << m.components[i].hash << '\n';
    if (m.routing.native_modes) for (size_t i = 0; i < m.native_components.size(); ++i)
        out << native_router_suffixes[i] << ' ' << m.native_components[i].bytes << ' ' << m.native_components[i].hash << '\n';
    out.close(); check(bool(out), "IVF manifest write failed");
}
void write_manifest(const std::string& prefix, const IndexManifest& m) { write_manifest_file(prefix + ".manifest", m); }
void copy_component(const std::string& from, const std::string& to) {
    claim_artifact(to);
    std::ifstream input(from, std::ios::binary); std::ofstream output(to, std::ios::binary);
    check(bool(input) && bool(output), "Cannot copy IVF component");
    char buffer[8192];
    while (input.read(buffer, sizeof(buffer)) || input.gcount()) output.write(buffer, input.gcount());
    check(input.eof() && !input.bad(), "IVF component copy read failed");
    output.close(); check(bool(output), "IVF component copy write failed");
}
bool same_routing(const RouterOptions& a, const RouterOptions& b) {
    return a.degree == b.degree && a.construction_breadth == b.construction_breadth &&
        a.query_breadth == b.query_breadth && a.insert_breadth == b.insert_breadth && a.seed == b.seed &&
        a.native_modes == b.native_modes && a.mode == b.mode && a.refine_breadth == b.refine_breadth &&
        a.refine_reverse_edges == b.refine_reverse_edges && a.pca_variance == b.pca_variance &&
        a.router_rerank_factor == b.router_rerank_factor && a.native_work_limit == b.native_work_limit;
}
void require_new_artifact(const std::string& path) {
    check(!std::filesystem::exists(std::filesystem::symlink_status(path)), "Mutable IVF output artifact exists");
}
uint64_t mutable_other_allowance(const IndexManifest& m, uint32_t frames, uint64_t heap_file_bytes) {
    const uint64_t slot_bytes = 8 + ((uint64_t(m.heap.dim) + 7) / 8 * 8) * element_size(m.training.data_type);
    const auto slots = (m.heap.page_size - 16) / (slot_bytes + 1);
    check(slots > 0 && heap_file_bytes % m.heap.page_size == 0, "Invalid mutable IVF heap layout/file size");
    const auto pages = ((uint64_t(m.heap.id_capacity) + 256 * slots - 1) / (256 * slots)) * 256;
    uint64_t bytes = heap_allowance(m, frames, std::max(pages, heap_file_bytes / m.heap.page_size));
    add(bytes, m.heap.id_capacity); add(bytes, pq_allowance(m));
    add(bytes, CentroidRouter::estimate_build_bytes(m.training.dim, m.training.nlist, m.routing)); add(bytes, 65536);
    return bytes;
}
struct MutableManifest {
    uint32_t live = 0, retained = 0;
    std::array<ArtifactDigest, 10> components{};
    bool native = false;
    std::array<ArtifactDigest, 3> native_components{};
};
MutableManifest read_mutable_manifest(const std::string& prefix) {
    check(std::filesystem::file_size(prefix + ".manifest") <= 4096, "Mutable IVF manifest exceeds framing limit");
    std::ifstream in(prefix + ".manifest", std::ios::binary); in.imbue(std::locale::classic());
    std::string version; in >> version; check(version == "IVFMUTABLE1" || version == "IVFMUTABLE2", "Unsupported mutable IVF version");
    MutableManifest m; m.native = version == "IVFMUTABLE2";
    m.live = integer<uint32_t>(in); m.retained = integer<uint32_t>(in);
    for (size_t i = 0; i < m.components.size(); ++i) {
        word(in, mutable_component_suffixes[i]); m.components[i] = {integer<uint64_t>(in), integer<uint64_t>(in)};
    }
    if (m.native) for (size_t i = 0; i < m.native_components.size(); ++i) {
        word(in, native_router_suffixes[i]); m.native_components[i] = {integer<uint64_t>(in), integer<uint64_t>(in)};
    }
    in >> std::ws; check(in.eof(), "Trailing mutable IVF manifest data"); return m;
}
void validate_frozen_components(const MutableManifest& current, const IndexManifest& origin) {
    check(current.native == origin.routing.native_modes && (!current.native || current.native_components == origin.native_components),
          "Mutable IVF checkpoint changed native router/PCA identity");
    for (size_t i : {0u, 1u, 2u, 7u})
        check(current.components[i] == origin.components[i], "Mutable IVF checkpoint changed frozen router/PQ identity");
    check(current.live <= current.retained && current.retained <= origin.heap.id_capacity - origin.first_id,
          "Invalid mutable IVF checkpoint population");
}
} // namespace

template<typename T>
void NativeIndex<T>::build(const std::string& prefix, const std::string& vectors, const std::string& pivots,
                          const std::string& codes, const BuildConfig& training, const HeapOptions& heap,
                          const RouterOptions& routing, uint32_t first_id, TrainingStats* training_stats) {
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
    if (routing.native_modes) for (auto suffix : native_router_suffixes) require_new_artifact(prefix + suffix);
    claim_artifact(prefix + ".manifest"); // Reserves the prefix against another builder.
    std::vector<float> centroids;
    {
        auto trained = train_centroids_ann(vectors, training, routing);
        if (training_stats) *training_stats = std::move(static_cast<TrainingStats&>(trained));
        centroids = std::move(trained.centroids);
    }
    auto router = CentroidRouter::build(training.dim, centroids, routing);
    if (routing.native_modes) m.pca_retained_dim = router->projected_dim();
    std::vector<float>().swap(centroids);
    router->save(prefix);
    ListBuildOptions list_options{training.base_count, first_id, heap.id_capacity, m.chunks, budget - retained};
    auto lists = PostingLists::build<T>(vectors, codes, *router, list_options);
    lists.save(prefix + ".lists");
    auto storage = HeapStorage<T>::create(heap, prefix + ".heap");
    storage->load_vectors(vectors, first_id, training.base_count);
    storage->checkpoint(prefix + ".heap.meta");
    copy_component(pivots, prefix + ".pq.bin");
    check(m.sources == std::array<ArtifactDigest, 3>{artifact_digest(vectors), artifact_digest(pivots), artifact_digest(codes)},
          "IVF source changed during build");
    for (size_t i = 0; i < m.components.size(); ++i) m.components[i] = artifact_digest(prefix + index_component_suffixes[i]);
    if (routing.native_modes) for (size_t i = 0; i < m.native_components.size(); ++i) m.native_components[i] = artifact_digest(prefix + native_router_suffixes[i]);
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
    if (m.routing.native_modes) for (size_t i = 0; i < m.native_components.size(); ++i)
        check(artifact_digest(prefix + native_router_suffixes[i]) == m.native_components[i], "IVF native component integrity mismatch");
    auto result = std::unique_ptr<NativeIndex>(new NativeIndex);
    result->_manifest = m; result->_estimated_open_bytes = bytes;
    result->_router = CentroidRouter::load(prefix, router_bytes);
    check(result->_router->dim() == m.training.dim && result->_router->nlist() == m.training.nlist &&
          same_routing(result->_router->options(), m.routing) && (!m.routing.native_modes || result->_router->projected_dim() == m.pca_retained_dim), "IVF manifest/router mismatch");
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

template<typename T>
std::unique_ptr<MutableIndex<T>> MutableIndex<T>::fork(const std::string& source,
    const std::string& heap_path, const MutableIndexOptions& options) {
    check(!heap_path.empty(), "Empty mutable IVF private heap path");
    require_new_artifact(heap_path); require_new_artifact(heap_path + ".pq.bin");
    auto original = NativeIndex<T>::open(source, {options.buffer_frames, options.memory_budget_bytes});
    const auto& m = original->manifest();
    MutableListOptions list_options{m.training.nlist, m.chunks, m.heap.id_capacity, m.first_id};
    uint64_t initial_lists = MutablePostingLists::estimate_empty_bytes(list_options);
    add(initial_lists, m.training.base_count, uint64_t(m.chunks) + 4);
    const uint64_t slot_bytes = 8 + ((uint64_t(m.heap.dim) + 7) / 8 * 8) * sizeof(T);
    const auto slots = (m.heap.page_size - 16) / (slot_bytes + 1);
    check(slots > 0, "Mutable IVF heap page cannot hold a slot");
    const uint64_t pages = ((uint64_t(m.heap.id_capacity) + 256 * slots - 1) / (256 * slots)) * 256;
    uint64_t future_heap = heap_allowance(m, options.buffer_frames,
        std::max(pages, m.components[4].bytes / m.heap.page_size));
    add(future_heap, m.heap.id_capacity); // IVF logical-state ledger (not backend active flags).
    uint64_t peak = original->estimated_open_bytes();
    add(peak, initial_lists); add(peak, future_heap); add(peak, 65536);
    check(peak <= options.memory_budget_bytes && peak <= SIZE_MAX, "Mutable IVF fork exceeds composed workspace allowance");
    uint64_t other = future_heap;
    add(other, pq_allowance(m)); add(other, CentroidRouter::estimate_build_bytes(m.training.dim, m.training.nlist, m.routing));
    add(other, 65536);
    check(other < options.memory_budget_bytes, "Mutable IVF retained components exceed workspace allowance");
    const uint64_t list_budget = options.list_memory_budget_bytes ? options.list_memory_budget_bytes : options.memory_budget_bytes - other;
    check(list_budget >= initial_lists && list_budget <= options.memory_budget_bytes - other,
          "Mutable IVF list allocation budget incompatible with composed allowance");
    auto result = std::unique_ptr<MutableIndex>(new MutableIndex);
    result->_origin = m; result->_fork_bytes = peak; result->_other_bytes = other;
    result->_lists = MutablePostingLists::from_immutable(*original->_lists, list_budget);
    result->_router = std::move(original->_router); result->_pq = std::move(original->_pq);
    original->_lists.reset(); original->_heap.reset(); // No permanent duplicate CSR/heap metadata.
    copy_component(source + ".heap", heap_path); // Exclusive claim, real bytes, no writable source symlink.
    result->_pq_artifact_path = heap_path + ".pq.bin";
    copy_component(source + ".pq.bin", result->_pq_artifact_path);
    check(artifact_digest(result->_pq_artifact_path) == m.components[7], "Mutable IVF private PQ copy integrity mismatch");
    check(artifact_digest(heap_path) == m.components[4], "Mutable IVF heap copy integrity mismatch");
    auto runtime_heap = m.heap; runtime_heap.buffer_frames = options.buffer_frames;
    result->_heap = HeapStorage<T>::reopen_mutable_copy(runtime_heap, heap_path, source + ".heap.meta");
    check(result->_heap->size() == result->_lists->size(), "Mutable IVF fork heap/list population mismatch");
    for (size_t i = 0; i < m.components.size(); ++i)
        check(artifact_digest(source + index_component_suffixes[i]) == m.components[i], "Immutable IVF source changed during fork");
    if (m.routing.native_modes) for (size_t i = 0; i < m.native_components.size(); ++i)
        check(artifact_digest(source + native_router_suffixes[i]) == m.native_components[i], "Native IVF source changed during fork");
    return result;
}
template<typename T>
void MutableIndex<T>::require_healthy() const { check(!failed(), "Mutable IVF session failed; discard owned heap rather than checkpointing"); }
template<typename T>
uint64_t MutableIndex<T>::estimated_mutation_bytes(bool exact) const {
    uint64_t bytes = 65536;
    add(bytes, _router->nlist(), 2);
    if (_router->has_native_modes()) { add(bytes, _router->nlist(), 128); add(bytes, _router->dim(), sizeof(float)); }
    add(bytes, exact ? _router->nlist() : std::min(_router->nlist(), _router->options().insert_breadth), 64);
    add(bytes, _router->dim(), sizeof(T) + sizeof(float));
    add(bytes, _heap->aligned_dim(), sizeof(T)); add(bytes, _pq->chunks());
    check(bytes <= SIZE_MAX, "Mutable IVF insertion workspace overflow"); return bytes;
}
template<typename T>
void MutableIndex<T>::insert(uint32_t id, std::span<const T> coordinates, const MutationLimits& limits) {
    std::shared_lock admission(_lifecycle);
    require_healthy();
    check(coordinates.size() == _router->dim(), "Mutable IVF insert dimension mismatch");
    for (T value : coordinates) check(std::isfinite(double(value)), "Nonfinite mutable IVF insert coordinate");
    check(estimated_mutation_bytes(limits.exact_routing) <= limits.workspace_bytes, "Mutable IVF insert exceeds workspace allowance");
    auto reservation = _lists->reserve_insert(id);
    bool prepared = false;
    try {
        const auto list = _router->assign(coordinates, limits.exact_routing);
        std::vector<uint8_t> codes(_pq->chunks(), 0); _pq->encode(coordinates.data(), codes.data());
        _heap->prepare_insert(id, coordinates); prepared = true;
        reservation.commit_ready(list, codes, [&] { _heap->publish_insert(id); });
        prepared = false;
    } catch (...) {
        if (prepared) {
            try { _heap->cancel_insert(id); }
            catch (...) { _failed = true; }
        }
        if (_heap->failed()) _failed = true;
        throw;
    }
}
template<typename T>
bool MutableIndex<T>::erase(uint32_t id) {
    std::shared_lock admission(_lifecycle);
    require_healthy();
    if (!_lists->erase(id)) return false;
    try { check(_heap->retire(id), "Mutable IVF deleted list ID lacks a live heap record"); }
    catch (...) { _failed = true; throw; }
    return true;
}
template<typename T>
uint64_t MutableIndex<T>::estimated_query_bytes(const SearchConfig& config, bool exact) const {
    require_healthy(); return estimate_mutable_query_workspace(*_router, *_lists, *_pq, *_heap, config, exact);
}
template<typename T>
QueryResult MutableIndex<T>::query(std::span<const T> vector, const SearchConfig& config, const QueryLimits& limits) const {
    std::shared_lock admission(_lifecycle);
    require_healthy(); return search_mutable(*_router, *_lists, *_pq, *_heap, vector, config, limits);
}
template<typename T>
QueryResult MutableIndex<T>::rerank_candidates(std::span<const T> vector, std::span<const PQCandidate> candidates,
    const SearchConfig& config, const QueryLimits& limits) const {
    std::shared_lock admission(_lifecycle);
    require_healthy(); return rerank_mutable_candidates(*_router, *_lists, *_pq, *_heap, vector, candidates, config, limits);
}

template<typename T>
void MutableIndex<T>::validate_checkpoint_membership() const {
    check(_lists->pending_insertions() == 0 && _lists->size() == _heap->size(), "Mutable IVF checkpoint is not quiescent/consistent");
    for (uint32_t id = 0; id < _heap->id_capacity(); ++id) {
        const auto state = _heap->_mutable_states[id];
        if (id < _lists->first_id()) {
            check(state == HeapStorage<T>::MutableState::Absent, "Mutable IVF heap record below allowed ID range"); continue;
        }
        check(_lists->contains(id) == (state == HeapStorage<T>::MutableState::Live) &&
              _lists->deleted(id) == (state == HeapStorage<T>::MutableState::Deleted) &&
              state != HeapStorage<T>::MutableState::Prepared, "Mutable IVF heap/list live/tombstone mismatch");
    }
}
template<typename T>
uint64_t MutableIndex<T>::estimated_checkpoint_bytes() const {
    uint64_t bytes = 65536;
    add(bytes, _heap->id_capacity(), 32); add(bytes, _heap->retained_file_bytes() / _origin.heap.page_size, 16);
    add(bytes, _origin.training.nlist, 16); add(bytes, _origin.training.dim, sizeof(float));
    if (_router->has_native_modes()) add(bytes, _router->nlist(), 32); // Native graph validation BFS/indegrees.
    return bytes;
}
template<typename T>
void MutableIndex<T>::checkpoint(const std::string& prefix, uint64_t workspace_bytes) {
    std::unique_lock admission(_lifecycle); require_healthy();
    check(!prefix.empty() && estimated_checkpoint_bytes() <= workspace_bytes, "Mutable IVF checkpoint exceeds workspace allowance/empty prefix");
    validate_checkpoint_membership();
    require_new_artifact(prefix + ".manifest");
    for (auto suffix : mutable_component_suffixes) require_new_artifact(prefix + suffix);
    if (_origin.routing.native_modes) for (auto suffix : native_router_suffixes) require_new_artifact(prefix + suffix);
    check(artifact_digest(_pq_artifact_path) == _origin.components[7], "Mutable IVF private PQ sidecar changed");
    claim_artifact(prefix + ".manifest"); // Exclusive prefix reservation; empty on partial failure.
    _router->save(prefix);
    _lists->save_checkpoint(prefix + ".mlists");
    _heap->checkpoint_mutable_copy(prefix + ".heap", prefix + ".heap.meta", prefix + ".heap.state");
    copy_component(_pq_artifact_path, prefix + ".pq.bin");
    claim_artifact(prefix + ".origin"); write_manifest_file(prefix + ".origin", _origin);
    MutableManifest current; current.live = _lists->size(); current.retained = _heap->retained_size(); current.native = _origin.routing.native_modes;
    for (size_t i = 0; i < current.components.size(); ++i) current.components[i] = artifact_digest(prefix + mutable_component_suffixes[i]);
    if (current.native) for (size_t i = 0; i < current.native_components.size(); ++i) current.native_components[i] = artifact_digest(prefix + native_router_suffixes[i]);
    validate_frozen_components(current, _origin);
    std::ofstream out(prefix + ".manifest", std::ios::binary); out.imbue(std::locale::classic());
    out << (current.native ? "IVFMUTABLE2" : "IVFMUTABLE1") << ' ' << current.live << ' ' << current.retained << '\n';
    for (size_t i = 0; i < current.components.size(); ++i)
        out << mutable_component_suffixes[i] << ' ' << current.components[i].bytes << ' ' << current.components[i].hash << '\n';
    if (current.native) for (size_t i = 0; i < current.native_components.size(); ++i)
        out << native_router_suffixes[i] << ' ' << current.native_components[i].bytes << ' ' << current.native_components[i].hash << '\n';
    out.close(); check(bool(out), "Mutable IVF checkpoint manifest write failure");
}
template<typename T>
std::unique_ptr<MutableIndex<T>> MutableIndex<T>::reopen(const std::string& prefix, const std::string& heap_path,
    const MutableIndexOptions& options) {
    check(!prefix.empty() && !heap_path.empty() && options.buffer_frames >= 2, "Invalid mutable IVF reopen options");
    require_new_artifact(heap_path); require_new_artifact(heap_path + ".pq.bin");
    const auto current = read_mutable_manifest(prefix);
    for (size_t i = 0; i < current.components.size(); ++i)
        check(artifact_digest(prefix + mutable_component_suffixes[i]) == current.components[i], "Mutable IVF checkpoint integrity mismatch");
    if (current.native) for (size_t i = 0; i < current.native_components.size(); ++i)
        check(artifact_digest(prefix + native_router_suffixes[i]) == current.native_components[i], "Mutable native component integrity mismatch");
    auto origin = read_manifest_file(prefix + ".origin"); validate_frozen_components(current, origin);
    check(origin.training.data_type == type<T>(), "Mutable IVF checkpoint dtype mismatch");
    const auto other = mutable_other_allowance(origin, options.buffer_frames, current.components[4].bytes);
    check(other < options.memory_budget_bytes, "Mutable IVF reopen components exceed memory allowance");
    MutableListOptions list_options{origin.training.nlist, origin.chunks, origin.heap.id_capacity, origin.first_id,
        options.list_memory_budget_bytes ? options.list_memory_budget_bytes : options.memory_budget_bytes - other};
    check(list_options.memory_budget_bytes <= options.memory_budget_bytes - other, "Mutable IVF reopen list budget exceeds composed allowance");
    const auto list_info = MutablePostingLists::inspect_checkpoint(prefix + ".mlists", list_options);
    check(list_info.live == current.live && list_info.estimated_bytes <= list_options.memory_budget_bytes,
          "Mutable IVF reopen list population/budget mismatch");
    auto result = std::unique_ptr<MutableIndex>(new MutableIndex);
    result->_origin = origin; result->_other_bytes = other; result->_fork_bytes = other + list_info.estimated_bytes;
    result->_lists = MutablePostingLists::load_checkpoint(prefix + ".mlists", list_options);
    const auto router_bytes = CentroidRouter::estimate_build_bytes(origin.training.dim, origin.training.nlist, origin.routing);
    result->_router = CentroidRouter::load(prefix, router_bytes);
    check(result->_router->dim() == origin.training.dim && result->_router->nlist() == origin.training.nlist &&
          same_routing(result->_router->options(), origin.routing) && (!origin.routing.native_modes || result->_router->projected_dim() == origin.pca_retained_dim), "Mutable IVF checkpoint router mismatch");
    result->_pq = std::make_unique<PQCodebook<T>>(prefix + ".pq.bin", origin.training.dim, origin.chunks);
    copy_component(prefix + ".heap", heap_path);
    result->_pq_artifact_path = heap_path + ".pq.bin"; copy_component(prefix + ".pq.bin", result->_pq_artifact_path);
    check(artifact_digest(heap_path) == current.components[4] && artifact_digest(result->_pq_artifact_path) == current.components[7],
          "Mutable IVF reopen private copy integrity mismatch");
    auto runtime_heap = origin.heap; runtime_heap.buffer_frames = options.buffer_frames;
    result->_heap = HeapStorage<T>::reopen_mutable_checkpoint_copy(runtime_heap, heap_path, prefix + ".heap.meta", prefix + ".heap.state");
    check(result->_heap->size() == current.live && result->_heap->retained_size() == current.retained, "Mutable IVF checkpoint heap population mismatch");
    result->validate_checkpoint_membership();
    for (size_t i = 0; i < current.components.size(); ++i)
        check(artifact_digest(prefix + mutable_component_suffixes[i]) == current.components[i], "Mutable IVF checkpoint changed during reopen");
    if (current.native) for (size_t i = 0; i < current.native_components.size(); ++i)
        check(artifact_digest(prefix + native_router_suffixes[i]) == current.native_components[i], "Mutable native checkpoint changed during reopen");
    return result;
}
template class MutableIndex<float>;
template class MutableIndex<int8_t>;
template class MutableIndex<uint8_t>;
} // namespace diskann::ivf
