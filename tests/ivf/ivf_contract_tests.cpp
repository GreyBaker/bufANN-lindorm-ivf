#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "bufann/ivf_config.h"
#include "bufann/ivf_io.h"
#include "bufann/ivf_pq.h"

namespace {
using namespace diskann::ivf;
namespace fs = std::filesystem;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()>& operation, const char* message) {
    try { operation(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

template <typename T>
void write_value(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void write_bin(const fs::path& path, uint32_t rows, uint32_t cols, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    write_value(out, rows);
    write_value(out, cols);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    check(static_cast<bool>(out), "fixture write failed");
}

template <typename T>
void patch_value(const fs::path& path, uint64_t offset, T value) {
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    out.seekp(static_cast<std::streamoff>(offset));
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    check(static_cast<bool>(out), "fixture patch failed");
}

constexpr uint32_t kDim = 5;
constexpr uint32_t kChunks = 2;
const std::array<uint32_t, kDim> kPermutation{4, 0, 3, 1, 2};
const std::array<uint32_t, kChunks + 1> kOffsets{0, 2, 5};
const std::array<float, kDim> kMean{10, 20, -5, 30, 2};

float center(uint32_t id, uint32_t dim) { return id * 0.25f + dim * 0.125f - 32.0f; }

std::array<uint64_t, 5> write_pivots(const fs::path& path) {
    std::array<uint64_t, 5> offsets{4096, 0, 0, 0, 0};
    offsets[1] = offsets[0] + 8 + 256 * kDim * 4;
    offsets[2] = offsets[1] + 8 + kDim * 4;
    offsets[3] = offsets[2] + 8 + kDim * 4;
    offsets[4] = offsets[3] + 8 + (kChunks + 1) * 4;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    write_value(out, uint32_t{5}); write_value(out, uint32_t{1});
    for (const auto offset : offsets) write_value(out, offset);
    out.seekp(offsets[0]);
    write_value(out, uint32_t{256}); write_value(out, kDim);
    for (uint32_t i = 0; i < 256; ++i) {
        for (uint32_t d = 0; d < kDim; ++d) write_value(out, center(i, d));
    }
    write_value(out, kDim); write_value(out, uint32_t{1});
    for (const float value : kMean) write_value(out, value);
    write_value(out, kDim); write_value(out, uint32_t{1});
    for (const auto value : kPermutation) write_value(out, value);
    write_value(out, kChunks + 1); write_value(out, uint32_t{1});
    for (const auto value : kOffsets) write_value(out, value);
    check(static_cast<bool>(out), "PQ fixture write failed");
    return offsets;
}

void test_config() {
    check(parse_data_type("uint8") == DataType::UInt8, "dtype parse");
    check(element_size(DataType::Float) == 4, "float size");
    rejects([] { parse_data_type("UInt8"); }, "unexpected dtype accepted");
    BuildConfig build;
    build.dim = 128; build.base_count = 1000; build.nlist = 100; build.sample_rate = 0.1;
    build.validate();
    check(build.training_count() == 100, "sample count");
    build.nlist = 101;
    rejects([&] { build.validate(); }, "unsupported nlist accepted");
    build.nlist = 100;
    for (const double rate : {0.0, -1.0, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
        build.sample_rate = rate;
        rejects([&] { build.validate(); }, "invalid sample rate accepted");
    }
    build.sample_rate = 0.1;
    build.memory_budget_gb = std::numeric_limits<double>::infinity();
    rejects([&] { build.validate(); }, "infinite budget accepted");
    SearchConfig search;
    search.recall_at = 3; search.rerank_factor = 1.5;
    search.validate(1);
    check(search.candidate_budget(100) == 5, "candidate ceiling");
    check(search.candidate_budget(2) == 2, "candidate live-count clamp");
    check(search.candidate_budget(0) == 0, "empty candidate budget");
    search.nprobe = 2;
    rejects([&] { search.validate(1); }, "too many probes accepted");
    search.nprobe = 1;
    for (const double factor : {0.5, std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(), 1e100}) {
        search.rerank_factor = factor;
        rejects([&] { search.validate(1); }, "invalid rerank factor accepted");
    }
}

void test_io(const fs::path& dir) {
    const auto vectors = (dir / "vectors.bin").string();
    write_bin<uint8_t>(vectors, 2, 3, {0, 255, 4, 10, 11, 12});
    check(inspect_vectors(vectors, DataType::UInt8, 3).rows == 2, "vector rows");
    rejects([&] { inspect_vectors(vectors, DataType::Float, 3); }, "wrong dtype accepted");
    rejects([&] { inspect_vectors(vectors, DataType::UInt8, 2); }, "wrong dimension accepted");
    write_bin<int8_t>(vectors, 2, 3, {-128, 127, -1, 0, 1, 2});
    check(inspect_vectors(vectors, DataType::Int8, 3).cols == 3, "int8 vectors");
    write_bin<float>(vectors, 1, 3, {1, 2, 3});
    check(inspect_vectors(vectors, DataType::Float, 3).rows == 1, "float vectors");
    write_bin<uint8_t>(vectors, 2, 3, {0});
    rejects([&] { inspect_bin(vectors, 1); }, "truncated payload accepted");
    write_bin<uint8_t>(vectors, 1, 1, {0, 1});
    rejects([&] { inspect_bin(vectors, 1); }, "trailing bytes accepted");
    write_bin<uint8_t>(vectors, UINT32_MAX, 1, {});
    rejects([&] { inspect_bin(vectors, 1); }, "negative signed header accepted");
    write_bin<uint8_t>(vectors, 1, 1, {0});
    rejects([&] { inspect_bin(vectors, std::numeric_limits<size_t>::max()); }, "size overflow accepted");
    rejects([&] { inspect_bin((dir / "absent.bin").string(), 1); }, "absent file accepted");

    const auto codes = (dir / "codes.bin").string();
    write_bin<uint8_t>(codes, 2, 2, {0, 255, 100, 4});
    inspect_pq_codes(codes, 2, 2);
    rejects([&] { inspect_pq_codes(codes, 1, 2); }, "wrong PQ code count accepted");
    rejects([&] { inspect_pq_codes(codes, 2, 3); }, "wrong PQ chunks accepted");
    const auto truth = (dir / "truth.bin").string();
    write_bin<uint32_t>(truth, 2, 2, {0, 1, 2, 3, 0, 0, 0, 0});
    inspect_truth(truth, 2, 2);
    rejects([&] { inspect_truth(truth, 3, 2); }, "wrong GT queries accepted");
    rejects([&] { inspect_truth(truth, 2, 3); }, "shallow GT accepted");

    const auto ids = (dir / "ids.bin").string();
    auto write_ids = [&](uint32_t count, const std::vector<uint32_t>& values) {
        std::ofstream out(ids, std::ios::binary | std::ios::trunc);
        write_value(out, count);
        for (const auto value : values) write_value(out, value);
    };
    write_ids(2, {4, 1});
    check(load_counted_ids(ids, 5) == std::vector<uint32_t>({4, 1}), "ID order changed");
    write_ids(0, {});
    check(load_counted_ids(ids, 5).empty(), "zero-ID stream rejected");
    write_ids(2, {1, 1});
    rejects([&] { load_counted_ids(ids, 5); }, "duplicate IDs accepted");
    write_ids(1, {5});
    rejects([&] { load_counted_ids(ids, 5); }, "out-of-range ID accepted");
    write_ids(2, {1});
    rejects([&] { load_counted_ids(ids, 5); }, "truncated IDs accepted");
}

void test_pivot_validation(const fs::path& dir) {
    const auto path = dir / "corrupt_pivots.bin";
    auto offsets = write_pivots(path);
    const auto metadata = inspect_pq_pivots(path.string());
    check(metadata.dim == kDim && metadata.chunks == kChunks, "pivot metadata");
    patch_value(path, 8, uint64_t{0});
    rejects([&] { inspect_pq_pivots(path.string()); }, "invalid offset accepted");
    offsets = write_pivots(path);
    patch_value(path, offsets[2] + 8, uint32_t{0});
    rejects([&] { inspect_pq_pivots(path.string()); }, "duplicate permutation accepted");
    offsets = write_pivots(path);
    patch_value(path, offsets[2] + 8, uint32_t{kDim});
    rejects([&] { inspect_pq_pivots(path.string()); }, "out-of-range permutation accepted");
    offsets = write_pivots(path);
    patch_value(path, offsets[3] + 12, uint32_t{0});
    rejects([&] { inspect_pq_pivots(path.string()); }, "empty PQ chunk accepted");
    offsets = write_pivots(path);
    patch_value(path, offsets[3] + 16, uint32_t{4});
    rejects([&] { inspect_pq_pivots(path.string()); }, "incomplete chunks accepted");
    offsets = write_pivots(path);
    patch_value(path, offsets[0] + 8, std::numeric_limits<float>::quiet_NaN());
    rejects([&] { inspect_pq_pivots(path.string()); }, "non-finite pivots accepted");
    offsets = write_pivots(path);
    patch_value(path, offsets[1] + 8, std::numeric_limits<float>::infinity());
    rejects([&] { inspect_pq_pivots(path.string()); }, "non-finite mean accepted");
    write_bin<float>(path, 256, kDim, std::vector<float>(256 * kDim));
    rejects([&] { inspect_pq_pivots(path.string()); }, "legacy format silently accepted");
}

template <typename T>
void test_pq(const fs::path& dir, const std::string& dtype) {
    const auto path = dir / ("adc_" + dtype + "_pq_pivots.bin");
    write_pivots(path);
    const auto before_size = fs::file_size(path);
    PQCodebook<T> codebook(path.string(), kDim);
    check(codebook.chunks() == 2, "codebook chunks");
    rejects([&] { PQCodebook<T> wrong(path.string(), kDim + 1); }, "codebook dimension accepted");
    check(!fs::exists(path.string() + "_tables_T.runtime.bin"), "ADC loader wrote transpose cache");
    check(!fs::exists(path.string() + "_all_to_all.runtime.bin"), "ADC loader wrote pair-distance cache");
    check(fs::file_size(path) == before_size, "ADC loader changed pivots");

    const auto reference_path = dir / ("reference_" + dtype + "_pq_pivots.bin");
    write_pivots(reference_path);
    diskann::FixedChunkPQTable<T> reference;
    reference.load_pq_centroid_bin(reference_path.string().c_str(), kChunks);
    check(fs::exists(reference_path.string() + "_all_to_all.runtime.bin"), "legacy default cache behavior changed");

    const T first = std::is_same_v<T, uint8_t> ? static_cast<T>(250) : static_cast<T>(-30);
    const std::array<T, kDim> query{first, static_cast<T>(4), static_cast<T>(8),
                                   static_cast<T>(30), static_cast<T>(2)};
    std::array<uint8_t, kChunks> encoded{}, expected{};
    codebook.encode(query.data(), encoded.data());
    reference.deflate_vec(query.data(), expected.data());
    check(encoded == expected, "PQ encoder differs from existing implementation");
    std::vector<float> distances, expected_distances(256 * kChunks);
    codebook.populate_distances(query.data(), distances);
    reference.populate_chunk_distances(query.data(), expected_distances.data());
    check(distances == expected_distances, "ADC LUT differs from existing implementation");
    const std::array<uint8_t, kChunks> source_code{0, 255}, target_code{128, 10};
    float pair_distance = 0, pair_oracle = 0;
    reference.compute_distances(source_code.data(), target_code.data(), &pair_distance, 1);
    for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
        for (uint32_t d = kOffsets[chunk]; d < kOffsets[chunk + 1]; ++d) {
            const float delta = center(source_code[chunk], d) - center(target_code[chunk], d);
            pair_oracle += delta * delta;
        }
    }
    check(pair_distance == pair_oracle, "legacy pair-distance behavior changed");
    for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
        float best = std::numeric_limits<float>::max();
        uint8_t best_id = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            float value = 0;
            for (uint32_t d = kOffsets[chunk]; d < kOffsets[chunk + 1]; ++d) {
                const float delta = center(i, d) - (static_cast<float>(query[kPermutation[d]]) - kMean[kPermutation[d]]);
                value += delta * delta;
            }
            check(std::abs(distances[chunk * 256 + i] - value) < 0.001f, "ADC differs from reconstruction oracle");
            if (value < best) { best = value; best_id = static_cast<uint8_t>(i); }
        }
        check(encoded[chunk] == best_id, "PQ code differs from reconstruction oracle");
    }
    check(codebook.distance(encoded.data(), distances) ==
          distances[encoded[0]] + distances[256 + encoded[1]], "code lookup order");
    rejects([&] { codebook.encode(nullptr, encoded.data()); }, "null encoding input accepted");
    rejects([&] { codebook.distance(encoded.data(), {}); }, "short LUT accepted");

    std::atomic<bool> consistent{true};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            for (size_t rep = 0; rep < 100; ++rep) {
                std::vector<float> local_distances;
                std::array<uint8_t, kChunks> local_codes{};
                codebook.encode(query.data(), local_codes.data());
                codebook.populate_distances(query.data(), local_distances);
                if (local_codes != encoded || local_distances != distances) consistent = false;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    check(consistent, "concurrent PQ read mismatch");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--logger") {
            diskann::cout << std::string(4097, 'x') << "::end::" << std::flush;
            return 0;
        }
        if (argc != 2) throw std::invalid_argument("expected a fixture directory");
        const fs::path dir(argv[1]);
        fs::create_directories(dir);
        test_config();
        test_io(dir);
        test_pivot_validation(dir);
        test_pq<float>(dir, "float");
        test_pq<int8_t>(dir, "int8");
        test_pq<uint8_t>(dir, "uint8");
        std::cout << "IVF configuration, binary input, and PQ contracts passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << std::endl;
        return 1;
    }
}
