#include "bufann/ivf_pq.h"
#include "ann_exception.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

using namespace diskann::ivf;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<typename T>
std::vector<T> block(std::ifstream& in, uint64_t offset, size_t count) {
    in.seekg(offset + 8);
    std::vector<T> data(count);
    in.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    check(bool(in), "Truncated parity fixture"); return data;
}
template<typename T>
void verify(const std::string& data, const std::string& prefix) {
    const auto shape = inspect_bin(data, sizeof(T));
    check(shape.rows <= 8192 && shape.cols <= 64, "Parity tool is fixture-only (8192 rows, 64 dimensions max)");
    const auto pivots = prefix + "_pq_pivots.bin";
    const auto metadata = inspect_pq_pivots(pivots, shape.cols);
    inspect_pq_codes(prefix + "_pq_compressed.bin", shape.rows, metadata.chunks);
    std::ifstream source(data, std::ios::binary), pq(pivots, std::ios::binary), codes(prefix + "_pq_compressed.bin", std::ios::binary);
    const auto vectors = block<T>(source, 0, size_t(shape.rows) * shape.cols);
    for (T value : vectors) check(std::isfinite(double(value)), "Nonfinite parity vector");
    const auto stored = block<uint8_t>(codes, 0, size_t(shape.rows) * metadata.chunks);
    const auto offsets = block<uint64_t>(pq, 0, 5);
    const auto centers = block<float>(pq, offsets[0], 256 * shape.cols);
    const auto mean = block<float>(pq, offsets[1], shape.cols);
    check(std::any_of(mean.begin(), mean.end(), [](float value) { return std::abs(value) > 1e-3f; }),
          "Trained fixture must exercise nonzero global centering");
    const auto permutation = block<uint32_t>(pq, offsets[2], shape.cols);
    const auto chunks = block<uint32_t>(pq, offsets[3], metadata.chunks + 1);
    PQCodebook<T> book(pivots, shape.cols, metadata.chunks);
    std::vector<uint8_t> encoded(metadata.chunks);
    std::vector<float> lut;
    uint64_t mismatches = 0, checked_lut_entries = 0;
    for (uint32_t row = 0; row < shape.rows; ++row) {
        const T* vector = vectors.data() + size_t(row) * shape.cols;
        book.encode(vector, encoded.data());
        for (uint32_t c = 0; c < metadata.chunks; ++c) {
            const auto helper_code = stored[size_t(row) * metadata.chunks + c];
            if (helper_code == encoded[c]) continue;
            ++mismatches;
            auto cost = [&](uint8_t code) {
                double distance = 0, norms = 0;
                for (uint32_t d = chunks[c]; d < chunks[c + 1]; ++d) {
                    const float q = float(vector[permutation[d]]) - mean[permutation[d]];
                    const float center = centers[size_t(code) * shape.cols + d];
                    distance += (double(q) - center) * (double(q) - center);
                    norms += double(q) * q + double(center) * center;
                }
                return std::pair{distance, norms};
            };
            const auto a = cost(helper_code), b = cost(encoded[c]);
            // SGEMM nearest-center assignment and scalar deflate need not be
            // byte-identical at float cancellation/ties. Require equivalent
            // chunk objectives within this predeclared numerical bound.
            const double allowance = 32 * std::numeric_limits<float>::epsilon() * (1 + a.second + b.second);
            check(std::abs(a.first - b.first) <= allowance, "Helper/native PQ code disagreement outside rounding allowance");
        }
        if (row % 257 != 0) continue;
        book.populate_distances(vector, lut);
        for (uint32_t c = 0; c < metadata.chunks; ++c) for (uint32_t code = 0; code < 256; ++code) {
            double reference = 0;
            for (uint32_t d = chunks[c]; d < chunks[c + 1]; ++d) {
                const float q = float(vector[permutation[d]]) - mean[permutation[d]];
                const double diff = centers[size_t(code) * shape.cols + d] - double(q);
                reference += diff * diff;
            }
            const float actual = lut[size_t(c) * 256 + code];
            check(std::isfinite(actual) && std::abs(actual - reference) <= 1e-5 * std::max(1.0, reference), "Trained LUT differs from reconstruction oracle");
            ++checked_lut_entries;
        }
    }
    std::cout << "Trained helper parity: rows=" << shape.rows << " dim=" << shape.cols
              << " chunks=" << metadata.chunks << " code_bytes=" << stored.size()
              << " rounding_equivalent_mismatches=" << mismatches << " lut_entries=" << checked_lut_entries << '\n';
}
int main(int argc, char** argv) {
    try {
        check(argc == 4, "Usage: ivf_pq_helper_parity float|int8|uint8 vectors.bin helper_prefix");
        switch (parse_data_type(argv[1])) {
            case DataType::Float: verify<float>(argv[2], argv[3]); break;
            case DataType::Int8: verify<int8_t>(argv[2], argv[3]); break;
            case DataType::UInt8: verify<uint8_t>(argv[2], argv[3]); break;
        }
        return 0;
    } catch (const diskann::ANNException& error) { std::cerr << "ERROR: " << error.message() << '\n'; }
      catch (const std::exception& error) { std::cerr << "ERROR: " << error.what() << '\n'; }
    return 1;
}
