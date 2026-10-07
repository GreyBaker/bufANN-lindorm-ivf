#include "bufann/ivf_io.h"

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace diskann::ivf {
namespace {

[[noreturn]] void invalid(const std::string& path, const std::string& reason) {
    throw std::runtime_error(path + ": " + reason);
}

uint64_t checked_product(uint64_t a, uint64_t b, const std::string& path) {
    if (b != 0 && a > (std::numeric_limits<uint64_t>::max() - 8) / b) {
        invalid(path, "payload size overflow");
    }
    return a * b;
}

class Reader {
  public:
    explicit Reader(const std::string& path) : path_(path), in_(path, std::ios::binary) {
        if (!in_) invalid(path, "cannot open file");
        size = std::filesystem::file_size(path);
        if (size > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            invalid(path, "file exceeds stream offset range");
        }
    }

    uint64_t integer(uint64_t offset, size_t bytes) {
        if (offset > size || bytes > size - offset) invalid(path_, "truncated metadata");
        // Sequential metadata/code streams must not seek for every element:
        // that defeats ifstream buffering, particularly on large ID streams.
        if (offset != cursor_) in_.seekg(static_cast<std::streamoff>(offset));
        std::array<unsigned char, 8> raw{};
        in_.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
        if (!in_) invalid(path_, "metadata read failed");
        cursor_ = offset + bytes;
        uint64_t value = 0;
        for (size_t i = 0; i < bytes; ++i) value |= static_cast<uint64_t>(raw[i]) << (8 * i);
        return value;
    }

    BinMetadata header(uint64_t offset) {
        const uint32_t rows = static_cast<uint32_t>(integer(offset, 4));
        const uint32_t cols = static_cast<uint32_t>(integer(offset + 4, 4));
        if (rows == 0 || cols == 0 || rows > INT32_MAX || cols > INT32_MAX) {
            invalid(path_, ".bin counts must be positive signed int32 values");
        }
        return {rows, cols};
    }

    uint64_t block_end(uint64_t offset, BinMetadata shape, size_t elem_size) {
        uint64_t payload = checked_product(shape.rows, shape.cols, path_);
        payload = checked_product(payload, elem_size, path_);
        if (offset > size || size - offset < 8 || payload > size - offset - 8) {
            invalid(path_, "truncated payload");
        }
        return offset + 8 + payload;
    }

    void floats(uint64_t offset, uint64_t count) {
        for (uint64_t i = 0; i < count; ++i) {
            const uint32_t bits = static_cast<uint32_t>(integer(offset + i * 4, 4));
            float value;
            static_assert(sizeof(value) == sizeof(bits));
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value)) invalid(path_, "non-finite PQ value");
        }
    }

    uint64_t size = 0;

  private:
    std::string path_;
    std::ifstream in_;
    uint64_t cursor_ = 0;
};

}  // namespace

BinMetadata inspect_bin(const std::string& path, size_t bytes_per_element) {
    if (bytes_per_element == 0) throw std::invalid_argument("element size must be positive");
    Reader reader(path);
    const auto shape = reader.header(0);
    if (reader.block_end(0, shape, bytes_per_element) != reader.size) {
        invalid(path, "unexpected trailing bytes");
    }
    return shape;
}

BinMetadata inspect_vectors(const std::string& path, DataType type, uint32_t dim) {
    const auto shape = inspect_bin(path, element_size(type));
    if (shape.cols != dim) invalid(path, "vector dimension mismatch");
    return shape;
}

BinMetadata inspect_pq_codes(const std::string& path, uint32_t rows, uint32_t chunks) {
    const auto shape = inspect_bin(path, 1);
    if (shape.rows != rows || shape.cols != chunks) invalid(path, "PQ code shape mismatch");
    return shape;
}

BinMetadata inspect_truth(const std::string& path, uint32_t query_count, uint32_t recall_at) {
    // GT is header + all uint32 IDs + all float distances, not interleaved.
    const auto shape = inspect_bin(path, 8);
    if (shape.rows != query_count || shape.cols < recall_at) invalid(path, "ground truth shape mismatch");
    return shape;
}

std::vector<uint32_t> load_counted_ids(const std::string& path, uint32_t id_capacity) {
    Reader reader(path);
    const uint32_t count = static_cast<uint32_t>(reader.integer(0, 4));
    if (count > INT32_MAX || reader.size != 4 + static_cast<uint64_t>(count) * 4) {
        invalid(path, "invalid counted-ID length");
    }
    if (count > id_capacity) invalid(path, "ID count exceeds dataset capacity");
    std::vector<uint32_t> ids;
    ids.reserve(count);
    std::unordered_set<uint32_t> seen;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t id = static_cast<uint32_t>(reader.integer(4 + static_cast<uint64_t>(i) * 4, 4));
        if (id >= id_capacity || !seen.insert(id).second) invalid(path, "out-of-range or duplicate ID");
        ids.push_back(id);
    }
    return ids;
}

PQMetadata inspect_pq_pivots(const std::string& path) {
    Reader reader(path);
    const auto header = reader.header(0);
    if (header.rows != 5 || header.cols != 1) invalid(path, "expected five-offset PQ pivots format");
    std::array<uint64_t, 5> offsets{};
    for (size_t i = 0; i < offsets.size(); ++i) {
        offsets[i] = reader.integer(8 + i * 8, 8);
        if (offsets[i] > reader.size || (i == 0 ? offsets[i] < 48 : offsets[i] <= offsets[i - 1])) {
            invalid(path, "invalid PQ block offsets");
        }
    }
    if (offsets.back() != reader.size) invalid(path, "PQ end offset differs from file size");
    std::array<BinMetadata, 4> shapes{};
    for (size_t i = 0; i < shapes.size(); ++i) {
        shapes[i] = reader.header(offsets[i]);
        if (reader.block_end(offsets[i], shapes[i], 4) != offsets[i + 1]) {
            invalid(path, "PQ block length differs from offset table");
        }
    }
    const uint32_t dim = shapes[0].cols;
    if (shapes[0].rows != 256 || shapes[1].rows != dim || shapes[1].cols != 1 ||
        shapes[2].rows != dim || shapes[2].cols != 1 || shapes[3].cols != 1 ||
        shapes[3].rows < 2 || shapes[3].rows - 1 > dim) {
        invalid(path, "invalid PQ table shapes");
    }
    std::vector<bool> seen(dim, false);
    for (uint32_t i = 0; i < dim; ++i) {
        const uint64_t value = reader.integer(offsets[2] + 8 + static_cast<uint64_t>(i) * 4, 4);
        if (value >= dim || seen[value]) invalid(path, "PQ rearrangement is not a permutation");
        seen[value] = true;
    }
    uint64_t previous = 0;
    for (uint32_t i = 0; i < shapes[3].rows; ++i) {
        const uint64_t value = reader.integer(offsets[3] + 8 + static_cast<uint64_t>(i) * 4, 4);
        if ((i == 0 && value != 0) || (i > 0 && value <= previous) || value > dim) {
            invalid(path, "invalid PQ chunk boundaries");
        }
        previous = value;
    }
    if (previous != dim) invalid(path, "PQ chunks do not cover all dimensions");
    reader.floats(offsets[0] + 8, static_cast<uint64_t>(256) * dim);
    reader.floats(offsets[1] + 8, dim);
    return {dim, shapes[3].rows - 1};
}

}  // namespace diskann::ivf
