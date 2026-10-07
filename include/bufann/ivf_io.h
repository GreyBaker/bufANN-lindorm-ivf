#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bufann/ivf_config.h"

namespace diskann::ivf {

struct BinMetadata {
    uint32_t rows;
    uint32_t cols;
};

// Validate packed little-endian .bin files without loading their payloads.
BinMetadata inspect_bin(const std::string& path, size_t bytes_per_element);
BinMetadata inspect_vectors(const std::string& path, DataType type, uint32_t dim);
BinMetadata inspect_pq_codes(const std::string& path, uint32_t rows, uint32_t chunks);
BinMetadata inspect_truth(const std::string& path, uint32_t query_count, uint32_t recall_at);
std::vector<uint32_t> load_counted_ids(const std::string& path, uint32_t id_capacity);

struct PQMetadata {
    uint32_t dim;
    uint32_t chunks;
};

// Validate the five-offset PQ format emitted by the current build helpers.
// Old sidecar-based pivots are deliberately rejected rather than guessed.
PQMetadata inspect_pq_pivots(const std::string& path);

}  // namespace diskann::ivf
