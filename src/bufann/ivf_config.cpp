#include "bufann/ivf_config.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace diskann::ivf {

DataType parse_data_type(const std::string& name) {
    if (name == "float") return DataType::Float;
    if (name == "int8") return DataType::Int8;
    if (name == "uint8") return DataType::UInt8;
    throw std::invalid_argument("data_type must be float, int8, or uint8");
}

const char* data_type_name(DataType type) {
    switch (type) {
        case DataType::Float: return "float";
        case DataType::Int8: return "int8";
        case DataType::UInt8: return "uint8";
    }
    throw std::invalid_argument("invalid IVF data type");
}

size_t element_size(DataType type) {
    switch (type) {
        case DataType::Float: return sizeof(float);
        case DataType::Int8:
        case DataType::UInt8: return 1;
    }
    throw std::invalid_argument("invalid IVF data type");
}

uint32_t BuildConfig::training_count() const {
    if (!std::isfinite(sample_rate) || sample_rate <= 0 || sample_rate > 1) {
        throw std::invalid_argument("sample_rate must be finite and in (0, 1]");
    }
    return static_cast<uint32_t>(std::ceil(base_count * sample_rate));
}

void BuildConfig::validate() const {
    element_size(data_type);
    if (dim == 0 || base_count == 0 || nlist == 0) {
        throw std::invalid_argument("dim, base_count, and nlist must be positive");
    }
    if (nlist > training_count()) {
        throw std::invalid_argument("nlist exceeds the centroid training sample count");
    }
    if (max_reps == 0 || build_threads == 0) {
        throw std::invalid_argument("max_reps and build_threads must be positive");
    }
    if (!std::isfinite(memory_budget_gb) || memory_budget_gb <= 0) {
        throw std::invalid_argument("memory_budget_gb must be finite and positive");
    }
}

void SearchConfig::validate(uint32_t nlist) const {
    if (nprobe == 0 || nprobe > nlist) {
        throw std::invalid_argument("nprobe must be in [1, nlist]");
    }
    if (recall_at == 0 || query_threads == 0) {
        throw std::invalid_argument("recall_at and query_threads must be positive");
    }
    if (!std::isfinite(rerank_factor) || rerank_factor < 1) {
        throw std::invalid_argument("rerank_factor must be finite and >= 1");
    }
    const long double budget = std::ceil(static_cast<long double>(recall_at) * rerank_factor);
    if (budget >= static_cast<long double>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument("rerank candidate budget exceeds size_t");
    }
}

size_t SearchConfig::candidate_budget(size_t live_count) const {
    // Validate even if live_count is zero; do not hide an invalid factor.
    validate(nprobe);
    const long double budget = std::ceil(static_cast<long double>(recall_at) * rerank_factor);
    if (budget >= static_cast<long double>(live_count)) return live_count;
    return static_cast<size_t>(budget);
}

}  // namespace diskann::ivf
