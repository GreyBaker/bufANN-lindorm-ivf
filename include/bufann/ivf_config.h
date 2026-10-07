#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace diskann::ivf {

enum class DataType { Float, Int8, UInt8 };

DataType parse_data_type(const std::string& name);
const char* data_type_name(DataType type);
size_t element_size(DataType type);

struct BuildConfig {
    DataType data_type = DataType::Float;
    uint32_t dim = 0;
    uint32_t base_count = 0;
    uint32_t nlist = 0;
    double sample_rate = 1.0;
    uint32_t max_reps = 15;
    uint32_t build_threads = 1;
    uint64_t seed = 0;
    double memory_budget_gb = 1.0;

    uint32_t training_count() const;
    void validate() const;
};

struct SearchConfig {
    uint32_t nprobe = 1;
    uint32_t recall_at = 10;
    double rerank_factor = 2.0;
    uint32_t query_threads = 1;

    void validate(uint32_t nlist) const;
    size_t candidate_budget(size_t live_count) const;
};

}  // namespace diskann::ivf
