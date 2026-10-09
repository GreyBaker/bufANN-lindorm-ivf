#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace diskann::ivf {

enum class DataType { Float, Int8, UInt8 };
enum class TrainerMode { Exact, AnnRebuild, AnnReuse, AnnReuseSeeded };
enum class CentroidInit { Sample, RandomPartition };
TrainerMode parse_trainer_mode(const std::string& name);
const char* trainer_mode_name(TrainerMode mode);
CentroidInit parse_centroid_init(const std::string& name);
const char* centroid_init_name(CentroidInit init);

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
    // Preserve programmatic/legacy fixture defaults. Native CLI/harness defaults
    // explicitly select AnnReuseSeeded + RandomPartition for new builds.
    TrainerMode trainer_mode = TrainerMode::AnnRebuild;
    CentroidInit init = CentroidInit::Sample;
    uint32_t training_breadth = 64;

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
