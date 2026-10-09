#include "bufann/ivf_config.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace diskann::ivf {

TrainerMode parse_trainer_mode(const std::string& name) {
    if (name == "exact") return TrainerMode::Exact;
    if (name == "ann-rebuild") return TrainerMode::AnnRebuild;
    if (name == "ann-reuse") return TrainerMode::AnnReuse;
    if (name == "ann-reuse-seeded") return TrainerMode::AnnReuseSeeded;
    if (name == "ann-reuse-count") return TrainerMode::AnnReuseCount;
    if (name == "ann-reuse-seeded-count") return TrainerMode::AnnReuseSeededCount;
    if (name == "lv") return TrainerMode::LV;
    if (name == "lv-plus") return TrainerMode::LVPlus;
    throw std::invalid_argument("Unknown centroid trainer mode");
}
const char* trainer_mode_name(TrainerMode mode) {
    switch (mode) {
        case TrainerMode::Exact: return "exact";
        case TrainerMode::AnnRebuild: return "ann-rebuild";
        case TrainerMode::AnnReuse: return "ann-reuse";
        case TrainerMode::AnnReuseSeeded: return "ann-reuse-seeded";
        case TrainerMode::AnnReuseCount: return "ann-reuse-count";
        case TrainerMode::AnnReuseSeededCount: return "ann-reuse-seeded-count";
        case TrainerMode::LV: return "lv";
        case TrainerMode::LVPlus: return "lv-plus";
    }
    throw std::invalid_argument("Invalid centroid trainer mode");
}
bool trainer_reuses_graph(TrainerMode mode) {
    return mode != TrainerMode::Exact && mode != TrainerMode::AnnRebuild;
}
bool trainer_seeds_assignment(TrainerMode mode) {
    return mode == TrainerMode::AnnReuseSeeded || mode == TrainerMode::AnnReuseSeededCount || mode == TrainerMode::LVPlus;
}
bool trainer_optimized_search(TrainerMode mode) { return mode == TrainerMode::LV || mode == TrainerMode::LVPlus; }
bool trainer_count_candidates(TrainerMode mode) {
    return trainer_optimized_search(mode) || mode == TrainerMode::AnnReuseCount || mode == TrainerMode::AnnReuseSeededCount;
}
CentroidInit parse_centroid_init(const std::string& name) {
    if (name == "sample") return CentroidInit::Sample;
    if (name == "random_partition") return CentroidInit::RandomPartition;
    throw std::invalid_argument("init must be sample or random_partition");
}
const char* centroid_init_name(CentroidInit init) {
    switch (init) {
        case CentroidInit::Sample: return "sample";
        case CentroidInit::RandomPartition: return "random_partition";
    }
    throw std::invalid_argument("Invalid centroid initialization");
}

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
    trainer_mode_name(trainer_mode); centroid_init_name(init);
    if (!training_breadth || training_breadth > INT32_MAX)
        throw std::invalid_argument("training_breadth must be in [1, INT32_MAX]");
    if (!std::isfinite(training_rerank_factor) || training_rerank_factor < 1 ||
        !training_refine_breadth || training_refine_breadth > INT32_MAX)
        throw std::invalid_argument("Invalid training rerank/refinement settings");
    if (!trainer_optimized_search(trainer_mode) && (training_rerank_factor != 3 || training_refine_breadth != 128))
        throw std::invalid_argument("Training rerank/refinement overrides require lv or lv-plus");
    if (trainer_optimized_search(trainer_mode) && init != CentroidInit::RandomPartition)
        throw std::invalid_argument("Paper-aligned LV/LV+ requires random_partition initialization");
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
