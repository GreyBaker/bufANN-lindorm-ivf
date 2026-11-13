#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "bufann/ivf_defaults.h"

namespace diskann::ivf {

  enum class DataType { Float, Int8, UInt8 };
  enum class TrainerMode {
    Exact,
    AnnRebuild,
    AnnReuse,
    AnnReuseSeeded,
    AnnReuseCount,
    AnnReuseSeededCount,
    LV,
    LVPlus
  };
  enum class CentroidInit { Sample, RandomPartition };
  TrainerMode  parse_trainer_mode(const std::string& name);
  const char*  trainer_mode_name(TrainerMode mode);
  bool         trainer_reuses_graph(TrainerMode mode);
  bool         trainer_seeds_assignment(TrainerMode mode);
  bool         trainer_optimized_search(TrainerMode mode);
  bool         trainer_count_candidates(TrainerMode mode);
  CentroidInit parse_centroid_init(const std::string& name);
  const char*  centroid_init_name(CentroidInit init);

  DataType    parse_data_type(const std::string& name);
  const char* data_type_name(DataType type);
  size_t      element_size(DataType type);

  struct BuildConfig {
    DataType data_type = DataType::Float;
    uint32_t dim = 0;
    uint32_t base_count = 0;
    uint32_t nlist = defaults::nlist;
    double   sample_rate = 1.0;
    uint32_t max_reps = 15;
    uint32_t build_threads = defaults::build_threads;
    uint64_t seed = 0;
    double   memory_budget_gb = defaults::memory_budget_gb;
    TrainerMode  trainer_mode = TrainerMode::LVPlus;
    CentroidInit init = CentroidInit::RandomPartition;
    uint32_t     training_breadth = 64;
    // Independent of final query/insert router settings. Only LV/LV+ use these.
    double   training_rerank_factor = 3;
    uint32_t training_refine_breadth = 128;

    uint32_t training_count() const;
    void     validate() const;
  };

  struct SearchConfig {
    uint32_t nprobe = defaults::nprobe;
    uint32_t recall_at = 10;
    double   rerank_factor = 3.0;
    uint32_t query_threads = defaults::query_threads;

    void   validate(uint32_t nlist) const;
    size_t candidate_budget(size_t live_count) const;
  };

}  // namespace diskann::ivf
