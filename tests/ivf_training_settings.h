#pragma once

#include "bufann/ivf_config.h"
#include "ivf_cli.h"

namespace ivf_cli {
  // Shared build/calibration policy, independent of frozen router settings.
  inline void read_training_settings(const Arguments& args,
                                     diskann::ivf::BuildConfig& config) {
    using namespace diskann::ivf;
    config.trainer_mode =
        parse_trainer_mode(args.get("--trainer_mode", "ann-reuse-seeded"));
    config.init = parse_centroid_init(args.get("--init", "random_partition"));
    config.training_breadth = args.u32("--training_breadth", 64);
    config.training_rerank_factor = args.real("--training_rerank_factor", 3);
    config.training_refine_breadth = args.u32("--training_refine_breadth", 128);
  }
}  // namespace ivf_cli
