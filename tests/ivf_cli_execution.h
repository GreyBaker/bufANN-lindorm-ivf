#pragma once
#include "ivf_cli.h"
#include "bufann/ivf_config.h"
#include "bufann/ivf_io.h"

namespace ivf_cli {
void execute_build(const Arguments& args, const diskann::ivf::BuildConfig& config,
                   const diskann::ivf::BinMetadata& vectors, const diskann::ivf::PQMetadata& pq);
void execute_driver(const Arguments& args);
} // namespace ivf_cli
