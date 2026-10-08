#pragma once

#include "bufann/ivf_centroids.h"
#include <functional>
#include <memory>
#include <span>

namespace diskann::ivf::detail {
class AssignmentIndex {
 public:
    virtual ~AssignmentIndex() = default;
    virtual uint32_t assign(std::span<const float> vector) const = 0;
};
using AssignmentFactory = std::function<std::unique_ptr<AssignmentIndex>(std::span<const float>)>;
// Shared sampling/reduction logic; optional graph assignment does not make the
// dependency-light exact trainer depend on hnswlib. Factory builds are serial,
// assignment must be immutable/thread-safe, and exceptions propagate to caller.
CentroidTraining train_centroids_impl(const std::string& path, const BuildConfig& config,
    const TrainerLimits& limits, const AssignmentFactory* factory, uint64_t index_workspace_bytes);
} // namespace diskann::ivf::detail
