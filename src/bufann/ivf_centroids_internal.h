#pragma once

#include "bufann/ivf_centroids.h"
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>

namespace diskann::ivf {
  struct RouterOptions;
}
namespace diskann::ivf::detail {
  struct AssignmentMeasurements {
    uint64_t graph_work = 0, returned_candidates = 0, rerank_distances = 0;
    double   projection_seconds = 0, traversal_seconds = 0, rerank_seconds = 0;
  };
  class AssignmentIndex {
   public:
    virtual ~AssignmentIndex() = default;
    virtual uint32_t assign(std::span<const float> vector, uint32_t previous,
                            uint64_t& work) const = 0;
    virtual uint32_t assign_measured(std::span<const float>  vector,
                                     uint32_t                previous,
                                     AssignmentMeasurements& m) const {
      return assign(vector, previous, m.graph_work);
    }
    virtual void describe_pass(TrainingPass&) const {
    }
    virtual void refresh(std::span<const float>) {
      throw std::logic_error("Assignment index cannot refresh");
    }
    virtual uint32_t unreachable() const {
      return 0;
    }
  };
  using AssignmentFactory =
      std::function<std::unique_ptr<AssignmentIndex>(std::span<const float>)>;
  // Quiescent construction; centroid storage must outlive the assignment index.
  std::unique_ptr<AssignmentIndex> make_reusable_assignment(
      std::span<const float> centroids, const BuildConfig& config,
      const RouterOptions& options);
  // Shared sampling/reduction logic; optional graph assignment does not make
  // the dependency-light exact trainer depend on hnswlib. Factory builds are
  // serial, assignment must be immutable/thread-safe, and exceptions propagate
  // to caller.
  CentroidTraining train_centroids_impl(
      const std::string& path, const BuildConfig& config,
      const TrainerLimits& limits, const AssignmentFactory* factory,
      uint64_t index_workspace_bytes, bool reuse = false, bool seeded = false);
}  // namespace diskann::ivf::detail
