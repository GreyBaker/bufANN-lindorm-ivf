#pragma once
#include <cstdint>
#include <iosfwd>
#include <span>
#include <vector>
namespace diskann::ivf {
  class CentroidPCA {
   public:
    static CentroidPCA fit(uint32_t dim, std::span<const float> centroids,
                           double   variance = .95,
                           uint64_t memory_bytes = 1ull << 30,
                           uint64_t work_limit = 1000000000);
    std::vector<float> project(std::span<const float> vector) const;
    void               write(std::ostream& out) const;
    static CentroidPCA read(std::istream& in, uint32_t dim,
                            std::span<const float> centroids, double variance,
                            uint64_t memory_bytes, uint64_t work_limit);
    uint32_t           retained_dim() const {
      return retained_;
    }
    double retained_variance() const {
      return retained_variance_;
    }
    bool degenerate() const {
      return degenerate_;
    }
    std::span<const float> projected() const {
      return projected_;
    }
    static uint64_t estimate_bytes(uint32_t dim, uint32_t count);
    // Necessary covariance-only lower bound; solver/validation retain their
    // runtime guards. Call before expensive sampling or graph construction.
    static void preflight_work(uint32_t dim, uint32_t count,
                               uint64_t work_limit);

   private:
    uint32_t            dim_ = 0, retained_ = 0;
    double              target_ = .95, retained_variance_ = 1;
    bool                degenerate_ = false;
    std::vector<double> mean_, basis_,
        eigenvalues_;  // Full deterministic eigensystem.
    std::vector<float> projected_;
    void validate(std::span<const float> centroids, uint64_t work_limit) const;
  };
}  // namespace diskann::ivf
