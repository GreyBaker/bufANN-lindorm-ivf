#include "bufann/ivf_pca.h"
#include <algorithm>
#include <cmath>
#include <istream>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace diskann::ivf {
  namespace {
    void check(bool ok, const char* why) {
      if (!ok)
        throw std::runtime_error(why);
    }
    template<class T>
    void put(std::ostream& out, T x) {
      out.write(reinterpret_cast<const char*>(&x), sizeof(x));
    }
    template<class T>
    T get(std::istream& in) {
      T x{};
      in.read(reinterpret_cast<char*>(&x), sizeof(x));
      check(bool(in), "Truncated PCA artifact");
      return x;
    }
    void tick(uint64_t& work, uint64_t n, uint64_t limit) {
      check(n <= limit - work, "PCA work limit exceeded");
      work += n;
    }
    std::vector<double> covariance(uint32_t dim, std::span<const float> v,
                                   std::span<const double> mean, uint64_t& work,
                                   uint64_t limit) {
      std::vector<double> a(size_t(dim) * dim, 0);
      for (size_t r = 0; r < v.size() / dim; ++r)
        for (uint32_t i = 0; i < dim; ++i)
          for (uint32_t j = i; j < dim; ++j) {
            tick(work, 1, limit);
            a[size_t(i) * dim + j] += (double(v[r * dim + i]) - mean[i]) *
                                      (double(v[r * dim + j]) - mean[j]);
          }
      for (uint32_t i = 0; i < dim; ++i)
        for (uint32_t j = i; j < dim; ++j) {
          a[size_t(i) * dim + j] /= double(v.size() / dim);
          a[size_t(j) * dim + i] = a[size_t(i) * dim + j];
          check(std::isfinite(a[size_t(i) * dim + j]),
                "Nonfinite PCA covariance");
        }
      return a;
    }
  }  // namespace
  uint64_t CentroidPCA::estimate_bytes(uint32_t dim, uint32_t count) {
    check(dim && count, "Invalid PCA shape");
    const uint64_t cells = uint64_t(dim) * dim, rows = uint64_t(dim) * count;
    check(cells <= (UINT64_MAX - 65536) / 80 &&
              rows <= (UINT64_MAX - 65536 - cells * 80) / 16,
          "PCA estimate overflow");
    return 65536 + cells * 80 + rows * 16;
  }
  std::vector<float> CentroidPCA::project(std::span<const float> v) const {
    check(v.size() == dim_, "PCA query dimension mismatch");
    std::vector<float> out(retained_, 0);
    for (uint32_t k = 0; k < retained_; ++k) {
      double sum = 0;
      for (uint32_t d = 0; d < dim_; ++d) {
        check(std::isfinite(v[d]), "Nonfinite PCA query");
        sum += (double(v[d]) - mean_[d]) * basis_[size_t(d) * dim_ + k];
      }
      check(std::isfinite(sum) &&
                std::abs(sum) <= std::numeric_limits<float>::max(),
            "PCA projection overflow");
      out[k] = float(sum);
    }
    return out;
  }
  CentroidPCA CentroidPCA::fit(uint32_t dim, std::span<const float> v,
                               double target, uint64_t budget, uint64_t limit) {
    check(dim && v.size() % dim == 0 && v.size() / dim <= INT32_MAX &&
              target > 0 && target <= 1 && std::isfinite(target) && limit,
          "Invalid PCA fit configuration");
    check(estimate_bytes(dim, uint32_t(v.size() / dim)) <= budget,
          "PCA fit exceeds allowance");
    CentroidPCA p;
    p.dim_ = dim;
    p.target_ = target;
    p.mean_.assign(dim, 0);
    uint64_t work = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      check(std::isfinite(v[i]), "Nonfinite PCA centroid");
      p.mean_[i % dim] += v[i];
    }
    for (auto& x : p.mean_) {
      x /= double(v.size() / dim);
      check(std::isfinite(x), "PCA mean overflow");
    }
    auto   a = covariance(dim, v, p.mean_, work, limit);
    double trace = 0;
    for (uint32_t d = 0; d < dim; ++d)
      trace += a[size_t(d) * dim + d];
    std::vector<double> eigenvectors(size_t(dim) * dim, 0);
    for (uint32_t d = 0; d < dim; ++d)
      eigenvectors[size_t(d) * dim + d] = 1;
    // Cyclic double Jacobi: no BLAS/runtime/dependency change in isolated
    // builds.
    bool converged = trace == 0;
    for (uint32_t sweep = 0; sweep < 64 && !converged; ++sweep) {
      double largest = 0;
      for (uint32_t i = 0; i < dim; ++i)
        for (uint32_t j = i + 1; j < dim; ++j) {
          const double off = a[size_t(i) * dim + j];
          largest = std::max(largest, std::abs(off));
          if (std::abs(off) <=
              std::max(trace, std::numeric_limits<double>::min()) * 1e-13)
            continue;
          const double theta =
              .5 * std::atan2(2 * off,
                              a[size_t(j) * dim + j] - a[size_t(i) * dim + i]);
          const double c = std::cos(theta), s = std::sin(theta);
          const double ai = a[size_t(i) * dim + i], aj = a[size_t(j) * dim + j];
          a[size_t(i) * dim + i] = c * c * ai - 2 * s * c * off + s * s * aj;
          a[size_t(j) * dim + j] = s * s * ai + 2 * s * c * off + c * c * aj;
          a[size_t(i) * dim + j] = a[size_t(j) * dim + i] = 0;
          tick(work, uint64_t(dim) * 4, limit);
          for (uint32_t k = 0; k < dim; ++k) {
            if (k != i && k != j) {
              const double x = a[size_t(k) * dim + i],
                           y = a[size_t(k) * dim + j];
              a[size_t(k) * dim + i] = a[size_t(i) * dim + k] = c * x - s * y;
              a[size_t(k) * dim + j] = a[size_t(j) * dim + k] = s * x + c * y;
            }
            const double x = eigenvectors[size_t(k) * dim + i],
                         y = eigenvectors[size_t(k) * dim + j];
            eigenvectors[size_t(k) * dim + i] = c * x - s * y;
            eigenvectors[size_t(k) * dim + j] = s * x + c * y;
          }
        }
      converged = largest <=
                  std::max(trace, std::numeric_limits<double>::min()) * 1e-11;
    }
    check(converged, "PCA eigensolve did not converge");
    std::vector<uint32_t> order(dim);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto i, auto j) {
      return a[size_t(i) * dim + i] > a[size_t(j) * dim + j];
    });
    p.eigenvalues_.resize(dim);
    p.basis_.resize(size_t(dim) * dim);
    for (uint32_t k = 0; k < dim; ++k) {
      const double value = a[size_t(order[k]) * dim + order[k]];
      check(
          value >= -std::max(trace, std::numeric_limits<double>::min()) * 1e-10,
          "Invalid negative PCA eigenvalue");
      p.eigenvalues_[k] = std::max(0.0, value);
      for (uint32_t d = 0; d < dim; ++d)
        p.basis_[size_t(d) * dim + k] =
            eigenvectors[size_t(d) * dim + order[k]];
    }
    // Canonical coordinate-axis basis for repeated eigenvalue subspaces.
    for (uint32_t begin = 0; begin < dim;) {
      uint32_t end = begin + 1;
      while (end < dim &&
             std::abs(p.eigenvalues_[end] - p.eigenvalues_[begin]) <=
                 std::max(trace, std::numeric_limits<double>::min()) * 1e-10)
        ++end;
      if (end - begin > 1) {
        const auto old = p.basis_;
        uint32_t   written = begin;
        for (uint32_t axis = 0; axis < dim && written < end; ++axis) {
          std::vector<double> q(dim, 0);
          for (uint32_t d = 0; d < dim; ++d)
            for (uint32_t k = begin; k < end; ++k) {
              tick(work, 1, limit);
              q[d] += old[size_t(d) * dim + k] * old[size_t(axis) * dim + k];
            }
          for (uint32_t k = begin; k < written; ++k) {
            double dot = 0;
            for (uint32_t d = 0; d < dim; ++d)
              dot += q[d] * p.basis_[size_t(d) * dim + k];
            for (uint32_t d = 0; d < dim; ++d)
              q[d] -= dot * p.basis_[size_t(d) * dim + k];
          }
          double norm = 0;
          for (double x : q)
            norm += x * x;
          if (norm <= 1e-16)
            continue;
          for (uint32_t d = 0; d < dim; ++d)
            p.basis_[size_t(d) * dim + written] = q[d] / std::sqrt(norm);
          ++written;
        }
        check(written == end, "PCA degenerate basis construction failed");
      }
      for (uint32_t k = begin; k < end; ++k) {
        uint32_t pivot = 0;
        for (uint32_t d = 1; d < dim; ++d)
          if (std::abs(p.basis_[size_t(d) * dim + k]) >
              std::abs(p.basis_[size_t(pivot) * dim + k]))
            pivot = d;
        if (p.basis_[size_t(pivot) * dim + k] < 0)
          for (uint32_t d = 0; d < dim; ++d)
            p.basis_[size_t(d) * dim + k] *= -1;
      }
      begin = end;
    }
    p.degenerate_ = trace == 0;
    double total = std::accumulate(p.eigenvalues_.begin(), p.eigenvalues_.end(),
                                   0.0),
           kept = 0;
    if (p.degenerate_) {
      p.retained_ = dim;
      p.retained_variance_ = 1;
    } else {
      do {
        kept += p.eigenvalues_[p.retained_++];
      } while (p.retained_ < dim && kept / total < target);
      p.retained_variance_ = kept / total;
    }
    for (size_t r = 0; r < v.size() / dim; ++r) {
      auto row = p.project(v.subspan(r * dim, dim));
      p.projected_.insert(p.projected_.end(), row.begin(), row.end());
    }
    p.validate(v, limit);
    return p;
  }
  void CentroidPCA::validate(std::span<const float> v, uint64_t limit) const {
    check(dim_ && retained_ && retained_ <= dim_ && v.size() % dim_ == 0 &&
              mean_.size() == dim_ && basis_.size() == uint64_t(dim_) * dim_ &&
              eigenvalues_.size() == dim_ &&
              projected_.size() == (v.size() / dim_) * retained_ &&
              target_ > 0 && target_ <= 1 && std::isfinite(target_) &&
              std::isfinite(retained_variance_),
          "Invalid PCA framing/settings");
    for (auto x : mean_)
      check(std::isfinite(x), "Invalid PCA mean");
    for (auto x : basis_)
      check(std::isfinite(x), "Invalid PCA basis");
    uint64_t            work = 0;
    std::vector<double> means(dim_, 0);
    for (size_t i = 0; i < v.size(); ++i) {
      check(std::isfinite(v[i]), "Invalid PCA source centroid");
      means[i % dim_] += v[i];
    }
    for (uint32_t d = 0; d < dim_; ++d)
      check(std::abs(means[d] / double(v.size() / dim_) - mean_[d]) <=
                1e-10 * std::max(1.0, std::abs(mean_[d])),
            "PCA mean/source mismatch");
    const auto cov = covariance(dim_, v, mean_, work, limit);
    double     trace = 0;
    for (uint32_t d = 0; d < dim_; ++d)
      trace += cov[size_t(d) * dim_ + d];
    check(degenerate_ == (trace == 0), "PCA zero-variance mismatch");
    if (degenerate_)
      for (uint32_t d = 0; d < dim_; ++d)
        for (uint32_t k = 0; k < dim_; ++k)
          check(basis_[size_t(d) * dim_ + k] == (d == k ? 1 : 0),
                "Zero-variance PCA requires the identity basis");
    double total = 0, kept = 0;
    for (uint32_t k = 0; k < dim_; ++k) {
      uint32_t pivot = 0;
      for (uint32_t d = 1; d < dim_; ++d)
        if (std::abs(basis_[size_t(d) * dim_ + k]) >
            std::abs(basis_[size_t(pivot) * dim_ + k]))
          pivot = d;
      check(basis_[size_t(pivot) * dim_ + k] > 0,
            "Noncanonical PCA eigenvector sign");
      check(std::isfinite(eigenvalues_[k]) && eigenvalues_[k] >= 0 &&
                (k == 0 ||
                 eigenvalues_[k] <= eigenvalues_[k - 1] + trace * 1e-10),
            "Invalid PCA eigenvalue order");
      total += eigenvalues_[k];
      if (k < retained_)
        kept += eigenvalues_[k];
      for (uint32_t j = k; j < dim_; ++j) {
        tick(work, dim_, limit);
        double dot = 0;
        for (uint32_t d = 0; d < dim_; ++d)
          dot += basis_[size_t(d) * dim_ + k] * basis_[size_t(d) * dim_ + j];
        check(std::abs(dot - (k == j ? 1 : 0)) <= 1e-7,
              "Nonorthogonal PCA basis");
      }
      for (uint32_t d = 0; d < dim_; ++d) {
        tick(work, dim_, limit);
        double cv = 0;
        for (uint32_t j = 0; j < dim_; ++j)
          cv += cov[size_t(d) * dim_ + j] * basis_[size_t(j) * dim_ + k];
        check(std::abs(cv - eigenvalues_[k] * basis_[size_t(d) * dim_ + k]) <=
                  std::max(trace, 1e-300) * 1e-7,
              "PCA eigenvector/source mismatch");
      }
    }
    check(std::abs(total - trace) <= std::max(trace, 1e-300) * 1e-7,
          "PCA trace mismatch");
    const double ratio = degenerate_ ? 1 : kept / total;
    check(std::abs(retained_variance_ - ratio) < 1e-8 &&
              ratio + 1e-10 >= target_ &&
              (degenerate_ ? retained_ == dim_
                           : (retained_ == 1 ||
                              (kept - eigenvalues_[retained_ - 1]) / total <
                                  target_ + 1e-12)),
          "Invalid PCA retained dimension/variance");
    for (size_t r = 0; r < v.size() / dim_; ++r) {
      const auto row = project(v.subspan(r * dim_, dim_));
      for (uint32_t k = 0; k < retained_; ++k)
        check(row[k] == projected_[r * retained_ + k],
              "PCA projected centroid mismatch");
    }
  }
  void CentroidPCA::write(std::ostream& out) const {
    put(out, uint64_t(0x31414350564649ull));
    put(out, dim_);
    put(out, retained_);
    put(out, uint32_t(projected_.size() / retained_));
    put(out, target_);
    put(out, retained_variance_);
    put(out, uint32_t(degenerate_));
    for (const auto* data : {&mean_, &basis_, &eigenvalues_})
      for (auto x : *data)
        put(out, x);
    for (auto x : projected_)
      put(out, x);
    check(bool(out), "PCA write failed");
  }
  CentroidPCA CentroidPCA::read(std::istream& in, uint32_t dim,
                                std::span<const float> v, double target,
                                uint64_t budget, uint64_t limit) {
    check(dim && v.size() % dim == 0 &&
              estimate_bytes(dim, uint32_t(v.size() / dim)) <= budget,
          "PCA load allowance/shape invalid");
    check(get<uint64_t>(in) == 0x31414350564649ull && get<uint32_t>(in) == dim,
          "Invalid PCA format/dimension");
    CentroidPCA p;
    p.dim_ = dim;
    p.retained_ = get<uint32_t>(in);
    check(p.retained_ && p.retained_ <= dim &&
              get<uint32_t>(in) == v.size() / dim,
          "Invalid PCA counts");
    p.target_ = get<double>(in);
    p.retained_variance_ = get<double>(in);
    const auto degenerate = get<uint32_t>(in);
    check(p.target_ == target && degenerate <= 1, "PCA config mismatch");
    p.degenerate_ = degenerate != 0;
    p.mean_.resize(dim);
    p.basis_.resize(size_t(dim) * dim);
    p.eigenvalues_.resize(dim);
    p.projected_.resize((v.size() / dim) * p.retained_);
    for (auto* data : {&p.mean_, &p.basis_, &p.eigenvalues_})
      for (auto& x : *data)
        x = get<double>(in);
    for (auto& x : p.projected_)
      x = get<float>(in);
    p.validate(v, limit);
    return p;
  }
}  // namespace diskann::ivf
