#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "bufann/ivf_io.h"
#include "pq_table.h"

namespace diskann::ivf {

  // Owns the existing quantizer; only the caller's scratch is mutated by ADC.
  // Load once before starting workers. No residual quantization or retraining.
  template<typename T>
  class PQCodebook {
   public:
    explicit PQCodebook(const std::string& pivots_path, uint32_t expected_dim,
                        uint32_t expected_chunks = 0)
        : metadata_(expected_dim > 0
                        ? inspect_pq_pivots(pivots_path, expected_dim)
                        : throw std::invalid_argument(
                              "PQ dimension must be positive")),
          table_(std::make_unique<FixedChunkPQTable<T>>()) {
      if (metadata_.dim != expected_dim)
        throw std::invalid_argument("PQ dimension mismatch");
      if (expected_chunks != 0 && metadata_.chunks != expected_chunks)
        throw std::invalid_argument("PQ chunk count mismatch");
      table_->load_pq_centroid_bin(pivots_path.c_str(), metadata_.chunks, 0,
                                   false);
    }

    uint32_t dim() const {
      return metadata_.dim;
    }
    uint32_t chunks() const {
      return metadata_.chunks;
    }

    void encode(const T* vector, uint8_t* code) const {
      if (vector == nullptr || code == nullptr)
        throw std::invalid_argument("null PQ encode buffer");
      table_->deflate_vec(vector, code);
    }

    void populate_distances(const T*            query,
                            std::vector<float>& distances) const {
      if (query == nullptr)
        throw std::invalid_argument("null PQ query");
      distances.resize(static_cast<size_t>(256) * chunks());
      table_->populate_chunk_distances(query, distances.data());
    }

    float distance(const uint8_t*            code,
                   const std::vector<float>& distances) const {
      if (code == nullptr ||
          distances.size() != static_cast<size_t>(256) * chunks()) {
        throw std::invalid_argument("invalid PQ lookup buffers");
      }
      float result = 0;
      for (uint32_t c = 0; c < chunks(); ++c)
        result += distances[static_cast<size_t>(256) * c + code[c]];
      return result;
    }

   private:
    PQMetadata                            metadata_;
    std::unique_ptr<FixedChunkPQTable<T>> table_;
  };

}  // namespace diskann::ivf
