#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "bufann/ivf_router.h"

namespace diskann::ivf {
struct ListBuildOptions {
    uint32_t base_count = 0;
    uint32_t first_id = 0;
    uint32_t id_capacity = 0;
    uint32_t chunks = 0;
    uint64_t memory_budget_bytes = 1024ull * 1024 * 1024;
    bool exact_assignment = false; // Small-fixture oracle, not the production default.
    uint64_t max_assignment_searches = 1000000000;
    uint64_t max_exact_distance_evaluations = 1000000000;
};
struct PQCandidate {
    uint32_t id;
    float squared_distance;
    bool operator==(const PQCandidate&) const = default;
};
struct ListScan {
    std::vector<PQCandidate> candidates; // Nearest first; ID breaks distance ties.
    uint64_t scanned_codes = 0;
};
struct ScanProfile { double per_code_us = 0; uint64_t scanned_codes = 0; };

// Immutable CSR ID/code lists. Global, non-residual 8-bit PQ only. IDs remain
// first_id + input row; no insertion tail, coordinates or vector graph stored.
// All files must remain immutable throughout construction and loading.
class PostingLists {
 public:
    template<typename T>
    static PostingLists build(const std::string& vectors, const std::string& codes,
                              const CentroidRouter& router, const ListBuildOptions& options);
    void save(const std::string& path) const; // Exclusive new file, clean/quiescent access.
    static PostingLists load(const std::string& path, uint64_t workspace_bytes = 1024ull * 1024 * 1024);
    uint32_t dim() const { return _dim; }
    DataType data_type() const { return _type; }
    uint32_t first_id() const { return _first_id; }
    uint32_t id_capacity() const { return _id_capacity; }
    uint32_t nlist() const { return _offsets.empty() ? 0 : static_cast<uint32_t>(_offsets.size() - 1); }
    uint32_t chunks() const { return _chunks; }
    uint32_t size() const { return static_cast<uint32_t>(_ids.size()); }
    uint64_t estimated_build_bytes() const { return _estimated_bytes; }
    uint64_t estimated_load_bytes() const { return _estimated_load_bytes; }
    std::span<const uint32_t> ids(uint32_t list) const;
    std::span<const uint8_t> codes(uint32_t list) const;
    // Caller supplies one chunks*256 ADC table. Reject duplicate/invalid lists,
    // malformed/nonfinite tables and zero budget; keep at most budget entries.
    ListScan scan(std::span<const uint32_t> lists, std::span<const float> distances,
                  uint32_t budget, uint64_t workspace_bytes = 64ull * 1024 * 1024) const;
    // W8 uses the exact W4 scan/selection path on one list of real source codes.
    // Setup is outside timing; no alternative ADC kernel or per-list model term.
    static ScanProfile profile_scan(std::span<const uint8_t> codes, uint32_t chunks,
        std::span<const float> distances, uint32_t budget, uint32_t repeats,
        uint64_t workspace_bytes = 64ull * 1024 * 1024);
 private:
    PostingLists() = default;
    uint32_t _chunks = 0;
    uint32_t _dim = 0;
    DataType _type = DataType::Float;
    uint32_t _first_id = 0;
    uint32_t _id_capacity = 0;
    uint64_t _estimated_bytes = 0;
    uint64_t _estimated_load_bytes = 0;
    std::vector<uint64_t> _offsets;
    std::vector<uint32_t> _ids;
    std::vector<uint8_t> _codes;
};
} // namespace diskann::ivf
