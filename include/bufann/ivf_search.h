#pragma once

#include "bufann/ivf_heap.h"
#include "bufann/ivf_lists.h"
#include "bufann/ivf_pq.h"

namespace diskann::ivf {
struct ExactHit {
    uint32_t id;
    double squared_distance;
    bool operator==(const ExactHit&) const = default;
};
struct QueryLimits {
    uint64_t workspace_bytes = 64ull * 1024 * 1024;
    bool exact_routing = false;
};
struct QueryResult {
    std::vector<uint32_t> selected_lists;
    std::vector<ExactHit> hits;
    uint64_t scanned_codes = 0;
    uint64_t reranked_vectors = 0;
    uint64_t backend_batch_calls = 0; // Not disk reads; cache hits may require no I/O.
    uint64_t heap_pages_read = 0; // Measured backend page reads; not device requests/IOPS.
    uint64_t logical_fetch_stages = 0;
    // Per-query wall microseconds; scan includes bounded candidate selection.
    // Fetch includes lock wait, batch pins and aligned copies. Preflight/scratch
    // allocation overhead is not attributed to a stage. Totals across threads
    // are not whole-run wall time or CPU time.
    double routing_us = 0, adc_us = 0, scan_select_us = 0, fetch_us = 0, rerank_us = 0;
};
template<typename T>
uint64_t estimate_query_workspace(const CentroidRouter& router, const PostingLists& lists,
    const PQCodebook<T>& codebook, const HeapStorage<T>& heap, const SearchConfig& config,
    bool exact_routing = false);

// Query-only composition seam, not a benchmark driver. Immutable router,
// codebook/lists and clean heap required; per-query scratch is never shared.
// Returned hits use exact double L2 over heap coordinates for PQ candidates.
// Not exact over unprobed lists or candidates pruned by the rerank budget.
template<typename T>
QueryResult search(const CentroidRouter& router, const PostingLists& lists,
                   const PQCodebook<T>& codebook, HeapStorage<T>& heap,
                   std::span<const T> query, const SearchConfig& config,
                   const QueryLimits& limits = {});
} // namespace diskann::ivf
