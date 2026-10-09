#pragma once

#include "bufann/ivf_heap.h"
#include "bufann/ivf_lists.h"
#include "bufann/ivf_posting_lists.h"
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
    RoutingRequest routing{}; // Query-only override; never changes list/insert assignments.
};
struct QueryResult {
    std::vector<uint32_t> selected_lists;
    std::vector<ExactHit> hits;
    uint64_t scanned_codes = 0;
    uint64_t reranked_vectors = 0;
    uint64_t backend_batch_calls = 0; // Not disk reads; cache hits may require no I/O.
    uint64_t heap_pages_read = 0; // Measured backend page reads; not device requests/IOPS.
    uint64_t logical_fetch_stages = 0;
    uint64_t stale_candidates = 0; // Mutable queries only: removed by live revalidation.
    // Per-query wall microseconds; scan includes bounded candidate selection.
    // Fetch includes lock wait, batch pins and aligned copies. Preflight/scratch
    // allocation overhead is not attributed to a stage. Totals across threads
    // are not whole-run wall time or CPU time.
    double routing_us = 0, adc_us = 0, scan_select_us = 0, fetch_us = 0, rerank_us = 0;
    RoutingStats router_stats{};
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

template<typename T>
uint64_t estimate_mutable_query_workspace(const CentroidRouter&, const MutablePostingLists&,
    const PQCodebook<T>&, const HeapStorage<T>&, const SearchConfig&, bool exact_routing = false);
template<typename T>
QueryResult search_mutable(const CentroidRouter&, const MutablePostingLists&,
    const PQCodebook<T>&, HeapStorage<T>&, std::span<const T>, const SearchConfig&, const QueryLimits& = {});
// Mutable queries are weakly consistent, not collection-wide snapshots:
// copied candidates are revalidated before/after heap I/O and result selection.
// Deletes after the final ID check may still overlap a returned result; inserts
// after a list scan may be absent. Deleted-ID reuse is forbidden.
// Copied-candidate composition seam, also used to test stale scan/fetch races.
// Validates IDs/uniqueness/budget and drops deleted IDs, not missing live records.
template<typename T>
QueryResult rerank_mutable_candidates(const CentroidRouter&, const MutablePostingLists&,
    const PQCodebook<T>&, HeapStorage<T>&, std::span<const T>, std::span<const PQCandidate>,
    const SearchConfig&, const QueryLimits& = {});
} // namespace diskann::ivf
