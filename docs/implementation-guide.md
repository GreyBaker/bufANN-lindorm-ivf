# IVF-only: Implementation Guide

A single-node, mutable, disk-backed approximate nearest-neighbor (ANN) index inspired by LindormVector. This guide explains the current source, the reasons behind its design, and the limits of its guarantees. Paths are relative to `ivf-only/`.

## 1. Architecture and mental model

Think of centroids as library sections, posting lists as their catalogs, PQ codes as short book summaries, and heap records as complete books. The query chooses sections, compares summaries, and retrieves only promising complete books.

```text
BUILD: base vectors → centroid training → final centroid router
       → grouped IDs/PQ codes + graph-free coordinate heap

QUERY: query → centroid router → nprobe lists → scalar PQ scan
       → bounded finalists → coordinate batches → exact top-k

UPDATE: frozen router/PQ → prepare coordinates → publish posting
        delete posting by swap-with-last → retain physical record
```

`NativeIndex<T>` owns a `CentroidRouter`, `PostingLists`, `PQCodebook<T>`, and `HeapStorage<T>`. `MutableIndex<T>` adds mutable lists, lifecycle admission, and failure tracking. Supported coordinate types are float32, int8, and uint8. Centroids use float coordinates. Final distances use double-precision squared L2: `Σ(q[d] − x[d])²`.

**“IVF-only” does not mean graph-free search.** No DiskANN/Vamana graph over database vectors is required, but hierarchical navigable small world (HNSW) graphs over centroids accelerate routing and training. The heap reuses legacy storage machinery without database-vector edges.

### Scope and reading plan

| Sections | Subject | Learning outcome |
|---|---|---|
| 2 | Data and build | Trace source rows into artifacts. |
| 3–4 | Training and routing | Separate learning, traversal, and rerank budgets. |
| 5–6 | PQ, queries, storage | Explain approximation and bounded fetching. |
| 7–8 | Updates and concurrency | Reason about visibility, failures, and checkpoints. |
| 9–10 | Cost model and evaluation | Interpret predictions and measurements correctly. |

**Not implemented here:** distributed execution, multi-tenancy, hybrid/filtered search, residual IVF-PQ, 4-bit single-instruction/multiple-data (SIMD) fast scan, write-ahead-log (WAL) recovery, or physical slot reclamation. Functional fixtures are not production or paper-speedup qualification.

## 2. Data layout, artifacts, and native build

### Input and identity contracts

Vector `.bin` inputs have an 8-byte row-count/dimension header followed by packed coordinates. Dtype is supplied externally. The first `base_count` rows form the initial index; the remaining source rows may supply insertions. The benchmark preserves source-row IDs. The library permits an offset: `ID = first_id + source_row`, bounded by `id_capacity`.

IVF consumes existing global PQ pivots and codes; it does not train PQ during native build. Codes must cover the full source row population with the matching chunk count, including reserved insertion rows. Shapes and digests are checked, but external code-to-vector parity and training provenance are not independently proved. Inputs must remain trusted and immutable.

### Immutable lists: compact grouped arrays

One vector belongs to one centroid list. Immutable postings use compressed-sparse-row (CSR)-like storage: `offsets[nlist+1]`, grouped `ids[N]`, and aligned-by-entry `codes[N × chunks]`. List `c` spans `[offsets[c], offsets[c+1])`. Full coordinates are separate and accessed by ID; physical heap placement need not follow list order.

### Native build sequence

1. **Validate/admit:** reject output collisions; validate shapes, identities, memory estimates, and relevant work limits. Reserve a fresh manifest path.
2. **Train:** select training rows from the base prefix and produce float centroids. Sampling affects training only, not indexed population.
3. **Build final router:** construct and save a new router independently of the temporary training graph; retain fitted PCA dimension when applicable.
4. **Build postings:** reroute every base row through that final router. A sequential reader feeds bounded blocks to routing workers. After joining, serial histogram/prefix sums and source-order PQ grouping produce deterministic arrays.
5. **Populate heap:** release posting-build buffers and final router, stream base coordinates into the graph-free heap, checkpoint it, then release locator arrays before final hashing.
6. **Finish:** copy the PQ pivots into the bundle, recheck full source identities, hash component artifacts, and write the completed manifest last.

| Artifacts | Contents |
|---|---|
| `.centroids.bin`, `.hnsw`, `.router` | Centroids, stock graph, settings/identities |
| `.lists`, `.pq.bin` | Grouped IDs/codes and copied codebook |
| `.heap`, `.heap.meta`, `.heap.meta.ivf` | Coordinates, backend metadata, IVF descriptor |
| `.patched.hnsw`, `.refined.hnsw`, `.pca` | Additional optimized-router artifacts, if enabled |
| `.manifest` | Configuration and component digests |

Open checks digests and cross-component shape/settings/population agreement, validates graph framing, and checks posting IDs exist in the heap. Manifest completion is **not atomic publication or an fsync commit**. Failures retain partial artifacts; existing outputs are not overwritten. Checksums are integrity checks, not authentication.

**Read:** `src/bufann/ivf_index.cpp:404–595`; `src/bufann/ivf_lists.cpp:107–306`; `include/bufann/ivf_index.h:31–66`.

**Check:** Why can two shape-valid vector/code files still be incompatible? Their row order or encoding provenance may differ.

## 3. Centroid training and ANN k-means

**Intuition:** repeatedly assign rows to centroids, then move each centroid to the mean of its assigned rows. Approximate assignment avoids comparing every training row with every centroid.

### Shared loop and numerical behavior

The training count is `S = ceil(base_count × sample_rate)`. Seeded reservoir sampling reads and validates the base prefix and retains S float rows and their source IDs. Even a small sample still requires reading the base prefix. `nlist ≤ S` is mandatory.

`sample` initialization chooses distinct sample rows. `random_partition` shuffles rows, distributes them round-robin into balanced nonempty partitions, and initializes each centroid to its partition mean. LV/LV+ require the latter.

Each pass builds/refreshes its assignment index, assigns rows in parallel bounded blocks, joins workers, and serially accumulates double sums/occupancies to produce float means. Empty centroids recover from distinct high-assignment-error rows, breaking ties by source-row ID. This does not guarantee final nonempty clusters.

There are exactly `max_reps` mean-update passes plus a final assignment, so final occupancy and sum of squared errors (SSE) describe returned centroids. No convergence-based early stopping is used. ANN SSE uses chosen centroids; it need not equal the exhaustive nearest-centroid objective or decrease monotonically.

| Mode | Assignment behavior |
|---|---|
| `exact` | Exhaustive double-L2 comparisons |
| `ann-rebuild` | Rebuild stock HNSW every assignment pass |
| `ann-reuse` | Refresh reused full-coordinate patched graph |
| `ann-reuse-seeded` | Reuse plus prior-assignment entry point |
| `ann-reuse-count`, `ann-reuse-seeded-count` | Nearest-one float traversal controls, without/with seeding |
| `lv`, `lv-plus` | LindormVector-inspired optimized trainers: reuse, initial refinement, moving PCA, candidate rerank; LV+ adds seeding |

### What reuse and seeding really do

Refresh snapshots neighbors across all graph levels, replaces **all** centroid coordinates first, then serially reinserts stable IDs using old neighborhoods. Labels/levels persist; full-coordinate distances maintain links. Assignment workers are stopped during refresh, so they never see mixed old/new coordinates. Reuse is connection maintenance, not an unchanged graph.

Seeded modes use the row’s previous centroid ID as a bottom-layer entry, skipping ordinary upper-layer descent. The ID is evaluated against current coordinates/projection. The first pass has no prior assignments and uses ordinary global entry behavior; random partitions do not prepopulate search seeds.

### LV/LV+ schedule and limits

LV/LV+ refine only the initial graph and fit centroid principal component analysis (PCA) at fixed 95% variance. Each later refresh refits PCA, including before final assignment; it does not repeat initial refinement. They return up to `min(nlist, ceil(training_rerank_factor))` candidates and rerank using double-precision distances. Breadth controls exploration separately. Legacy ANN controls rescore breadth-sized sets; count controls return one float winner. The loop also computes the chosen centroid’s double objective distance.

`S × (max_reps + 1)` must fit the default one-billion-search guard; exact mode additionally checks that count × nlist against its distance-work limit. Thus 90M rows × 11 assignments = 990M; 16 assignments exceed the guard. PCA counted operations and refinement traversal have separate guards, not total construction/refresh work limits. Memory budgets estimate allocations, not hard resident-set-size (RSS) ceilings.

**Read:** `src/bufann/ivf_centroids.cpp:48–378`; `src/bufann/ivf_router.cpp:23–40,673–865`; `src/bufann/ivf_patched_graph.cpp:137–172`; `src/bufann/ivf_config.cpp:103–143`.

## 4. Final centroid routing and optimization

**Intuition:** the router is an ANN index whose database is the centroids. It returns promising list IDs, not final vector neighbors. Base assignment and insertion request one centroid; queries request nprobe.

| Router mode | Traversal and selection |
|---|---|
| `exact` | Enumerate all centroids and select by double squared L2 |
| `hnswlib` | Stock HNSW, full-coordinate float traversal, double rescoring |
| `patched-full` | Patched graph, full-coordinate traversal and rescoring |
| `refined` | Refined graph, full-coordinate traversal and rescoring |
| `pca` | Projected traversal, full-coordinate double centroid rerank |
| `pca-refined` | Refined graph plus projected traversal and centroid rerank |

Optimized artifacts must be enabled at build time. Enabling them retains stock HNSW, a patched graph, a separately built/refined graph, and PCA—even if only one mode is selected. Final router construction is independent of the training graph. Selecting LV+ alone does not select optimized final routing.

### Per-operation breadth without shared mutation

The shared graph’s `ef` stays at 1. Stock search requests `min(nlist, max(requested_count, breadth))` candidates; upstream `searchKnn` uses max(ef,k). This supplies query/insertion breadth without racing on `setEf`. Full-coordinate patched modes similarly oversample/rescore. Results sort by full-coordinate distance, then list ID. Requesting every centroid takes the exact enumeration branch.

### PCA: cheaper comparisons, not different full-vector storage

PCA fits the centroid population, not all database rows. It computes a centered double covariance, uses a cyclic double Jacobi eigensolver (at most 64 sweeps), orders eigenvalues, and retains the smallest prefix meeting the variance target. Query projection is `q′ = Uᵀ(q−μ)`; accumulation is double, output float, without whitening. Eigenvector signs/repeated-eigenvalue bases are canonicalized. Zero variance retains all dimensions with identity basis.

The HNSW topology remains full-coordinate-built. Projected coordinates replace the traversal distance space. PCA modes request `C = min(nlist, ceil(requested_count × router_rerank_factor))` centroid candidates, use breadth at least C, and rerank in original coordinates. Requested count is nprobe for queries, one for base assignment/insertion. Keeping 95% variance is not a 95% recall guarantee. Projection cost or limited dimensional reduction may erase its benefit.

### Bottom-layer refinement and connectivity

For each centroid, search/prune neighbors against the frozen pre-refinement graph. Replace bottom outgoing edges only; this replacement adds no reciprocal edges. Make one reinsertion pass over zero-indegree nodes in the proposed graph. Reachability and zero-indegree counts are measured, not guaranteed or repaired until convergence.

> **Three distinct budgets:** training rerank factor controls centroid candidates for one training assignment; router rerank factor controls centroid candidates for final PCA routing; vector rerank factor controls PQ finalists whose full coordinates are fetched. For nprobe=32 and router factor=3, rerank up to 96 centroids. For k=10 and vector factor=3, fetch up to 30 vectors.

**Read:** `src/bufann/ivf_router.cpp:300–452`; `src/bufann/ivf_pca.cpp:32–237`; `src/bufann/ivf_patched_graph.cpp:174–217`; patched headers in `dependencies/hnswlib/include/ivf_patched/`.

## 5. PQ scoring and complete query execution

### Global PQ8 and asymmetric distance computation

**Intuition:** split a vector into chunks and replace each chunk with the byte label of a representative subvector. A query builds a distance lookup table; each code is scored by adding one lookup per chunk.

`PQCodebook<T>` wraps the existing `FixedChunkPQTable<T>`. The codebook is global, with 256 representatives per chunk, and is not an IVF-centroid residual quantizer. Its stored global centering/rearrangement still applies. IVF does not retrain it. Inserts encode against this same frozen codebook.

```text
L[c,b] = query-to-representative squared distance for chunk c
ADC(code) = Σ L[c, code[c]]
immutable budget B = min(N, ceil(k × rerank_factor))
mutable budget B = min(id_capacity−first_id, ceil(k × rerank_factor))
```

Mutable budgeting uses usable ID capacity, not a racing live count; actual finalists cannot exceed observed live postings. Here N is immutable population, k is requested result count, and nprobe is the number of lists selected from nlist total lists.

The ADC table contains `256 × chunks` float entries. With 64 chunks, each code is 64 bytes and each query LUT is 64 KiB. For 90M vectors, code payload alone is 5.76 GB, excluding IDs, locators, router, buffers, and allocator overhead. Equal bytes/vector do not make this PQ8 scanner equivalent to Lindorm’s four-bit SIMD scanner.

### Scalar scan and global top-B

The scanner visits every code in every selected list. It sums float lookup distances and maintains a bounded max-heap across all selected lists, breaking ties by ID. Filling the heap does not stop scanning. Increasing B increases retained candidates, selection work, coordinate fetching, and exact reranking; it does not reduce the number of scanned codes.

### Query stages

1. **Validate/admit:** check dimension, finite coordinates, component compatibility, and conservative workspace allowance.
2. **Route:** find nprobe centroid/list IDs; record routing statistics and time.
3. **Prepare ADC:** build a caller-owned query LUT from frozen PQ pivots.
4. **Scan/select:** inspect selected-list codes and retain up to B candidates.
5. **Fetch:** copy candidate coordinates into bounded, aligned scratch batches.
6. **Exact rerank:** compute double squared L2 against full coordinates, sort by distance then ID, and truncate to k.

Native immutable query requires a clean, checkpointed/read-only heap. Mutable queries use a separate path with live-state checks (section 8). Query scratch is caller-owned; shared PQ/router objects are frozen. The outer driver schedules query concurrency; the single-query core does not become parallel merely because `query_threads` is set.

### Where approximation enters

ANN routing can miss useful centroids; selecting only nprobe lists excludes other vectors; PQ top-B can discard true neighbors. Exact reranking fixes finalist distances, not earlier omissions. Even exact centroid routing remains approximate end-to-end. On a quiescent immutable index, nprobe=nlist **and** B=N yield exhaustive exact scoring; either condition alone is insufficient.

**Read:** `include/bufann/ivf_pq.h`; `include/pq_table.h`; `src/bufann/ivf_lists.cpp:320–368`; `src/bufann/ivf_search.cpp:28–157`; `src/bufann/ivf_config.cpp:146–172`.

**Check:** A true nearest neighbor discarded by PQ cannot be recovered by exact reranking.

## 6. Coordinate heap, batching, and I/O safety

**Intuition:** fetch finalist vectors in groups, but never let concurrent operations pin more cached pages than the pool can safely support.

### Graph-free storage on a reused backend

`HeapStorage<T>` wraps `InPlaceGraphStore` and its buffer pool/libaio path. External IDs are backend node IDs; an ID-to-record-identifier (RID) locator identifies the physical slot. IVF records have degree zero. Reads validate node identity and degree rather than assuming a locator is correct. Logical liveness is separate from retained backend allocation.

Streaming immutable construction and mutable preparation fill pages locally. Generic appends retain round-robin allocation behavior. Filling one page before advancing avoids repeatedly cycling over a 256-page allocation batch with a small default pool. This changes physical placement, not source IDs or query semantics.

### What a batch does

1. Check logical availability under a short ledger mutex.
2. Acquire FIFO aggregate pin-admission credits outside that mutex.
3. Pin requested pages through the backend batch API, which can share/deduplicate page reads.
4. Use one page read guard at a time; validate slot/node identity and recheck logical state.
5. Copy coordinates to caller-owned scratch; release frame pins before admission credits.

Rows start on 64-byte boundaries; scratch stride covers the backend’s padded dimension. No pointer into an evictable page escapes. Duplicate IDs/pages do not recursively acquire a shared page mutex. Backend I/O is not performed under one global adapter ledger lock; coordinate writers are nevertheless serialized.

### Bounded aggregate admission

Let Fmin be the smallest buffer shard’s frame count. Admission capacity is `A = max(1, Fmin−1)`. Query batch size is capped by `min(256, max(1, floor(A/2)))`; writers request one credit. The conservative bound assumes every request could target one shard. FIFO admission, per-page guards, and consistent release order avoid unbounded aggregate pinning even for tiny/skewed pools, at the cost of contention and potentially reduced throughput.

> **One logical fetch stage ≠ one backend call ≠ one page read ≠ one device operation.** Large finalist sets split into bounded calls. Multiple vectors may share a page; cached pages need no pool read. Pool-read counters do not establish device IOPS or cold-storage behavior.

### Failures and instrumentation

The private IVF path validates complete reads/AIO completions and aligned full-page writes. Terminal heap/pool failures poison the session/admission, wake waiters, and refuse future successful checkpointing. Short writes are failures; interrupted operations are retried where implemented. These protections do not imply crash durability or a blanket change to unrelated legacy readers.

Fetch reports backend calls, thread-attributed pool reads, ledger-mutex wait, pin-admission wait, and backend-pin wall time. The three wait/pin intervals are nested within fetch wall time: do not add them again to stage totals or treat them as CPU/device utilization. Pool-wide counter snapshots are individually atomic; use quiescent phase deltas for exact phase accounting.

**Read:** `include/bufann/ivf_heap.h`; `include/bufann/ivf_pin_admission.h`; `src/bufann/ivf_heap.cpp:381–623`; `include/bufann/exact_aio.h`.

**Check:** Four candidates on one resident page can require zero new page reads.

## 7. Mutable lists and publication-safe updates

**Intuition:** insert into an existing section without relearning its boundaries; delete by removing its catalog entry. Full coordinates remain stable once committed.

### Mutable representation and state

Each list owns contiguous ID/PQ arrays, length/capacity, and a reader-writer lock. A bounded locator directory maps ID to owning list, slot, and atomic state. Initial capacity is list population +16; a full list grows by exactly 16 entries. Allocation admission includes old and replacement buffers while both coexist. Deletion does not shrink capacity.

```text
Posting state: Absent → Reserved → Live → Deleted
                        └ cancel → Absent
Heap state:    Absent → Prepared → Live → Deleted
                         └ cancel → Aborted → retry Prepared
```

Successful deletion permanently prevents ID reuse. Canceled unpublished insertion may retry; its retained aborted physical slot is rewritten rather than allocated again. Out-of-range IDs throw. Missing/nonlive deletion returns false and increments the missing-delete counter, including deletion racing with a reserved, unpublished insertion.

### Insertion protocol

1. Acquire shared lifecycle admission; reject failed sessions and invalid coordinates/workspace.
2. Atomically reserve an Absent ID, rejecting live/reserved/deleted IDs.
3. Route to one frozen centroid and encode with the frozen PQ codebook.
4. Prepare physical coordinates under serialized heap-writer admission.
5. Exclusively lock the destination list; grow if full and copy the new ID/code into an unpublished slot.
6. Publish heap logical readiness through a callback, then publish locator/list length/live state. No throwing operations remain after readiness succeeds.

The ordering prevents visible postings referencing unready coordinates. A reserved insertion is not visible merely because bytes exist in a heap page. This is an in-process visibility protocol, not a disk transaction. Centroids, graph topology, PCA, and codebook remain frozen across updates.

### Deletion protocol

Acquire the owning list’s exclusive lock and recheck live state. Move its last ID and PQ code into the removed slot, repair the moved ID’s locator, decrement length, and mark the removed ID Deleted. Then retire its heap record logically. For example, `[41,12,88,9]` deleting 12 becomes `[41,9,88]`. This is constant-entry list work (including a chunk-sized code copy), not coordinate-heap compaction.

### Failure boundaries and tradeoffs

Routing/encoding/list-growth failure can unwind a reservation and cancel prepared coordinates without poisoning the session. Terminal heap-coordinate failures poison it; cancellation failure marks the index failed. If heap retirement fails after posting deletion, the session is poisoned rather than rolled back. Failed sessions reject queries, mutations, and checkpoints. Harness-level partial workloads publish no successful completion, even when some individual updates committed.

Exact +16 growth limits spare capacity but repeatedly copies growing lists; severe insertion skew can make cumulative growth expensive. Frozen centroids avoid retraining cost but may fit shifted data poorly. Balanced insert/delete counts do not bound disk usage: deleted/aborted records remain retained, while newly committed IDs consume storage.

**Read:** `src/bufann/ivf_index.cpp:702–775`; `src/bufann/ivf_posting_lists.cpp:118–338`; `src/bufann/ivf_heap.cpp:381–431`.

**Check:** Explain why retrying an aborted ID is allowed but reinserting a deleted ID is not.

## 8. Consistency, checkpoints, and mixed rounds

### Weakly consistent mutable queries

A scan takes each selected list’s shared lock separately, not a collection-wide lock. It may observe different lists at different times. Reranking checks candidate liveness before fetching, after fetching, and while filtering sorted hits. Deleted candidates are dropped without refilling from the lists, so fewer than k hits may remain. A deletion can still overlap a hit’s final check/return. Late insertions may be missed. These safeguards prevent known stale candidates from being used; they do not provide snapshot isolation.

Mutable workspace admission uses possible ID capacity rather than a racing live count. Ledger/page protection safeguards coordinates. A scan holds one list’s shared lock across its entire scan; growth/deletion require an exclusive lock. Large/skewed lists can delay mutations; no bounded-wait guarantee follows. Ordinary calls share lifecycle admission and can overlap; checkpointing takes it exclusively.

### Fork, checkpoint, reopen

`fork` verifies the immutable source, converts lists, and makes real private copies of heap and PQ sidecar. It does not make a shared cache heap writable. A checkpoint waits for calls to quiesce, verifies no pending reservations and heap/list live/tombstone agreement, flushes, and writes a fresh immutable checkpoint family. The current session remains writable afterward.

Mutable artifacts include `.mlists`, `.heap.state`, and `.origin`, in addition to frozen router/PQ and heap metadata. Reopen copies checkpoint heap/PQ into new private paths, validates states/coordinates/membership and frozen identities, and rechecks source digests. Deleted and aborted physical records survive reopening.

Publication is manifest-last, not atomic/fsynced/crash-safe. No write-ahead log (WAL), rollback, interrupted-run recovery/resume, or slot reclamation is supplied. Flush failure poisons the heap; later output-copy/write failure can leave partial artifacts without necessarily poisoning a healthy session. Preserve failure evidence and use new paths.

### Three actual mixed schedulers

| Mode | Allocation and termination |
|---|---|
| `weighted` | Shared pool; finite streams chosen in proportion to remaining counts. Query percentage r gives floor(r×updates/(100−r)) queries unless explicit count supplied. |
| `fixed_8_8` | Exactly 8 mutation +8 query workers; finite query stream. |
| `update_driven` | floor(T/2) mutation workers, remaining query workers; stop query admission once all updates finish. No query quota. |

Dedicated mutation claims alternate insert/delete while both remain. Dedicated-role modes gate callbacks until all workers launch; weighted workers may execute during launch. Failures stop admission, join existing work, and rethrow without rolling back updates. OpenMP/MKL are constrained before mixed setup. Final `query_threads` controls scoring, not foreground allocation.

Foreground wall time spans launch through join, excluding staging/hash/setup/checkpoint/scoring. Callback latencies exclude dispatch/monitor bookkeeping but include native waits. Overlap measures admitted intervals, not CPU simultaneity. Concurrent queries are unscored; a separate post-join/checkpoint pass scores recall. Update-driven percentiles use at most 65,536 reservoir samples; means use all queries. Tiny update-driven rounds may have no queries; any mode may lack overlap.

**Read:** `src/bufann/ivf_search.cpp:230–380`; `src/bufann/ivf_index.cpp:615–699,803–967`; `tests/ivf_mixed_workload.h`; `tests/ivf_cli_execution.cpp`.

## 9. Cost-model selection and configuration

### Choosing nlist without final-query leakage

**Intuition:** more centroids create finer lists and potentially less PQ scanning, but increase routing and per-list work. The model chooses a candidate nlist minimizing mean `T1 + T2`, where T1 is routing time and T2 is estimated scan/selection time—not complete query latency.

1. Select seeded, disjoint base-derived sample and synthetic-query rows; compute exact double-L2 ground truth against that sample.
2. For each eligible nlist, train the declared trainer and build the final router. Reassign sample rows with the final base/insertion router, not temporary training labels.
3. For each synthetic query’s top-k true neighbors, collect distances to their assigned centroids. Keep multiplicity when neighbors share a centroid.
4. Choose the distance threshold at rank `ceil(target_recall × k)`. Count every centroid at or below the threshold and sum its sample occupancy.
5. Measure production-router time for that probe count. Estimate full-base code count as `sample_codes × N/S`; multiply by a measured per-code coefficient from the actual scalar scanner.
6. Average T1/T2 across queries and choose minimum total, breaking exact ties toward smaller nlist. Retain exclusions, settings, identities, and report provenance.

Threshold code counts use ideal centroid distances; approximate routing may select a different set. The tool records routed sample counts separately. ADC-table setup, coordinate fetch, and exact vector rerank are excluded from the objective. Thus the selected nlist need not maximize end-to-end QPS, nor does a probe estimate guarantee achieved ANN recall.

Default eligibility requires `S ≥ nlist × 32`: configurable project policy, not a paper mandate or adequacy proof. The coefficient profiles single-list scan/selection on an existing-code prefix and extrapolates linearly, not a complete multi-list query. Freeze source/PQ/settings/report before validation and reserve final queries. Prediction error is not permission to refit on the final set. The harness checks report/explicit-nlist agreement and records provenance.

### Defaults depend on entry point

| Setting | Current default or distinction |
|---|---|
| Programmatic trainer/init | `AnnRebuild / Sample` |
| Native CLI/harness trainer/init | `ann-reuse-seeded / random_partition`; examples explicitly choose LV+ |
| Training | sample rate 1, 15 passes, breadth 64; LV training rerank factor 3, refinement breadth 128 |
| Library/native query | nprobe 1, k=10, vector factor 2, one query thread; calibration vector factor defaults to 4 |
| Final router | hnswlib; degree 16, construction breadth 200, query breadth 128, insert breadth 64 |
| Mixed | Native: weighted, 16 workers, ratio 80; shell/Python: update_driven, 16 workers |
| Runtime / optional policy | 64 frames, 1 GiB allowance; standalone insertion: one worker/streaming; warmup zero. Mixed input is staged. |

Large-data recipes override these values; example settings are not defaults or qualified optima. Optimized final modes must be enabled separately. Library page-size admission is broader than Python harness admission; use entry-point-specific validation. Memory allowances guard estimated retained objects/scratch, not a hard process-RSS ceiling.

**Read:** `src/bufann/ivf_cost_model.cpp:31–172`; `tests/ivf_cost_model.cpp:280–403`; `include/bufann/ivf_config.h`; `include/bufann/ivf_router.h`; `benchmark/IVF/settings.py`; `benchmark/IVF/mixed.py`.

## 10. Verification, measurement, and study checklist

### Operational and measurement boundaries

Full heap-backed execution requires the Linux/native dependency-enabled build (MKL, OpenMP, libaio). Dependency-isolated drivers validate input contracts but do not execute native workloads. Use `benchmark/IVF/build.sh` and the current benchmark README for installed-library paths; no DiskANN graph conversion is needed.

Prepare/run scripts cover independent query, insert, delete, and mixed workloads. Source/settings/tool identities bind reusable cached bundles. Cache restoration is not a new build measurement. Fresh work paths/tags preserve prior and failed evidence. Checkpoint/reopen copies can substantially increase disk requirements.

- **Query timing:** excludes preparation/open/checks/scoring/export; lifetime RSS does not. Stage totals sum per-query wall times, not run wall time, and can exceed elapsed time with concurrency. Run time includes launch/join/result collection. Hashing/open/copies warm caches; zero warmup does not mean cold devices.
- **Insertion timing:** streaming includes worker source reads; staged input is admitted/loaded before mutation timing. Keep these policies separate when comparing worker counts. Heap work/publication and launch/join remain foreground costs.
- **Warmup:** optional serial queries derived from frozen base centroids run on the same index object before the final timed pass, after updates/checkpoint when applicable. They use no final-query/GT selection and are not scored. A separate-process warmup cannot transfer its application buffer pool.
- **Recall:** score against the actual live population after updates. Filtered deep ground truth (GT) must contain enough survivors; underflow fails. Native scoring is percentage recall with kth-distance tie expansion where available; truncated derived GT limits tie coverage. Missing hits count against recall.
- **Comparison:** post-round queries/second (QPS) is not concurrent QPS. Mixed `summary.json` references round `result.json` files and retains summed foreground wall/counts, not pooled quantiles or a duplicate root result. Record threads/cache, identities, storage/runtime, overlap, memory/RSS, repeats, and failures. Unequal-recall points do not prove matched-recall speedups.

### Tests worth reading as executable specifications

| Under `tests/ivf/` | Evidence to inspect |
|---|---|
| `ivf_centroid_tests.cpp`, `ivf_ann_training_tests.cpp`, `ivf_patched_router_tests.cpp` | Training schedules, candidate contracts, routing/projection |
| `ivf_list_tests.cpp`, `ivf_search_tests.cpp` | Grouping, independent ADC/exact oracle, full-budget exactness, page-sharing/batch counts |
| `ivf_mutable_list_tests.cpp`, `ivf_mutable_index_tests.cpp` | Growth, reservations, stale candidates, retries, checkpoint/reopen |
| `ivf_concurrent_fetch_tests.cpp`, `ivf_direct_io_tests.cpp`, `ivf_read_io_tests.cpp` | Bounded access, alignment, short/failing I/O |
| `ivf_cost_model_tests.cpp`, scheduler and Python integration tests | Prediction components, failure joining, orchestration contracts |

---

## Addendum A. Evidence and file-by-file activities

### A.1 What “evidence” means

Evidence labels distinguish four different kinds of support; they are not interchangeable:

| Evidence type | What it supports | What it does not establish |
|---|---|---|
| **Implementation inspection** | Actual branches, arithmetic, lock scopes, publication order, framing checks, and failure paths in the reviewed source | That every path executes correctly on every platform or is race-free |
| **Test-source inspection** | The properties a fixture asserts, its oracle, inputs, and injected failures | That the test passed on this checkout; a test name alone is not execution evidence |
| **Historical reports/documentation** | Recorded intent, prior runs, operational procedures, and known caveats | Current successful execution, current performance, or code fidelity when notes have drifted |
| **Derived reasoning** | Consequences of explicit mechanisms: e.g., no refill can produce fewer than k hits; +16 growth repeatedly copies arrays | Measured latency, memory, recall, scaling, or a formal correctness proof |

This guide and addendum are **static-review artifacts**. Neither records a new test/build run or reproduces historical performance numbers. Line ranges are navigation aids for this snapshot, not stable version identifiers; prefer the named classes/functions after edits. For reproducible evidence, record a revision or source digest plus binary/dependency identities, input/configuration hashes, commands, machine/runtime settings, logs, and exit status.

### A.2 Claim-to-evidence map

| Claim | Implementation evidence | Relevant fixture or limitation |
|---|---|---|
| Build consumes existing PQ and assigns all base rows independently of the training sample | `NativeIndex<T>::build`; `PostingLists::build<T>` | `ivf_search_tests.cpp`; source identities do not prove external code/vector parity |
| LV+ differs from LV by previous-assignment seeding | Trainer-mode helpers; reusable assignment policy; patched HNSW search entry | `ivf_ann_training_tests.cpp`; no paper-equivalent speedup follows |
| Training performs a final assignment after fixed mean-update passes | Shared loop in `ivf_centroids.cpp` | `ivf_centroid_tests.cpp`; not convergence-based training |
| ADC scans every code in selected lists but retains only B candidates | Immutable and mutable `scan` loops | `ivf_list_tests.cpp`, `ivf_search_tests.cpp`; finalist truncation remains approximate |
| Full-budget, all-list immutable search agrees with brute force | Query exact-distance loop and deterministic ordering | `ivf_search_tests.cpp:129–161` compares with an independent coordinate oracle |
| One logical fetch may produce multiple calls; same-page vectors can share reads | `copy_batch_limit`, `copy_coordinates_batch` | `ivf_search_tests.cpp:155–215`; counters are not device IOPS |
| Postings are not published before prepared heap readiness | `MutableIndex<T>::insert`; reservation `commit_ready`/`publish_reserved` | `ivf_mutable_index_tests.cpp`; in-process publication, not durable commit |
| Deleted IDs cannot be reused; aborted unpublished IDs may retry | Posting state transitions and heap mutable ledger | `ivf_mutable_list_tests.cpp`, `ivf_mutable_index_tests.cpp`; physical records remain retained |
| Queries are weakly consistent, with stale filtering and no refill | Per-list scan locks; `rerank_mutable_candidates` membership checks | `ivf_mutable_index_tests.cpp`; no collection snapshot or guaranteed k results |
| Checkpoint excludes active ordinary calls and preserves retained state | Lifecycle exclusive lock; checkpoint membership validation; mutable-copy persistence | Mutable-index and I/O fixtures; no crash recovery/fsync guarantee |
| Scheduler startup semantics differ by mode | Dedicated start gate versus weighted pool launch loop | Scheduler fixtures; weighted launch failure can follow already-applied updates |
| Predictor minimizes routing plus estimated scan, not full latency | `estimate_probes`, `choose_nlist`, calibration timing/coefficient code | `ivf_cost_model_tests.cpp`; prediction quality needs frozen real-data validation |

Fixture paths in this table are under `tests/ivf/`. A fixture must be built and executed successfully before citing it as a passing result.

### A.3 Core implementation: file-by-file activities

“Activities” below means **what each file does during operation and what to inspect in it**.

| File(s) | Operational activity and review focus |
|---|---|
| `include/bufann/ivf_index.h`; `src/bufann/ivf_index.cpp` | Compose complete native/mutable indexes; own build/open/fork/reopen/query/update/checkpoint lifecycles; validate manifests and frozen-component agreement. Start here for cross-component ordering and failures. |
| `include/bufann/ivf_config.h`; `src/bufann/ivf_config.cpp` | Define/parse trainer and dtype policies; validate training/search settings; calculate training sample and finalist budgets. Distinguish programmatic defaults from CLI policy. |
| `include/bufann/ivf_centroids.h`; `src/bufann/ivf_centroids.cpp` | Sample base rows, initialize means, dispatch assignments, accumulate sums/occupancy/SSE, recover empty clusters, and perform final assignment. Check counters separately from total work. |
| `src/bufann/ivf_centroids_internal.h` | Define the internal assignment-index/factory interface and measurements connecting the shared training loop to alternative assignment implementations. |
| `include/bufann/ivf_router.h`; `src/bufann/ivf_router.cpp` | Build/load centroid indexes, select routing modes, enforce query/insert breadth, project/rerank candidates, and implement ANN training adapters and refresh policy. |
| `include/bufann/ivf_patched_graph.h`; `src/bufann/ivf_patched_graph.cpp` | Adapt modified HNSW; measure directed reachability/indegree; refresh coordinates and neighborhoods; propose/replace bottom edges and reinsert zero-indegree nodes. |
| `dependencies/hnswlib/include/ivf_patched/hnswalg.h` | Implement the actual traversal hooks, seeded bottom-layer entry, coordinate replacement, and reinsertion/edge-update primitives beneath the wrapper. Read this when wrapper-level descriptions are insufficient. |
| `include/bufann/ivf_pca.h`; `src/bufann/ivf_pca.cpp` | Fit/project/serialize/validate centroid PCA; count bounded covariance/solver work; handle eigenvalue degeneracy and retained-dimension selection. |
| `include/bufann/ivf_lists.h`; `src/bufann/ivf_lists.cpp` | Route all base rows; build grouped immutable arrays; save/load and validate framing; scan ADC codes and profile scalar scan/selection. |
| `src/bufann/ivf_lists_internal.h` | Coordinate bounded block routing and joined worker phases during list construction. Histogram and grouping remain outside the parallel routing phase. |
| `include/bufann/ivf_posting_lists.h`; `src/bufann/ivf_posting_lists.cpp` | Own mutable arrays, locator states, reservations, exact +16 growth, swap deletion, list locking, mutable scan, and checkpoint reconstruction. |
| `include/bufann/ivf_pq.h`; `include/pq_table.h` | Wrap/load the frozen PQ codebook; encode insertion vectors and build ADC LUTs; implement underlying centering/rearrangement and chunk quantization. |
| `include/bufann/ivf_search.h`; `src/bufann/ivf_search.cpp` | Estimate query scratch; sequence routing/LUT/scan/fetch/rerank; sort exact hits; collect stage metrics; filter stale mutable candidates without refilling. |
| `include/bufann/ivf_heap.h`; `src/bufann/ivf_heap.cpp` | Adapt graph-free typed records; serialize coordinate preparation; manage mutable states, page-local population, bounded coordinate copies, poisoning, and immutable/mutable heap checkpoint formats. |
| `include/bufann/ivf_pin_admission.h` | Implement FIFO aggregate credit admission, capacity bounds, terminal failure propagation, and waiter wakeup. Check resource acquisition/release order with heap code. |
| `include/bufann/ivf_fetch_timing.h` | Attribute ledger-lock and fetch subinterval timing without treating nested intervals as separate query stages. |
| `include/bufann/inplace_backend.h`; `src/bufann/inplace_backend.cpp` | Provide reused physical slot/page layout, locators, buffer frames, page guards, batch pinning, allocation, flushing, and low-level I/O. Only graph-free IVF usage is in this guide’s scope. |
| `include/bufann/exact_aio.h` | Validate complete asynchronous read submission/completion in the scoped IVF backend path; inspect error/short-completion handling rather than inferring it from ordinary read APIs. |
| `include/bufann/ivf_cost_model.h`; `src/bufann/ivf_cost_model.cpp` | Implement disjoint calibration splits, exact sample GT, multiplicity-preserving probe thresholds, scaled code counts, and candidate selection by T1+T2. |

### A.4 Input, CLI, harness, and build-system files

The files named `tests/ivf_build.cpp` and `tests/ivf_driver.cpp` are operational command-line tools.

| File(s) | Operational activity and review focus |
|---|---|
| `src/bufann/ivf_artifacts.cpp` | Exclusively claim fresh artifact paths; compute whole-file/prefix FNV-1a hashes and byte-count/hash digests. Distinguish native integrity digests from Python SHA-256 cache identities and from authentication. |
| `src/bufann/ivf_io.cpp` | Validate vector/PQ framing, sizes/overflow, pivot offsets/finite values/permutations/chunk boundaries, and bounded unique counted IDs. GT shape inspection alone does not validate tag identity or distance ordering; the native execution layer performs additional content checks. |
| `tests/ivf_build.cpp`; `tests/ivf_driver.cpp`; `tests/ivf_cli.h` | Parse build/workload options, enforce input/metric contracts, and dispatch validation-only versus compiled native execution. Do not mistake an input-validation success for index construction or workload execution. |
| `tests/ivf_cli_execution.cpp` | Implement typed native workflows, composed memory admission, private copies, identity checks, mutations/checkpoints, warmup and parallel final queries; validate/score GT, export neighbors, and emit timing/resource JSON. |
| `tests/ivf_insert_input.h`; `tests/ivf_insert_workload.h`; `tests/ivf_query_warmup.h` | Implement insertion input-policy/worker coordination and centroid-derived warmup. Distinguish staged payload memory and source-read timing from native mutation cost. |
| `tests/ivf_mixed_workload.h`; `tests/weighted_update_pool.h` | Dispatch dedicated-role or weighted streams; account for callback failures, joins, completion/overlap, and latency storage. Inspect the different startup admission behaviors directly. |
| `tests/ivf_cost_model.cpp`; `tests/ivf_router_diagnostics.cpp` | Compose calibration sampling/training/router timing/scanner profiling and fixed-router measurements, respectively. Separate centroid recall from end-to-end ANN recall. |
| `benchmark/IVF/harness.py` | Prepare/restore SHA-256-keyed immutable caches using real copies; verify prepared identities; run query or mutation-then-checkpoint-query processes while preserving native query metrics and execution provenance. |
| `benchmark/IVF/mixed.py` | Validate equal-cap round plans and per-round ID/GT families; chain native processes through checkpoints; validate driver-written results/live counts; publish aggregates with relative round-result paths. |
| `benchmark/IVF/settings.py`; `benchmark/IVF/common.py` | Own defaults and build/query policy separation, input/resource guards, calibration-report compatibility, identities, exclusive output/log creation, command streaming, affinity/timeouts, and runtime provenance. |
| `benchmark/IVF/cost_model.py`; `benchmark/IVF/router_sweep.py`; `benchmark/IVF/qualify.py` | Freeze calibration provenance without official-query inputs; sweep routers on existing centroids; orchestrate trainer/query comparisons and declared gates. Successful orchestration is not production qualification. |
| `benchmark/IVF/download_sift.py` | Download validated SIFT base prefixes/public queries; rewrite prefix row headers; reject overwrite/resume and retain partial output. Matching download shape does not replace downstream PQ/GT identity checks. |
| `cmake/IVF.cmake`; `cmake/Hnswlib.cmake` | Define dependency-light C++20 targets and public no-fast-math numeric policy; optionally add routing/list targets, vendored modified headers, and a pinned stock upstream comparison. |
| `src/CMakeLists.txt`; `tests/ivf/tools.cmake`; `tests/ivf/CMakeLists.txt` | Compose backend/search/index targets, gate native CLI execution and diagnostics by available targets/platform, and register contract/integration fixtures. Inspect target gates before assuming a binary can execute heap workloads. |
| `benchmark/IVF/README.md`; `docs/s02-heap-adapter.md` through milestone-specific notes | Explain operational procedures, historical rationale, and recorded validation. Cross-check against current code; old scheduler, patch-generation, and serialization descriptions can lag implementation. |
