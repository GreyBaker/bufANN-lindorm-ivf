# Lindorm-style IVF: architectural summary of the committed implementation

## Scope and overall assessment

**The implementation is a single-node, mutable, disk-backed IVF-PQ engine inspired by Lindorm's ANN core—not a reproduction of its distributed database.** It preserves the central separation between memory-resident compressed search and a bounded final fetch of full coordinates. It implements ANN-accelerated centroid training, graph reuse and assignment seeding, PCA-assisted centroid traversal, bottom-layer graph refinement, real-time posting-list changes, and a centroid-count cost model. It supplies its own local storage, concurrency, persistence, and experimental infrastructure where Lindorm relies on its database platform.

The most consequential departures are:

1. **Local execution and storage:** no LDMaster/LDServer cluster, LindormTable, LindormDFS, region migration, replicated storage, or network candidate aggregation.
2. **Global, non-residual PQ8 with a scalar scanner:** not the paper's evaluated four-bit SIMD scanning configuration; RaBitQ is absent.
3. **Explicit interpretations of underspecified training details:** fixed iteration counts, deterministic sampling/reduction, initial-only refinement, and PCA refitting between assignments.
4. **A controlled HNSW implementation and ablation suite:** checksum-pinned stock hnswlib plus a separately maintained modified variant, rather than the paper's Faiss comparison.
5. **In-process publication safety and clean checkpoints, not WAL-backed commits:** update visibility is carefully ordered, but there is no crash recovery or durable transaction protocol.
6. **Engineering beyond description:** bounded buffer-pin admission, page-local allocation, failure poisoning, artifact identity checks, resource guards, and explicit measurement contracts.

## 1. System boundary and component architecture

The principal composition is:

```text
NativeIndex<T>
  ├─ CentroidRouter       full centroids + centroid HNSW/PCA artifacts
  ├─ PostingLists         memory-resident grouped vector IDs and PQ codes
  ├─ PQCodebook<T>        frozen global quantizer
  └─ HeapStorage<T>       disk coordinates + buffer pool + ID-to-slot mapping

MutableIndex<T>
  ├─ frozen router and PQ
  ├─ MutablePostingLists  per-list locks, capacity and ID states
  ├─ private writable coordinate heap and logical-state ledger
  └─ lifecycle admission, checkpoint composition and failure state
```

The implementation uses C++ templates for float32, int8, and uint8 coordinates. Centroids are float32 even when database coordinates are integers. The supported index metric is squared Euclidean distance; the presence of other metrics elsewhere in the inherited repository does not make them IVF features.

There is **no graph over database vectors** on the IVF path. HNSW exists only over the much smaller centroid population. The coordinate heap reuses a graph-capable backend but sets database-record graph degree to zero. Thus “IVF-only” means independence from DiskANN/Vamana data-graph construction, not absence of all graph algorithms.

**Relation to paper §§3.2–3.4:** the logical search pipeline is preserved, but the distributed coordinator/worker stages collapse into one process. One centroid/codebook family is shared across this index's lists; this is not distributed metadata sharing, cross-table deduplication, or an elasticity mechanism. There is no simulated network or distributed latency component.

**Sources:** [`ivf_index.h`](../include/bufann/ivf_index.h), [`ivf_index.cpp`](../src/bufann/ivf_index.cpp), [`ivf_search.cpp`](../src/bufann/ivf_search.cpp), [`ivf_heap.h`](../include/bufann/ivf_heap.h).

## 2. Identity, input contracts, and physical/logical layout

### Stable source-row identity

Vector inputs use an eight-byte header containing uint32 row count and dimension, followed by packed coordinates. Dtype is supplied separately. The first `base_count` rows are the initial database; later rows can supply insertions. The benchmark preserves source-row IDs across vectors, PQ codes, ground truth, and mutations. The library additionally permits `external_id = first_id + source_row`.

IDs belong to a bounded, predeclared capacity. They are also the storage backend's node IDs, but **not physical slot numbers**: an ID-to-RID locator identifies the page and slot. Several arrays are capacity-sized, so sparse, very large external ID spaces are not free. An inserted vector is assigned to one list, not replicated across adjacent clusters.

This substitutes a simple identity contract for Lindorm's arbitrary database row/routing keys and region ownership. Physical coordinates need not be laid out in centroid order.

### Two posting representations

* **Immutable:** CSR-like `offsets[nlist+1]`, grouped `ids[N]`, and `codes[N × chunks]`. Each list is a contiguous range; IDs inside a list preserve source order.
* **Mutable:** each list owns separate contiguous ID/code arrays, a length, capacity, and reader-writer mutex. A locator directory maps each external ID to list, slot, and atomic lifecycle state.

Separating these representations gives compact immutable bundles and localized mutable changes without pretending the same storage format serves both purposes optimally.

### PQ is an external, frozen input

Native IVF build loads existing pivots and codes; it does **not** train PQ itself. Codes must cover the entire supplied source population, including reserved insertion rows, and match the pivot chunk count. Standalone helpers can produce those inputs, and the documented full-source recipe may train PQ on rows reserved for later insertion. That is distinct from centroid training, which uses only the initial base prefix.

Framing, sizes, finite values, and identities are checked, but the builder does not independently prove that every supplied code encodes its corresponding vector or that the codebook's training provenance is appropriate. Shape-valid but differently ordered inputs can still be semantically incompatible. The manifest explicitly identifies the policy as `global_nonresidual_pq8_external_unverified`.

**Sources:** [`ivf_io.cpp`](../src/bufann/ivf_io.cpp), [`ivf_lists.cpp`](../src/bufann/ivf_lists.cpp), [`ivf_posting_lists.cpp`](../src/bufann/ivf_posting_lists.cpp), [`ivf_pq.h`](../include/bufann/ivf_pq.h).

## 3. Build architecture: train once, reroute all rows, stream coordinates

The committed build has deliberately separate phases:

1. Validate configuration, input shapes, output collisions, and relevant resource limits; hash source vectors, pivots, and codes.
2. Exclusively claim a fresh manifest path, reserving the output family.
3. Train centroids from the configured sample of the base prefix.
4. Construct and persist a **new final router**, independent of the temporary training graph/PCA.
5. Route **every base vector** through that final base/insertion router and group its ID and supplied PQ code.
6. Release completed posting-build buffers and the final router; stream original coordinates into the heap.
7. Checkpoint and release heap metadata, copy the PQ pivots, recheck source identities, hash all components, and write the completed manifest last.

Sampling reduces training population, not indexed population. Likewise, temporary training assignments are not reused as final list labels: different training/final search policies can choose different centroids, so final routing is an explicit correctness boundary.

### Parallelism is selective and bounded

Training assignments run in parallel blocks. Final list routing uses a single sequential source reader feeding bounded waves to persistent routing workers. Workers write independent label slots. Histogram construction, prefix sums, and source-order PQ grouping occur serially after routing joins. This avoids concurrent offset writes and preserves deterministic grouped output independent of worker completion order.

Heap construction remains serial. Releasing the grouped lists before coordinate population is material at scale: 90M PQ64 vectors require about 6.12 GB just for codes plus uint32 IDs. Worker buffers, routing scratch, and stack allowances are included in preflight estimates; `build_threads` does not parallelize every phase.

These are implementation-specific build/memory choices beyond the paper's algorithmic outline, not a distributed training implementation.

**Sources:** `NativeIndex<T>::build` in [`ivf_index.cpp`](../src/bufann/ivf_index.cpp), [`ivf_lists_internal.h`](../src/bufann/ivf_lists_internal.h), [`ivf_lists.cpp`](../src/bufann/ivf_lists.cpp).

## 4. Centroid learning: deterministic Lloyd loop with interchangeable assignment engines

### Sampling, initialization, and numerical policy

For base population `N`, training uses `S = ceil(N × sample_rate)` rows, selected by seeded reservoir sampling. The sampler still reads and validates the entire base prefix. It retains sampled source IDs for diagnostics and deterministic tie handling. `nlist ≤ S` is required.

Two initialization policies exist:

* `sample`: distinct sampled rows become initial centroids.
* `random_partition`: shuffle sample rows, distribute them round-robin into balanced nonempty initial partitions, and use partition means. LV/LV+ require this policy.

The second policy concretizes the paper's random-partition initialization; it is specifically balanced rather than unconstrained independent random assignment. Initial partition membership does not seed the first ANN assignment.

Assignments run in parallel, but means and objective totals use ordered serial reductions. Coordinate sums and exact squared distances accumulate in double; new means are stored as float. Empty clusters are reset to distinct rows with the greatest assignment error, breaking ties by original source-row ID. This deterministic recovery rule is additional implementation detail; it does not ensure every returned cluster is nonempty.

### Fixed passes, not convergence stopping

The code executes exactly `max_reps` assignment/mean-update passes followed by one final assignment. The last assignment ensures returned occupancy and SSE refer to the returned centroids. There is no convergence threshold or early termination.

This differs from the paper's “until convergence” description (§4.2). Approximate assignments also mean reported SSE is the error to the **chosen** centroid, not necessarily the exhaustive nearest-centroid objective, and it need not decrease monotonically.

### Assignment modes and what each isolates

| Trainer | Committed assignment behavior |
|---|---|
| `exact` | Exhaustive double-L2 nearest centroid. |
| `ann-rebuild` | Rebuild stock full-coordinate HNSW for every assignment pass. |
| `ann-reuse` | Refresh a maintained full-coordinate modified graph; double-rescore a breadth-sized candidate set. |
| `ann-reuse-seeded` | Same reuse, with the previous assignment as a bottom-layer entry. |
| `ann-reuse-count` | Full-coordinate traversal returning one float-distance winner; exploration breadth remains separate. |
| `ann-reuse-seeded-count` | Same nearest-one control with previous-assignment seeding. |
| `lv` | Reuse, initial graph refinement, PCA traversal, and original-coordinate candidate reranking. |
| `lv-plus` | Identical to LV except for previous-assignment seeding. |

The count controls matter: the older reuse controls return and double-rescore up to the traversal breadth, whereas LV requests `min(nlist, ceil(training_rerank_factor))` centroid candidates for nearest-one assignment. Without separating exploration from returned candidates, a comparison would confound graph optimization with a different reranking workload.

### Reuse means active graph maintenance

At a joined assignment boundary, refresh:

1. snapshots old neighbors across all levels;
2. replaces **all** centroid coordinates;
3. serially reinserts every stable centroid ID using old neighborhoods as entry information;
4. measures resulting directed reachability and zero indegree.

Graph levels and labels persist, but links are maintained through update search, pruning, and reciprocal connection logic. This is neither rebuilding from scratch nor retaining an unchanged adjacency graph. Old neighbors are selected at eligible levels with global-entry fallback where necessary. No assignment worker sees mixed old/new coordinates.

For seeded training searches, the previous centroid is evaluated at its **current** coordinates and used as a bottom-layer entry, skipping normal upper-layer descent. The first pass has no prior assignment and falls back to ordinary entry behavior. The implementation remembers one previous nearest centroid per row, a concrete choice within the paper's broader “nearest centroids” entry-point description.

### LV/LV+ resolve a schedule the paper does not fully prescribe

LV modes refine the initial graph once, fit PCA to retain 95% centroid variance, and refit PCA after each subsequent graph refresh, including before final assignment. They do not repeat the initial frozen-graph refinement every iteration. Graph maintenance always uses full coordinates; only assignment traversal uses projected coordinates. Candidate selection is followed by original-coordinate double reranking and a separately counted objective-distance evaluation.

This moving-PCA/initial-refinement schedule is an explicit implementation interpretation, not a claim that Alibaba used precisely that schedule. Training PCA's 95% target is fixed independently of the configurable final-router variance target. Temporary training graphs and projections are discarded, not persisted as the final search router.

**Sources:** [`ivf_centroids.cpp`](../src/bufann/ivf_centroids.cpp), `training_search_policy`, `make_reusable_assignment`, and `train_centroids_ann` in [`ivf_router.cpp`](../src/bufann/ivf_router.cpp), [`ivf_patched_graph.cpp`](../src/bufann/ivf_patched_graph.cpp), [`ivf_config.cpp`](../src/bufann/ivf_config.cpp).

## 5. Centroid routing: separately selectable graph, refinement, and compression

The final router returns list IDs. Base assignment and insertion request one centroid; a query requests `nprobe`. It is independently configured from the trainer.

| Router | Traversal | Final centroid ranking |
|---|---|---|
| `exact` | Enumerate all centroids | Original-coordinate double L2. |
| `hnswlib` | Stock HNSW, full-coordinate float kernels | Double-rescore returned candidates. |
| `patched-full` | Modified HNSW, full coordinates | Double-rescore candidates. |
| `refined` | Modified graph after bottom-layer refinement | Double-rescore candidates. |
| `pca` | Modified graph, projected centroid coordinates | Original-coordinate double rerank. |
| `pca-refined` | Refined graph, projected coordinates | Original-coordinate double rerank. |

Selecting LV+ training **does not enable or select** PCA/refined final routing. Optimized final artifacts must be enabled during build; query mode can then be overridden independently of the persisted base/insertion mode. This separation supports useful ablations but also permits a configuration substantially less optimized than the paper's full pipeline.

### Per-call breadth instead of mutating shared HNSW settings

Shared graphs retain `ef = 1`. For stock HNSW, the requested candidate count is `min(nlist, max(requested_count, breadth))`; upstream's `max(ef, k)` behavior supplies effective breadth without concurrent `setEf` calls. Full-coordinate modified modes use similar oversampling and double rescoring. Projected modes request:

```text
C = min(nlist, ceil(requested_count × router_rerank_factor))
effective breadth ≥ C
```

They traverse in projected space, rerank those C centroids in original space, and retain the requested number. Ordering is distance then list ID. Requesting all centroids takes exact enumeration rather than depending on graph connectivity.

This per-call design is important for overlapping queries and insertions with different breadths. It also means the `hnswlib` mode is an unmodified **dependency**, but the surrounding router adds candidate oversampling and double rescoring; it is not simply raw upstream `searchKnn(query, nprobe)` timing.

### PCA implementation choices

PCA fits the centroid population, not database vectors. The implementation computes a centered double covariance matrix, solves it with a cyclic double Jacobi eigensolver capped at 64 sweeps, and retains the smallest leading-eigenvalue prefix meeting the variance target. Projection uses double accumulation and float output, with no whitening. Eigenvector signs and repeated-eigenvalue bases are canonicalized; zero variance uses the full-dimensional identity basis. Load validation checks the projection against the source centroids.

The graph topology is built in full-coordinate space and then searched using a separate projected coordinate array. **The implementation does not eliminate full-coordinate centroid storage.** With optimized modes enabled it retains a stock graph, a modified unrefined graph, a separately constructed/refined graph, full centroids, and PCA data. That is valuable for comparisons but costs more memory than a production representation retaining only the chosen graph. PCA here reduces traversal-coordinate traffic, not necessarily the total router memory footprint.

Keeping 95% variance is not a 95% recall guarantee, and projection overhead can outweigh reduced-distance cost when dimensional reduction is small.

### Refinement and connectivity are bounded heuristics

For every centroid, refinement searches the frozen pre-refinement graph, applies the standard neighbor-pruning heuristic, and records proposed bottom-layer neighbors. All proposals are then installed. This replacement changes outgoing bottom edges only and adds no reciprocal links by itself. Upper layers remain intact.

The implementation makes one reinsertion pass over nodes with zero indegree in the proposed graph. It measures directed bottom-layer reachability from the entry point and zero-indegree counts afterward. It does **not** enforce strong connectivity, compute an SCC repair closure, or repeat repair until connected. This realizes the paper's isolated-node reinsertion idea (§4.3), but does not turn the paper's empirical connectivity observation into a guarantee.

**Sources:** [`ivf_router.h`](../include/bufann/ivf_router.h), [`ivf_router.cpp`](../src/bufann/ivf_router.cpp), [`ivf_pca.cpp`](../src/bufann/ivf_pca.cpp), [`ivf_patched_graph.cpp`](../src/bufann/ivf_patched_graph.cpp).

## 6. HNSW dependency and comparison architecture

The paper evaluates against Faiss HNSW. This repository instead uses:

* a checksum-pinned, fetched, **unmodified hnswlib v0.8.0** comparison;
* a checked-in modified variant under `dependencies/hnswlib/include/ivf_patched/`, isolated in namespace `ivf_hnswlib`.

Modified headers are authoritative source, not generated patches. They add per-call projected traversal coordinates and counters, caller-provided entries, seeded bottom-layer search, quiescent coordinate replacement/reinsertion, and validated bottom-edge replacement. They retain upstream queues, stopping rules, visited pools, pruning, and SIMD distance kernels rather than introducing a new graph-search algorithm. Local lookahead/empty-neighbor safety fixes are also retained.

Artifacts accept the producer identity `hnswlib-0.8.0-stock1`; older producer labels are deliberately rejected rather than relabeled. This records dependency identity separately from binary framing versions. The label identifies the stock comparison, not a claim that optimized graphs are unmodified.

There is a qualification caveat: pristine upstream retains known SIMD-prefetch safety issues. Sanitizer configurations default to disabling manual HNSW vectorization for both variants. Scalar sanitizer results therefore do not qualify the SIMD path, and old safety-corrected comparison measurements cannot be relabeled as new stock measurements.

**Sources:** [`dependencies/hnswlib/README.md`](../dependencies/hnswlib/README.md), [`cmake/Hnswlib.cmake`](../cmake/Hnswlib.cmake), modified [`hnswalg.h`](../dependencies/hnswlib/include/ivf_patched/hnswalg.h).

## 7. Quantization, candidate selection, and complete query execution

### Deliberate PQ8/scalar specialization

`PQCodebook<T>` wraps the inherited `FixedChunkPQTable<T>`: 256 representatives per chunk, one byte per chunk, with its stored global centering/rearrangement. Codes describe global vectors, **not residuals relative to IVF centroids**. Insertions encode with that same frozen codebook.

For each query, a caller-owned float LUT contains `256 × chunks` distances. A code's asymmetric distance is the ordered float sum of one lookup per chunk. The scanner visits every code in every selected list and retains a global top-B max-heap, with vector-ID tie breaking.

The paper supports PQ/RaBitQ and evaluates PQ with four-bit codes and SIMD fast scanning (§§3.3, 4.1, 5.3). Neither RaBitQ nor a four-bit batch-transposed FastScan kernel is implemented here. For example, 64 PQ8 chunks and 128 PQ4 chunks both occupy 64 bytes/vector, but have different subspaces, lookup tables, approximation error, and execution cost. Equal byte budgets do not establish algorithmic or performance equivalence.

### Three distinct candidate budgets

| Budget | What it controls |
|---|---|
| Training rerank factor | Number of centroid candidates reranked for one LV/LV+ training assignment. |
| Router rerank factor | Number of centroid candidates reranked during projected final routing; proportional to nprobe for queries. |
| Vector rerank factor | Number of PQ finalists whose full database coordinates are fetched; proportional to requested k. |

For immutable population N, `B = min(N, ceil(k × vector_rerank_factor))`. Mutable query admission uses usable **ID capacity** instead of a changing live count, avoiding underestimating scratch during concurrent insertion. Actual returned candidates remain bounded by observed postings.

Larger B increases selection/fetch/exact-rerank work. It does not reduce scanned codes, and reaching B does not terminate scanning.

### Query sequence

```text
validate/admit workspace
  → route to nprobe lists
  → build PQ LUT
  → scalar scan + global top-B selection
  → bounded batches of copied full coordinates
  → double squared-L2 exact distances
  → distance/ID sort and top-k
```

This preserves the paper's “compressed computation first, fetch once at the end” architecture, but “once” means **one logical fetch stage**, not one physical read or one backend call. All exact distances use original stored coordinates. The core processes an individual query serially; outer drivers supply query-level concurrency. Setting `query_threads` alone does not parallelize a single scan.

Approximation enters through centroid ANN, the nprobe list restriction, and PQ finalist truncation. Exact reranking cannot recover excluded vectors. On an immutable quiescent index, searching all lists **and** retaining all N vectors yields exhaustive exact scoring; either condition alone is insufficient.

**Sources:** [`ivf_pq.h`](../include/bufann/ivf_pq.h), [`pq_table.h`](../include/pq_table.h), [`ivf_numeric_internal.h`](../include/bufann/ivf_numeric_internal.h), [`ivf_lists.cpp`](../src/bufann/ivf_lists.cpp), [`ivf_search.cpp`](../src/bufann/ivf_search.cpp).

## 8. Local coordinate storage and I/O safety

### Reuse the backend, remove vector-graph semantics

`HeapStorage<T>` adapts `InPlaceGraphStore`, its slotted pages, locator arrays, buffer pool, and Linux/libaio machinery. IVF uses zero-degree records and keeps PQ codes in posting arrays rather than the heap. Read paths validate node identity and zero degree instead of blindly trusting a locator. The default page size is 4096 bytes; padded coordinates must fit within one slot/page, so dimension support is bounded by dtype and page size.

This is a local substitute for LindormTable's full-vector batch gets, not a reconstruction of its storage engine, replication, or row semantics.

### Page-local allocation addresses a concrete backend mismatch

The inherited generic allocator can round-robin across a 256-page allocation batch. With the small 64-frame default pool, repeatedly touching all partially filled pages causes bulk-load thrashing. Immutable streaming construction and mutable coordinate preparation therefore fill one page before advancing. Generic append behavior remains unchanged.

This optimization changes physical placement only: source IDs, list membership, and serialized framing do not change. Mutable allocation and coordinate writes remain caller-serialized even when multiple workers route and encode insertions concurrently.

### Bounded copying instead of exposing pinned pointers

Coordinate fetches copy into caller-owned, 64-byte-aligned scratch with padded row strides. No pointer into an evictable frame escapes. The backend batch path can share/deduplicate page reads for multiple vectors on the same page. One page read guard is held at a time when copying, avoiding recursive locking for duplicate IDs/pages.

The adapter uses short ledger locks to check logical availability, releases that lock before pin admission and backend I/O, and rechecks liveness while copying. A global mutex is not held across the entire I/O stage.

### Aggregate pin admission is a safety/throughput tradeoff

Let `Fmin` be the smallest buffer shard's frame count. FIFO admission grants aggregate credits with:

```text
A = max(1, Fmin − 1)
query batch limit = min(256, max(1, floor(A / 2)))
writer request = 1 credit
```

The conservative bound assumes all requests might hit one shard. It bounds aggregate pinning across concurrent readers/writers rather than trusting each operation's batch bound independently. Frame pins release before credit leases; admission waits occur outside the logical-state mutex. Tiny/skewed pools remain bounded, but admission contention can limit throughput. This is not a universal concurrency speedup.

### Fail closed on terminal storage errors

The scoped IVF backend path checks complete reads/AIO completions and aligned full-page writes, including eviction and flush paths. Short writes fail; interruptions are retried where implemented. Aligned page snapshots avoid depending on an ordinary byte vector satisfying direct-I/O alignment.

Terminal heap/pool failures poison the session and admission mechanism, wake waiters, and prevent a later successful checkpoint. This does not provide crash durability or silently repair an inconsistent artifact. These protections are scoped to the IVF path, not a blanket claim about every inherited reader.

**Sources:** [`ivf_heap.cpp`](../src/bufann/ivf_heap.cpp), [`ivf_pin_admission.h`](../include/bufann/ivf_pin_admission.h), [`exact_aio.h`](../include/bufann/exact_aio.h), [`inplace_backend.h`](../include/bufann/inplace_backend.h), [`inplace_backend.cpp`](../src/bufann/inplace_backend.cpp).

## 9. Mutable posting lists and publication-safe updates

### Paper-faithful compact list operations

Each mutable list starts with `initial_population + 16` capacity. When full, it grows by **exactly 16**, not geometric vector growth. Growth admission accounts for old and new arrays coexisting, including concurrent growth on different lists. Deletion does not shrink capacity.

Deletion swaps the final ID/code into the removed slot, repairs the moved ID's locator, decrements length, and marks the removed ID deleted. This matches the paper's compact swap-with-last mechanism (§4.4). Its work is constant in list length, but includes copying one PQ code. It does not compact coordinates on disk.

The +16 policy is memory-frugal for short lists but causes repeated full-list copying under severe insertion skew. With very large nlist, per-list locks, array headers, and reserved slots are themselves material overhead.

### Reservation and coordinate preparation are separate from visibility

The implementation adds explicit state machines:

```text
posting: Absent → Reserved → Live → Deleted
                   └ cancel → Absent

heap:    Absent → Prepared → Live → Deleted
                   └ cancel → Aborted → retry Prepared
```

Insertion proceeds as follows:

1. Take shared lifecycle admission; validate session health, vector, and scratch allowance.
2. Atomically reserve an Absent ID, excluding duplicate concurrent insertions.
3. Route to one frozen centroid and encode with frozen PQ.
4. Prepare coordinates through the serialized heap writer, without publishing a posting.
5. Exclusively lock the destination list, perform any allocation, and copy the unpublished ID/code.
6. Publish heap logical readiness through a callback, then publish list length/locator/live state with no remaining throwing operations.

The key invariant is **a visible posting must never reference unready coordinates**. Bytes present in a page do not by themselves make an insertion visible. The callback ordering and exception cleanup supply in-process cross-component safety where the paper describes a database-level WAL commit.

### Identity and failure choices

Successful deletion permanently forbids reuse of that ID. Canceled, unpublished insertion may retry; an aborted heap record keeps its slot and is rewritten rather than allocating another slot. Missing/nonlive deletion returns false, including when racing with a reserved unpublished insertion; out-of-range IDs throw.

Routing, encoding, and growth failures can cancel reservations/prepared state without automatically poisoning a healthy session. Terminal coordinate errors poison it; cancellation failure marks it failed. If coordinate retirement fails after a posting deletion, the session fails rather than pretending the posting change was rolled back.

Deleted and aborted physical records remain retained. Consequently, equal insert/delete counts keep the **live** population stable but do not bound disk usage or reusable ID supply. No physical garbage collection, slot reclamation, or arbitrary replacement of a live vector under the same ID is implemented.

Centroids, final graphs, PCA, and PQ stay frozen across updates. There is no adaptive centroid movement, list splitting/merging, reassignment maintenance, or drift-triggered rebuilding in this committed architecture. That keeps mutations cheap and stable but can degrade routing fit as data shifts.

**Sources:** `MutableIndex<T>::insert`/`erase` in [`ivf_index.cpp`](../src/bufann/ivf_index.cpp), [`ivf_posting_lists.cpp`](../src/bufann/ivf_posting_lists.cpp), [`ivf_heap.cpp`](../src/bufann/ivf_heap.cpp).

## 10. Concurrency and consistency: safe overlap without snapshot isolation

Ordinary mutable queries and mutations take shared lifecycle admission; checkpointing takes it exclusively. Each list scan holds that list's shared lock for its full scan; growth and deletion need its exclusive lock. Lists are scanned independently, not under one collection-wide read lock.

Therefore a query can observe different lists at different times. Insertions after a list's scan may be missed. Deletion can invalidate a retained PQ candidate before reranking. The code checks candidate membership before fetching, after fetching, and when filtering sorted exact hits. Stale candidates are dropped **without rescanning or refilling**, so fewer than k hits may remain. A deletion can still overlap the final liveness check and return.

These semantics are explicitly weakly consistent: neither collection snapshot isolation nor a linearizable top-k read is promised. Stable coordinate slots and no committed-ID reuse simplify safe fetching, but do not change that visibility contract. Long/skewed list scans can delay writers; there is no bounded mutation-wait guarantee.

Memory admission for mutable queries uses possible ID capacity rather than a racing live count. Shared router/PQ data are frozen and query scratch is caller-owned. Coordinate writers serialize; routing/encoding and list operations can otherwise overlap. Thus “parallel insertion” does not mean parallel heap allocation or unconstrained disk writes.

**Sources:** [`ivf_search.cpp`](../src/bufann/ivf_search.cpp), [`ivf_posting_lists.cpp`](../src/bufann/ivf_posting_lists.cpp), lifecycle locks in [`ivf_index.cpp`](../src/bufann/ivf_index.cpp), [`ivf_heap.h`](../include/bufann/ivf_heap.h).

## 11. Persistence, private ownership, and failure boundaries

### Immutable artifact bundles

The local file family separates:

| Artifact | Purpose |
|---|---|
| `.centroids.bin`, `.hnsw`, `.router` | Full centroids, stock graph, settings and integrity metadata. |
| `.patched.hnsw`, `.refined.hnsw`, `.pca` | Optional optimized-router/ablation artifacts. |
| `.lists`, `.pq.bin` | Immutable grouped postings and copied codebook. |
| `.heap`, `.heap.meta`, `.heap.meta.ivf` | Coordinate pages, backend snapshot, IVF descriptor. |
| `.manifest` | Versioned configuration, source identities, and component digests. |

Open verifies component digests and cross-component shape/population/settings agreement. Graph framing is validated before the legacy HNSW loader follows offsets: labels, centroid bytes, edge IDs/levels, lengths, and layout must agree. Posting IDs must exist in the heap. Formats assume native little-endian data and the router's 64-bit host layout, not portable cross-platform interchange.

The paper persists index metadata/codes through wide tables (§3.4); this implementation instead uses explicit local bundles and defensive framing checks. Native digests use byte counts/FNV-1a; Python reusable-cache identities use SHA-256. These detect integrity changes, not trusted provenance or authenticated content.

### Mutation never writes into a reusable immutable source

`fork` validates the base and makes real private copies of its heap and PQ sidecar. Frozen router/PQ objects are reused in memory where appropriate, while immutable CSR lists become mutable lists. Reopening a mutable checkpoint similarly creates a new private writable heap. This isolates runs and prevents shared-cache corruption; it is not zero-copy cloning or a distributed shared-storage implementation.

Mutable checkpoints add `.mlists`, `.heap.state`, and `.origin`. Under exclusive lifecycle admission they require no pending reservations and matching heap/list live and tombstone state. They flush and write a **fresh immutable checkpoint family**; the current session remains writable afterward. Reopen validates retained physical records, metadata/bitmaps/locators, coordinates and padding, logical state, and unchanged frozen router/PQ identities. Deleted/aborted records persist across reopening.

### Manifest-last is not a transaction

Output paths are freshly/exclusively claimed and existing files are refused. A completed manifest is written last, but publication is not an atomic rename/fsync transaction. Partial files and failed evidence are retained. There is no WAL, rollback, automatic recovery, interrupted-run resume, or crash-safe generation switch.

This is the largest correctness-contract difference from paper §4.4: Lindorm commits only after data/index WAL persistence; this engine publishes in-process visibility and later creates clean checkpoints. Successful insertion must not be described as crash-durable insertion. Flush/storage failure poisons a session; a later output-copy failure can leave a partial checkpoint without necessarily invalidating an otherwise healthy session.

**Sources:** [`ivf_index.cpp`](../src/bufann/ivf_index.cpp), [`ivf_artifacts.cpp`](../src/bufann/ivf_artifacts.cpp), serialization/load paths in [`ivf_router.cpp`](../src/bufann/ivf_router.cpp), [`ivf_heap.cpp`](../src/bufann/ivf_heap.cpp), [`ivf_posting_lists.cpp`](../src/bufann/ivf_posting_lists.cpp).

## 12. Centroid-count selection: paper-inspired T1 + T2, not total-query optimization

The cost model preserves paper §4.1's central tradeoff: finer partitioning reduces scanned codes but increases centroid-routing cost. It evaluates a declared nlist grid rather than hard-coding sqrt(N) or treating 200K as universally optimal.

The procedure is:

1. Select seeded, **disjoint** training/sample rows and synthetic-query rows from the initial base. Official/final queries and GT are not inputs.
2. Compute exact double-L2 top-k GT against the sample.
3. For each eligible nlist, train centroids, construct the final router, and reassign sample rows through the final base/insertion router.
4. For each synthetic query, collect distances to the assigned centroids of its true top-k neighbors. Preserve multiplicity when several neighbors share a centroid.
5. Take the threshold at rank `ceil(target_recall × k)` and count every centroid within that inclusive radius. Sum sample occupancy inside the radius.
6. Measure actual production-router time for that probe count, with separate warmup/repeats, giving T1.
7. Estimate full-base scanned codes as `sample_codes × N/S`; multiply by a profiled per-code scalar scan/selection coefficient, giving T2.
8. Choose minimum mean `T1 + T2`, breaking exact ties toward smaller nlist; retain exclusions, identities, settings, and per-query details.

Preserving GT-neighbor multiplicity and inclusive distance ties makes the paper's rank/radius procedure unambiguous. Scaling expands by N/S, rather than shrinking sample counts by S/N.

### Departures and added policies

* The paper profiles fixed-size SIMD batches; this implementation profiles its real **scalar scan plus bounded candidate selection**, using an existing-code prefix (default 100K codes), and extrapolates linearly per code.
* Candidate eligibility defaults to `S ≥ 32 × nlist`. This configurable sample-per-centroid minimum is project policy, not a paper requirement or statistical adequacy proof.
* The estimated radius uses ideal full-coordinate centroid distances. Approximate routing can select a different list set; actual routed sample-code counts are reported separately rather than silently replacing the model.
* LUT construction, coordinate fetch, and exact vector reranking are excluded. A one-list coefficient does not fully model many-list overhead, cache locality, or concurrent contention.
* Calibration can use a different vector finalist factor from query defaults; the profile's selection cost must be interpreted with its recorded settings.
* Reports are identity-bound and intended to be frozen before validation/final evaluation; the harness rejects an explicit nlist inconsistent with the selected report.

Thus the model chooses a **routing-plus-scan estimate**, not a proven end-to-end QPS optimum, and its predicted nprobe does not guarantee achieved ANN recall. It does not model Lindorm's distributed network/storage behavior or automatically retune the index during updates.

**Sources:** [`ivf_cost_model.cpp`](../src/bufann/ivf_cost_model.cpp), operational [`tests/ivf_cost_model.cpp`](../tests/ivf_cost_model.cpp), `PostingLists::profile_scan` in [`ivf_lists.cpp`](../src/bufann/ivf_lists.cpp), [`benchmark/IVF/cost_model.py`](../benchmark/IVF/cost_model.py), [`settings.py`](../benchmark/IVF/settings.py).

## 13. Resource and numerical contracts are architectural, not incidental

Build, training, open, query, growth, checkpoint, and calibration have explicit preflight size/work checks with overflow protection. Estimates include relevant persistent structures, scratch, temporary replacement buffers, and conservative thread-stack allowances. They are **admission estimates, not hard process-RSS caps**.

Training normally permits at most one billion assignment searches: `S × (max_reps + 1)`. Exact training additionally limits that quantity times nlist. PCA covariance/solver/validation and refinement have separate counted-work limits; those are not comprehensive caps on all graph construction/refresh work. For example, 90M rows with ten mean-update passes require 990M assignments; fifteen passes require 1.44B and fail the default guard. Raising a PCA/routing limit does not raise that training-search limit.

Numeric domains are deliberately separate:

* HNSW traversal uses float kernels and rejects excessive coordinates that could overflow float squared distances.
* PQ LUT entries and chunk-order ADC accumulation remain float; changing summation precision/order can change finalist ties.
* Exact vector/centroid reranking, calibration truth, and training objective distances use ordered double squared-L2 calculations.
* PCA uses its own double fitting/validation arithmetic and float projected coordinates.

The public IVF CMake policy disables fast-math for consumers, overriding inherited aggressive floating-point assumptions so finite-value checks remain meaningful. Sorting consistently breaks equal final distances by ID. These choices prioritize stable, inspectable contracts; they do not establish bitwise equivalence to Faiss or paper results.

**Sources:** [`ivf_config.h`](../include/bufann/ivf_config.h), [`ivf_config.cpp`](../src/bufann/ivf_config.cpp), [`ivf_numeric_internal.h`](../include/bufann/ivf_numeric_internal.h), [`cmake/IVF.cmake`](../cmake/IVF.cmake), workspace estimators in the component implementations.

## 14. Workload scheduling and measurement infrastructure

This is additional experimental architecture, not Lindorm's production request scheduler.

### Sequential and mixed mutation execution

Standalone insertion defaults to one worker but permits bounded parallel workers. Streaming input validates requested coordinates before mutation, then includes worker source reads in insertion timing. Staged input admits/loads the coordinate array before timing. Its reported foreground rate excludes that loading but still includes native waits, heap work, publication, and worker launch/join. Those policies must not be compared as an isolated worker-count speedup. Standalone deletion remains serial.

Three mixed schedulers exist:

| Scheduler | Allocation and termination |
|---|---|
| `weighted` | Shared worker pool; finite operation streams selected in proportion to remaining counts. |
| `fixed_8_8` | Exactly eight mutation and eight query workers, finite query stream. |
| `update_driven` | floor(T/2) mutation workers, remaining query workers; replay queries until updates complete, without a query quota. |

Dedicated mutation claims alternate insert/delete while both streams remain. Dedicated-role mixed and parallel-insert workers launch behind an admission gate; weighted scheduling has different launch semantics and may execute while workers are still launching. Failures stop new admission, join admitted work, and rethrow; they do not roll back already committed mutations or publish a successful partial workload.

OpenMP/MKL concurrency is constrained before mixed execution to avoid hidden oversubscription. Final `query_threads` controls the later scoring pass, not the mixed foreground role split.

### Concurrent query timing is separate from recall scoring

Mixed foreground queries run against weakly consistent changing state and are not scored against a fictitious fixed GT. After updates join and checkpointing completes, a separate final query pass measures recall against the actual final live population. **Post-round query QPS is not concurrent foreground QPS.** Mixed rounds are chained through checkpoints and summarized using counts/times and relative round-result paths, not pooled percentiles.

Overlap means admitted operation intervals overlap, including waits; it does not prove simultaneous CPU execution. Update-driven query volume is unbounded in advance, so percentile storage uses a seeded reservoir capped at 65,536 samples, while means use all queries. Very short update rounds may admit no queries or show no overlap.

### Warmup is explicit and final-query-independent

Optional warmup queries are derived deterministically from frozen base centroids, rounded/clamped for integer dtypes, and run serially on the **same index object** before the timed final pass. They use no final-query/GT selection, are unscored, and have separate timing. This differs from launching a discarded warmup process, which cannot transfer its application buffer pool to another process.

Neither zero explicit warmup nor direct I/O proves a cold-device experiment: opening, hashing, copies, OS/device state, and existing process activity affect cache conditions.

### Provenance and ground truth

Python/shell orchestration separates reusable immutable caches from private work trees, binds caches/prepared runs to source/settings/tool identities, refuses overwrites, and retains logs, failures, neighbors, and structured results. Cache restoration is not a new build measurement. Changed untracked runtime environments still require explicit cache-tag/root discipline.

Ground truth is filtered for each workload's actual live IDs. Insufficient survivors fail rather than silently depressing the denominator. Helpers check coverage and recommend/generate deeper GT for declared update plans. Native recall uses percentage scores and kth-distance tie expansion where available; truncated derived GT can limit complete tie coverage. Missing hits count against recall.

Training comparisons, fixed-router sweeps, and repeated query sweeps are separate tools: centroid recall is not database-neighbor recall, and passing a minimum-recall gate is not proof of an identical-recall speedup over another system.

### Instrumentation boundaries

Queries report routing, LUT preparation, scan/selection, fetch, and exact-rerank wall times, plus scanned codes, finalists, backend calls, and pool reads. Fetch lock/admission/backend-pin intervals are **nested within fetch time**, not extra stages. A logical fetch can span several calls; several candidates can share a page; resident pages need no read. Pool-read counters are not device IOPS. Summed per-query stage times can exceed run wall time under concurrency, and lifetime RSS includes work outside the timed query phase.

**Sources:** [`ivf_cli_execution.cpp`](../tests/ivf_cli_execution.cpp), [`ivf_insert_workload.h`](../tests/ivf_insert_workload.h), [`ivf_mixed_workload.h`](../tests/ivf_mixed_workload.h), [`ivf_query_warmup.h`](../tests/ivf_query_warmup.h), [`benchmark/IVF/harness.py`](../benchmark/IVF/harness.py), [`mixed.py`](../benchmark/IVF/mixed.py), [`common.py`](../benchmark/IVF/common.py), [`benchmark/scripts/check_gt_coverage.py`](../benchmark/scripts/check_gt_coverage.py), [`benchmark/IVF/README.md`](../benchmark/IVF/README.md).

## 15. Defaults are not the fully optimized paper configuration

| Setting | Committed default / distinction |
|---|---|
| Programmatic training | `ann-rebuild`, `sample` initialization. |
| Native CLI/harness training | `ann-reuse-seeded`, `random_partition`; examples explicitly select LV+. |
| Training | sample rate 1, 15 mean-update passes, breadth 64. |
| LV training | centroid rerank factor 3, initial refinement breadth 128, 95% PCA variance. |
| Final router | stock `hnswlib`; optimized modes disabled unless requested. |
| HNSW | degree 16, construction breadth 200, query breadth 128, insertion breadth 64. |
| Native/library search | nprobe 1, k=10, vector rerank factor 2, one query thread. |
| Heap/runtime | 4096-byte pages, 64 buffer frames; common small-run memory allowance 1 GiB. |
| Standalone insertion / warmup | one worker, streaming input, zero warmup. |
| Mixed scheduling | native default `weighted`; shell/Python default `update_driven`, generally 16 foreground workers. |

Large-data recipes explicitly override these. The SIFT recipe uses 64 PQ8 chunks and opts into LV+, optimized artifacts, and `pca-refined` base/insertion and query routing. Its 200K lists, resource allowances, thread counts, and search budgets are example/profile choices—not library defaults, a universal cost-model result, or measured proof of paper equivalence.

**Sources:** [`ivf_config.h`](../include/bufann/ivf_config.h), [`ivf_router.h`](../include/bufann/ivf_router.h), [`ivf_heap.h`](../include/bufann/ivf_heap.h), [`ivf_training_settings.h`](../tests/ivf_training_settings.h), [`benchmark/IVF/settings.py`](../benchmark/IVF/settings.py), [`benchmark/IVF/README.md`](../benchmark/IVF/README.md).


## 16. Paper-to-implementation coverage matrix

| Paper mechanism | Committed status and qualification |
|---|---|
| One logical IVF index with shared centroids/codebooks | Implemented within one local index; no distributed sharing protocol. |
| In-memory compressed postings, disk full vectors | Implemented with local lists and coordinate heap. |
| Large nlist instead of blindly using sqrt(N) | Supported and cost-model guided; example values are not universally optimal. |
| ANN k-means | Implemented with exact/rebuild/reuse/seeded/LV variants. |
| Reuse graph and prior assignments | Implemented with joined coordinate refresh/reinsertion and one prior centroid seed. |
| Convergence-based clustering | Replaced by fixed passes plus final assignment. |
| Full-coordinate topology, PCA traversal, centroid rerank | Implemented, with explicit solver/schedules and retained full-coordinate/ablation copies. |
| Bottom-layer refinement and isolated-node reinsertion | Implemented as frozen proposals plus one zero-indegree repair pass; connectivity measured, not guaranteed. |
| PQ/RaBitQ and SIMD fast scan | Only global non-residual PQ8 and scalar scanning implemented; no RaBitQ/PQ4 FastScan. |
| One final full-vector batch-get stage | Logical pattern preserved; bounded local batch calls/page reads replace database/network gets. |
| Immediate compact posting mutations | Implemented, including +16 growth and swap deletion. |
| Data/index WAL-backed durable commit | Not implemented; publication-safe memory state and clean checkpoint families instead. |
| Distributed regions, random sharding, broadcast/merge | Not implemented. |
| LDMaster coordination, failover, elasticity, replication | Not implemented. |
| Filter planning, pre/post filtering, attribute pushdown | Not implemented as IVF query features. Live-ID GT filtering is evaluation, not filtered ANN. |
| Hybrid full-text/vector search and RRF | Not implemented. |
| Metadata deduplication across tables | Not implemented; local frozen-bundle reuse is a different capability. |
| Multi-tenant lists and promotion to tenant IVF | Not implemented. |
| Physical slot reclamation / adaptive drift maintenance | Not implemented in this committed revision; do not infer these from live updates. |
