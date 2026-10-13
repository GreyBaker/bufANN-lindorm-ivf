# Implementation Plan — Single-Node Lindorm IVF in BufANN

Some of this is out of date, most recent [planning](../benchmark/IVF/README.md) is elsewhere and may not be included here yet. 

## 1. Objective and scope

**2026-10-09 work-plan addendum:**
[Lindorm LV/LV+ training fidelity](lindorm-training-fidelity-addendum.md) schedules
optimized centroid search inside W9 training (discussion 1A) and separates
traversal breadth from returned-candidate rescoring (1B). It preserves existing
controls/evidence and makes the moving-PCA/refinement policy explicit. **LV and
LV+ must be designed to largely match the paper; LV+ adds seeding to the same
optimized LV path, not a different search/maintenance policy.** The
`m/nlist >= 32` review is now **active**, with a declared configurable project
minimum (default 32) and historical report semantics retained. The first LV/LV+
functional implementation is underway in the mounted checkout; qualification
remains open. PQ/SIMD scanning is **deferred and last**.

Implement an IVF backend **inside this BufANN checkout** following [executive-summary.md](executive-summary.md), then compare it with SPFresh. Work items keep their summary numbering: core items W1–W7 (summary §§4–10) and paper mechanisms W8–W10 (summary §13). **W1–W10 are required.** On 2026-10-09 the user promoted ANN-accelerated K-Means (W9/S09) into scope and subsequently restored the predictive cost model (W8/S10/M8), superseding its temporary deferral. Exact GEMM remains optional reference work. BufANN/Vamana and SPFresh remain comparison baselines. The paired SPFresh comparison (M5) runs last, after all required items, including W8 implementation and prediction validation, are qualified. Current status is in §5 and [implementation-progress.md](implementation-progress.md).

**Core deliverable (W1–W7):** one multithreaded process with

* one shared set of K-Means centroids and an HNSW router over them;
* memory-resident posting lists of vector IDs and existing PQ codes;
* full vectors in the BufANN heap, read through its buffer pool;
* search: route → PQ list scan → bounded candidates → one logical batched full-vector fetch → exact top-`k`;
* immediate inserts/deletes, safe under concurrent queries;
* SPFresh's dataset splits, mutation IDs, ground truth and workload structure.

**Execution sequence:** S08's functional W10 implementation is complete. The
2026-10-09 WSL run adds an initial SIFT1M build/query measurement at `nlist=1024`,
not large-`nlist` or production qualification. W9/S09 reuse/seeding and native
trainer selection are now implemented in Windows `ivf-only`, with mounted small-scale
checks; see [s09-ann-kmeans.md](s09-ann-kmeans.md) and progress for test status.
The user-updated native copy now matches the tested handoff; a fresh W9
1024-list SIFT1M build/query passed (373.99 s build, 242.45 s training,
92.42% Recall@10). Qualify the higher-`nlist` grid and real update workloads,
then calibrate and validate S10 before proceeding to S07/M5.
S10's estimator/tool/provenance are now implemented with mounted small fixtures;
the user native handoff at clean `9b1d6243` matches tested hashes, and a fresh
native build passed 6/6 targeted fixtures. The pinned 10K-list W9 study completed
and passed its evidence audit (matched ANN training controls, then seeded
downstream/fixed-router sweeps); the 50K manual feasibility control also completed
and passed its audit, explicitly under-sampled/W8-ineligible at 18 sample rows/list.
A frozen base-only SIFT1M W8 pilot completed and passed its audit: 10K selected,
best under its declared target-reaching probe policy, with absolute-cost
underprediction disclosed. This is not required SIFT10M prediction validation;
report-only shell preparation still needs an explicit matching IVF_NLIST workaround;
production calibration and its required study remain pending. See
[s10-cost-model.md](s10-cost-model.md). S07/SPFresh comparison is
deferred, **not underway**. Reusing unchanged workload/GT/scoring contracts is
not a performance comparison. The initial native W9 run is not a matched
training-control comparison or large-`nlist` qualification.

**Current deliverable adds:**

* **W10:** §4.3 optimized routing (PCA-compressed traversal with full-precision centroid rerank, bottom-layer refinement, isolated-node reinsertion), implemented as patches to hnswlib.
* **W9:** §4.2 ANN-accelerated K-Means with graph reuse and prior-assignment seeding, integrated as the large-`nlist` production trainer; exact and rebuilt-ANN controls remain available.
* **W8:** the §4.1 predictive `nlist` cost model, calibration tool, selection-report plumbing and prediction-vs-measurement validation are required. Manual grids remain necessary validation controls and interim feasibility studies, not a substitute for the predictor; never use reserved final queries to select `nlist`.

Paper anchors: §§3.3–3.4 query/storage, §4.1 large `nlist` and cost model, §4.3 centroid HNSW and its optimizations, §4.4 updates. Run coordinator/worker logic as local query execution, not simulated LDServers. Parallelize across queries and updates; each query scans its lists in its calling thread (no nested pools).

### Guiding principles

* **Implement the paper, not an improved paper.** Build what §§3–4 describe. Where the paper is ambiguous, choose the simplest reading that matches its text, record the choice in one line, and move on. Do not add mechanisms the paper doesn't describe (multi-pass repair schemes, alternative aggregation rules, new stopping criteria, extra modes) unless a measurement shows the paper's version cannot work, and then report it as a deviation.
* **Research prototype, not a product.** Build only the infrastructure needed to run, reproduce and measure experiments. No backward-compatibility layers, format negotiation or hardened input schemas beyond rejecting mismatched configurations.
* **Measure rather than gate.** Component gates check correctness. Performance and fidelity questions are answered by measurements reported in the paper, not by acceptance thresholds that block progress.

### Deliberate deviations

| Paper feature | This implementation |
|---|---|
| LindormTable/DFS, regions, RPCs, masters, replication | BufANN heap + buffer pool; no distributed subsystem |
| PQ/RaBitQ with 4-bit SIMD scanning | Existing global, non-residual 8-bit PQ; scalar ADC; 32 chunks. **PQ/SIMD follow-up is deferred and last under the training-fidelity addendum; no immediate implementation** |
| `nlist` cost model (§4.1) | **Required (W8/S10), functionally implemented.** Paper's `T1+T2` estimator profiles our scalar scan and selected W9/W10 settings; production calibration/declared-grid validation remain open |
| ANN K-Means with reuse/seeding (§4.2) | **Required (W9/S09).** Patched hnswlib, rather than Faiss; existing scalar exact checks suffice initially; GEMM reference work is optional |
| PCA traversal and refinement (§4.3) | Required (W10/S08), implemented as a minimal patch series on hnswlib; unmodified hnswlib remains the vanilla baseline |
| Dual-system WAL commit (§4.4) | Quiescent local checkpoints and clean reopen; no crash-atomic updates |
| Filtered/hybrid search, metadata sharing, multi-tenancy | Out of scope |

**Without fast scan.** The paper attributes part of its `nlist` curve's right-side rise to clusters too small for 32-code SIMD batches. Scalar ADC has no such penalty; routing cost and per-list overhead matter instead. Don't infer an upturn or its cause without measurements. Calibrate W8 to the actual scalar kernel, not the paper's.

**Frozen index state.** Centroids and PQ codebooks are frozen after the initial build, per the summary's comparison policy: no retraining, splits, merges or vector-level graph repair. The paper gives no long-term retraining policy; freezing is a project choice, and §4.3's skewed-insert experiment measures its cost.

## 2. Integration design

Paths are relative to `ivf-only/`; headers live in `include/bufann/` and sources in `src/bufann/` unless shown. Proposed paths do not imply implemented functionality.

| Component | Location (existing or proposed) | Reuses |
|---|---|---|
| Input/config/PQ contracts (S01) | `ivf_config.h`, `ivf_io.h`, `ivf_pq.h` + sources | `FixedChunkPQTable` encoder/ADC with opt-in pair-table omission |
| Lifecycle, search/update API | `ivf_index.{h,cpp}` | `inplace_backend.*` storage primitives, not `bufann_api.cpp` graph ops |
| Posting lists, locator, checkpoints | `ivf_posting_lists.{h,cpp}` | Binary I/O conventions, guarded heap access |
| Exact reference trainer | `ivf_centroids.{h,cpp}` | Existing scalar oracle; optional later bounded MKL GEMM implementation |
| ANN trainer and mutable training graph (W9) | `ivf_centroids.{h,cpp}`, `ivf_router.{h,cpp}`, `ivf_patched_graph.{h,cpp}` + hnswlib patches | Existing rebuilt-HNSW training control; caller-supplied entries; shared sampling/means/empty recovery |
| Router interface | `ivf_router.{h,cpp}` | hnswlib v0.8.0, vanilla and patched builds |
| hnswlib patch series (W10) | `dependencies/hnswlib-patches/` (proposed) | Upstream hnswlib v0.8.0 |
| PCA | `ivf_pca.{h,cpp}` | Implemented deterministic double covariance/cyclic Jacobi solver |
| Cost model (W8/S10; required, implemented) | `ivf_cost_model.{h,cpp}`, `tests/ivf_cost_model.cpp`, `benchmark/IVF/cost_model.py` | W9 trainer, W10 router, W4 scan loop |
| Build tool | `tests/ivf_build.cpp` | Typed `.bin` loading, PQ formats |
| Driver | `tests/ivf_driver.cpp` | `bufann_driver.cpp` loaders/metrics/rounds, `weighted_update_pool.h` |
| Benchmark scripts | `benchmark/IVF/common.sh`, `benchmark/IVF/sift/*` | `benchmark/scripts/utils.sh`, BufANN/SPFresh prepare/run structure |
| Sweeps, validation | `benchmark/IVF/qualify.py`, `tests/ivf_*`, `benchmark/IVF/cost_model.py` | GT and recall helpers; prediction-vs-measurement study remains pending |

### Dependencies

hnswlib v0.8.0 (Apache-2.0) is vendored with its archive and checksum recorded in `dependencies/hnswlib.md`. It is built two ways:

* **Vanilla:** upstream hnswlib plus only the prefetch-overread fix below. This is W2's router and the paper's vanilla comparison point (§§5.2.1–5.2.2).
* **Patched:** vanilla plus the W10 patch series. Patches are small, individually reviewable diffs limited to what §4.3 needs: caller-supplied entry points, traversal over a second (PCA-projected) vector array, bottom-layer out-edge replacement, and isolated-node reinsertion. No other algorithmic changes. Record each patch's purpose in one line.

**SIMD.** `NO_MANUAL_VECTORIZATION` is used only in sanitizer builds. Fix or bound the ASan-confirmed upper-layer prefetch overread as a patch applied to both builds, so production benchmarks run hnswlib's SIMD distance kernels. Benchmarking with SIMD disabled would inflate routing cost, shift the `nlist` optimum and handicap IVF against SPFresh. If the fix proves impractical, document the remaining exposure and keep SIMD on for benchmarks anyway.

**Fallback.** A from-scratch native centroid graph is not planned. Consider it only if a required patch proves infeasible, and only after a separate user decision. Faiss is not needed.

### Data ownership and artifacts

* **IDs.** Source row IDs are stable external IDs. Heap nodes carry those IDs, and metadata for `NPTS_FULL` is reserved before workers start. Never return heap slot indices as dataset IDs.
* **Lists.** Contiguous IDs plus a flat `length × pq_chunks` code block owned by the list. Avoid a second permanent full-dataset PQ array; account for any duplication.
* **Locator.** Preallocated live state, list ID and slot per ID. A live ID's list ID is stable; slots change only under the owning list's lock during swap deletion. No heap-slot or ID reuse within a run; deleted heap bytes are retained and reported.
* **Artifacts.** Existing immutable component suffixes (`.centroids.bin`, `.hnsw`, `.router`, `.lists`, `.heap`, `.heap.meta`, `.heap.meta.ivf`, `.pq.bin`, `.manifest`) and mutable families stay as implemented. W10 adds patched/refined graphs and `.pca`; W9's `IVFINDEX3` manifest stores trainer settings with an explicit optimized-mode flag, without adding index components. v1/v2 inputs still load and retain legacy producer settings in mutable `.origin` files. No additional compatibility layer is built. Loading rejects missing components and configuration mismatches.
* **Manifest.** Records dtype, metric, dimension, counts, PQ identity/chunks, trainer mode, initialization, training breadth, seed/pass cap, `nlist`, router mode and HNSW/refinement/PCA settings. W9 now propagates trainer settings into CLI records and harness cache identity; mode implies reuse/seeding policy and records label it explicitly. W8 adds selection provenance: manual or model-selected, report path/hash and settings in build/preparation metadata and experiment records. Preserve existing artifact contracts; do not silently relabel earlier manual builds as model-selected.
* **Cache.** Cache built bundles keyed by the manifest's build settings. Query-time settings (`nprobe`, rerank factors, router mode among built routers) need no rebuild.

## 3. Work items and acceptance gates

Order: W7 → W3 → W1 functional → W2 → W4 → W5 → W6 → W10 → W9 → W1–W4 production qualification → W8/S10 implementation and validation → M5 comparison/report. Existing functional work is retained; W9 precedes the large-`nlist` sweep, and W8 uses the qualified trainer/router/scan measurements.

S10 functional coding/fixtures have proceeded ahead of production qualification;
this does not close the prerequisite gates or qualify calibration on real data.

### W7 — Harness-compatible driver

1. Keep the implemented `ivf_driver`/`ivf_build` targets and wiring. Add the required `ivf_cost_model` tool and selection-report plumbing with W8, without changing existing workload or scoring behavior.
2. Adapt the BufANN driver's workload and I/O behavior, not its graph maintenance. `float` and `uint8` with L2 are the benchmarked configurations; the implemented `int8` path stays as is but needs no production testing.
3. Keep the implemented flags (`--index_prefix`, `--data_type`, `--dim`, `--workload`, `--query_file`, `--gt_file`, `--recall_at`, `--query_threads`, `--result_file`, `--full_data_file`, counted insert/delete files). `mixed.py` runs one joined, checkpointed round per process. Add `--router_mode`/`--router_rerank_factor` with W10. Keep `--nprobe`, vector rerank factor and router breadth distinct.
4. Emit JSON lines beginning `{"baseline":"IVF"` to stdout and `--result_file`, using `bufann_driver.cpp`'s field names and units plus router and trainer mode.
5. Provide `benchmark/IVF/sift/{query,insert,delete,update}{_prepare,}.sh` using `eval_tempfiles/<dataset>/IVF/<workload>/` and `eval_cached/`. Restore mutable files by copying, never through writable symlinks into the cache.
6. Build from the configured base prefix of `data/full.bin` or `data/base.bin`. Don't call `provision_index_into`, require a Vamana index, or modify `baselines/SPFresh/` or `benchmark/SPFresh/`.

**Gate:** targets compile; malformed inputs fail clearly; a tiny fixture yields parseable records and the expected prepare/run tree.

### W3 — Reuse existing PQ

1. Read `_pq_pivots.bin`/`_pq_compressed.bin`, checking dimensions, chunk count, code count and file lengths; infer chunks from metadata.
2. Score with the existing global centering and chunk offsets; no residuals. ADC-only loading skips PQ-to-PQ matrices, leaving other callers' defaults unchanged.
3. Repack base codes by list after assignment; encode inserts with the same codebook using a side-effect-free encoder.
4. Reuse the PQ build helpers with IVF prefix plumbing; keep `build_pq.sh`'s defaults.

**Gate:** code bytes and ADC distances match existing helpers, including inserts; save/load preserves scoring. Disclose that PQ trains on `NPTS_FULL`, including the insert tail.

### W1 — IVF build with large `nlist`

1. Flags: `--nlist` (manual or W8-selected, with provenance saved), `--trainer_mode`, `--init sample|random_partition`, `--sample_rate`, `--max_reps`, `--seed`, `--build_threads`, `--training_breadth`, base count, PQ prefix, router/refinement/PCA settings and output prefix. Native CLI/harness default to `ann-reuse-seeded`/`random_partition`/breadth 64; programmatic defaults remain legacy rebuilt/sample. IVF defaults are environment-overridable per profile. Add cost-model report identity/settings plumbing with W8; existing manual builds remain valid controls.
2. Train on a sample of the initial base, converting integer samples to float. Validate `nlist <= training_count <= base_count` and finiteness; recover empty clusters deterministically.
3. **W9's ANN-reuse-seeded trainer is the intended large-`nlist` production path.** Retain `ann-rebuild` as a control and the current scalar exact trainer as a fixture/tractable-quality oracle. Exact GEMM is optional later reference work, **not an implementation prerequisite or completion gate**. The illustrative 200K-centroid/6.4M-sample exact pass costs about 3.3 × 10¹⁴ flops; this motivates ANN training, not a claim about measured feasibility. Set the grid's upper end from measured ANN time/memory and sample adequacy on the target node. Avoid serial K-Means++ and per-cluster file fan-out.
4. Report end-to-end and per-iteration graph build/update, assignment and mean-update time, peak memory, occupancy, empty lists and objective. Label approximate assigned SSE separately from exact nearest-centroid evaluation. The initial SIFT1M run measured total native build time, not these training phases.
5. Stream the base through the configured router to build lists, never loading all raw vectors. Record router agreement with exact nearest-centroid assignment on a sample.
6. Create the heap directly with coordinate-bearing slots and empty neighbor lists; never invoke `bufann_build` for an unused Vamana graph.
7. Write and validate the bundle before publishing it.

**Functional gate:** deterministic single-threaded fixtures; complete graph-free bundle and reload; every base ID listed once with correct code and coordinates.
**Production gate:** W9 integrated into native build/cache/persistence; SIFT1M and uint8 SIFT10M builds across the feasible grid toward 200K within the measured time/memory budget. Report infeasible or under-sampled candidates rather than skipping them silently. The SIFT1M `nlist=1024` run closes the initial real-data build/reload smoke, not this large-`nlist` gate.

### W2 — Vanilla centroid HNSW

1. Build and persist vanilla hnswlib over full-precision float centroids with labels equal to list IDs, exposing construction parameters, seed and separate query/insert breadth.
2. Route integer queries as float; return unique, valid, nearest-first list IDs with `1 <= nprobe <= nlist`.
3. The graph is read-only after build/load; set search parameters before concurrency. S03's `ef=1` plus oversampling `k = min(nlist, max(nprobe, breadth))` with rescoring stays. Ensure breadth covers the maximum `nprobe`.
4. Keep an exact centroid mode for correctness tests and router recall; when `nprobe == nlist`, enumerate all lists.
5. S08 adds mode dispatch (`route`, `route_insert`) for W10's modes; vanilla stays selectable.

**Gate:** persistence round-trip; recall vs. exact top-`nprobe`; concurrent read-only search; routing timed separately from scanning.

### W4 — Query path and batched rerank

1. Route `nprobe` lists, build one ADC table, scan only those lists, and keep at most `B = min(live_count, ceil(k × rerank_factor))` candidates.
2. Hold a shared lock per list scan with per-thread scratch; release list locks before heap I/O. No pointer into a mutable list survives unlock.
3. Run **one logical fetch stage after scanning** via `pin_pages_for_nodes_batch`/`pin_pages_batch` and `BufferPool::pin_batch`: deduplicate pages, respect `kPinBatchMax` and pool capacity, and split large batches safely.
4. Copy coordinates under frame read guards after validating identity and liveness; compute exact L2; return sorted unique IDs, with an explicit invalid sentinel for missing results.
5. Time routing (including PCA projection and centroid rerank for W10 modes), scan, selection, fetch and rerank. Count scanned lists/codes, router distance computations, candidates, physical pages and cache hits/misses.

**Constraint:** `batch_compute_dists` pins one page at a time, so build the IVF rerank helper on the real batch primitives without changing legacy queries.

**Gate:** probing all lists with a budget covering every live vector matches brute-force top-`k` (tie-aware). Cover empty lists, fewer than `k` candidates, duplicate pages and batch-limit boundaries; instrumentation confirms no heap reads before candidate selection.

### W5 — Immediate inserts/deletes and clean persistence

1. **Insert:** reject invalid or duplicate live IDs; reserve the ID; `route_insert`; encode PQ; write and tag the heap node; append ID/code under the list's exclusive lock; publish only when heap and list are ready. Roll back failures.
2. **Growth:** add exactly 16 slots when a list fills.
3. **Delete:** under the owning list's exclusive lock, recheck the locator, swap the last ID/code into the slot, update the moved ID's locator, shrink and publish. Missing IDs are no-ops with a counter.
4. Mark deleted heap records inactive; leave slots unreclaimed.
5. Checkpoint only when workers are quiescent; reopen only complete, matching checkpoints. No WAL durability is claimed.

**Gate:** immediate visibility; swap-delete correctness at first/middle/last/only slots; growth; error handling; clean save/reopen; centroid, codebook and router hashes unchanged.

### W6 — Mixed-workload concurrency

1. One `shared_mutex` per list; locator/state arrays reserved before workers start; conflicting same-ID operations serialized. Guard a moved locator entry with the owning list's lock.
2. Lock order: per-ID reservation → owning list lock → short heap publication. Scans release list locks before frame guards.
3. Reuse `WeightedUpdatePool` for weighted scheduling, keeping round caps and post-round scoring barriers but no graph maintenance. The implemented fixed 8+8 split is kept as a secondary, separately labeled mode.
4. Routers, PCA and codebook are immutable; query scratch is thread-local. Avoid nested OpenMP/BLAS oversubscription.
5. Guarantee race freedom and visibility of completed updates. Score recall at quiescent round boundaries; concurrent phases measure throughput and latency.

**Gate:** same-list stress during growth and swap deletion; query-versus-delete during rerank; no corruption or deadlock; locator/list/live-count consistency every round; ASan/UBSan and a race detector on small fixtures.

### W10 — Optimized centroid routing (§4.3; S08)

Implemented as the hnswlib patch series (§2). Each step follows the paper's description; interpretations are noted inline.

1. **Patches.** Caller-supplied entry points; traversal over a second vector array; bottom-layer out-edge replacement; isolated-node reinsertion. With all features off, the patched build must behave identically to vanilla.
2. **Refinement.** After build, for each centroid run an HNSW search for its neighbours, apply hnswlib's standard pruning, and replace its bottom-layer out-edges. Compute all new lists before applying any, so the result doesn't depend on processing order. No reverse-edge insertion (the paper doesn't mention it).
3. **Isolated nodes.** As in the paper: find nodes with zero in-degree and reinsert them once. Report bottom-layer reachability from the entry point as a measurement; don't build a multi-pass repair scheme.
4. **PCA traversal.** Fit PCA on the centroids and keep the smallest `d'` explaining at least 95% of variance (`--pca_variance`). Traverse in PCA space, return `min(nlist, ceil(nprobe × router_rerank_factor))` candidates (default factor 3, as in the paper's experiments), and rerank them against full-precision centroids. Use a deterministic solver and fixed seed; record `d'`. Expect a small gain on 128-dimensional SIFT and report whatever is measured.
5. **Modes.** `exact`, `hnswlib` (vanilla), `patched-full` (patches built, features off), `refined`, `pca`, `pca-refined`, all behind `CentroidRouter`. `route_insert` uses the production mode with `nprobe = 1`.
6. **Persistence.** Save the patched graph and PCA; both are immutable after load.

**Gate:** `patched-full` returns the same results as vanilla for the same seed and inputs; persistence round-trip; concurrent read-only search. Then measure, on fixed centroids and lists: router recall vs. exact, QPS at matched recall, distance computations and reachability for each mode. Rerun W4 query-path and W6 concurrent-reader fixtures for new modes; the full S04–S06 suite is not rerun.

### W9 — ANN-accelerated K-Means (§4.2; S09; required)

**Planned fidelity correction:** follow
[the LV/LV+ addendum](lindorm-training-fidelity-addendum.md) for optimized training
search and count-based candidate reranking. The full-coordinate-only S09 path
below is the retained implementation/control, not the complete paper LV+ path.
The new path has its own explicit projection/refinement lifecycle and identity.

Implemented on patched hnswlib in Windows `ivf-only`, not a new graph
implementation; the rebuilt-HNSW trainer remains a selectable control. Mounted
small-scale verification and implementation choices are documented in
[s09-ann-kmeans.md](s09-ann-kmeans.md). The production measurement gate is still
open; notably, short under-sampled tests show reuse quality/reachability losses.

1. **Initialization and loop.** Use seeded random partitions and their means as the paper-style default; retain sampled-centroid initialization as a control. Each pass updates centroid means and assigns training vectors through ANN; preserve shared deterministic reductions and empty-cluster recovery. Keep the declared `max_reps` cap/iteration policy, record completed passes, and finish with assignment against the returned centroids. No new convergence heuristic is implied.
2. **Graph reuse.** Join assignment workers, snapshot all pre-update neighbor lists, replace all centroid coordinates in the training graph, then reinsert each stable ID once in ascending order. At each layer select the nearest eligible old neighbor as the entry; ordinary global descent is the fallback if none qualifies at the node's highest layer. Preserve levels and use upstream per-layer repair/search/pruning/reciprocal connection, not `updatePoint`'s extra two-hop neighbor refresh. The minimal training patch is separate from W10's frozen refinement. Do not add reciprocal-proposal passes, spanning-tree preservation or multi-pass connectivity repair.
3. **Assignment seeding.** Use each vector's previous centroid ID as the next pass's entry point; use the global entry point on the first pass or for an invalid/missing seed. Record fallback counts and assignment breadth. Centroid IDs remain stable across iterations.
4. **Controls and quality.** Expose `exact`, `ann-rebuild`, `ann-reuse`, `ann-reuse-seeded`. Compare shared sample IDs, initialization, seeds, iteration policy and resource envelopes; measure runtime, RSS, graph-update work, exact nearest-centroid SSE/MSE, occupancy and assignment agreement. Compare downstream recall at fixed probes/rerank and at declared matched-recall targets. Use the existing scalar exact trainer and bounded exact nearest-centroid evaluation on tractable inputs; disclose partial reference coverage at large scale. Exact evaluation of ANN-produced centroids is not evidence of equality to fully exact-trained centroids. GEMM can accelerate later reference experiments but is not required. Do not invent a 5% quality gate or require the paper's 5–10× speedup.
5. **Lifecycle and integration.** The training graph is mutable only between joined assignment phases, never shared with query/update sessions. After training, build and freeze the W10 query/insert router and PCA on the final centroids as before. Propagate trainer/initialization/breadth/reuse/seed settings through `ivf_build`, manifests, diagnostic exports, canonical JSON and build/cache identity; keep query-only overrides distinct. Full-coordinate ANN training is the first implementation; any W10-assisted training ablation must declare its per-iteration projection/refinement policy, not silently reuse stale PCA.

**Functional gate:** typed deterministic fixtures for initialization, coordinate
refresh, old-neighbor reinsertion, seeded assignment/fallbacks, empty recovery
and final occupancy; instrumented checks for unsafe concurrent refresh; native
build/reopen and cache identity reflect the selected trainer.

**Measurement gate:** bounded exact-vs-ANN quality/time comparisons, then real
SIFT1M/SIFT10M large-`nlist` training and downstream query measurements. Losses,
unreachable nodes, infeasible candidates and absent speedups are reported results,
not hidden by retuning or graph repair. This is required before M8 and M5; it is not
implemented by the 2026-10-09 smoke run.

### W8 — Predictive `nlist` cost model (§4.1; S10; required)

**Restored to the critical path by user direction on 2026-10-09; estimator,
calibration tool and provenance are functionally implemented.** Small checks use
existing components; production calibration and the required study remain open
before M5. See [s10-cost-model.md](s10-cost-model.md) for interpretations and use.
Manual sweeps remain validation measurements, not an implementation of W8.
Follow the paper's estimator, without expanding it into a storage cost model.

1. **Sampling.** Sample `m` base vectors (seeded) and about 100 disjoint held-out base rows as synthetic queries. Official benchmark queries and GT are never used for calibration. Compute exact top-`k` for each synthetic query within the sample.
2. **Candidates.** Keep a candidate only if `m >= nlist*min_sample_per_centroid`; default minimum 32. Record the declared policy, ratios, below-historical-32 warnings and dropped candidates. This is a **project adequacy policy**, not an explicit paper requirement or its SIMD batch size. Its empirical review is active under the addendum; version-1 reports retain implicit 32 and existing reports/exclusions are immutable.
3. **Training.** Cluster the sample with the selected production W9 trainer/settings at each candidate and build the production W10 router. Record trainer, initialization, assignment breadth, seed and iteration policy in the prediction report; use the same policy in the measured builds. Exact top-`k` synthetic-query GT remains exact even though training assignment is approximate.
4. **`nprobe` estimate (per query).** Rank the centroids of the query's GT neighbours' clusters by distance; `x` is the distance to the `ceil(ρ × k)`-th. The estimated `nprobe` is the number of centroids within `x`.
5. **T1.** Run the production router at each query's estimated `nprobe`; average the times.
6. **T2.** Count codes in the clusters within `x`, scale by `N/m`, and multiply by the per-code scan time profiled on the real W4 scan loop over 100K codes. Start with this single-coefficient model, as in the paper. Add a per-list overhead term only if the measured sweep shows systematic error, and report that as a deviation.
7. **Choice and report.** Select the `nlist` minimizing `T1 + T2`. Write a plain JSON report of input/sample identities, settings, eligible/dropped candidates, per-candidate estimates and the choice; pass the chosen value to `ivf_build --nlist` with the report path/hash recorded as its source. Fetch and exact rerank are outside the model, as in the paper; report how they vary. A minimum of `T1+T2` is not a claim of maximum end-to-end QPS, especially with the observed fetch/admission bottleneck.

**Functional gate:** seeded disjoint sample/query IDs, exact synthetic-query GT,
candidate eligibility/drop accounting, per-query probe estimates and scaled code
counts pass small deterministic fixtures; tool/report round-trip and selection
provenance survive preparation/cache use. Calibration never reads official queries.

**Required measurement gate:** on SIFT10M at recall@10 = 0.90 and recall@100 = 0.95,
compare estimated and measured T1, T2 and their sum across the declared `nlist`
grid (the Figure 6 analogue), with end-to-end latency/fetch/rerank alongside.
Profile and freeze the model before the validation sweep under declared CPU,
thread, cache and repetition settings. Report whether the selected `nlist` is
within 1.10× of the best measured `T1+T2`, and separately its end-to-end latency
ratio. Misses are reported results, not grounds for hidden refitting or a new
predictor mechanism. Completion requires the implementation and study, not a
guaranteed prediction-accuracy result.

## 4. Benchmark parity and measurement

### 4.1 Protocol (declared before final runs)

Before the M5 comparison, save one protocol file recording dataset, PQ and query identities; the tuning vs. final-comparison query split; hardware, allowed CPUs and thread counts (foreground, OpenMP, BLAS); cache/warmup policy; memory budgets; repeats; and executable versions. Final queries are reserved for comparison, never for tuning.

* **Component checks** (W8/W9/W10) use one seed and are measurements, except where §3 states a gate.
* **W10 reference:** patched-mode router recall should be within 0.02 absolute of vanilla at matched budgets. A miss is reported, not a blocker.
* **W8 reference:** report the §3 1.10× model-selection criterion and end-to-end ratio; disclose misses without post-validation refitting.
* **`nlist` selection:** declare the candidate grid, model calibration/validation protocol, feasible resource limits, recall target, tuning budget and selection rule before final runs. Save the frozen model report and tuning-set validation sweep; label manual controls separately. Freeze the chosen setting before scoring final queries, and provide SPFresh an equivalent tuning budget. Existing smoke runs have no predicted optimum.
* **Final comparison:** at least 3 repeats with fixed build seeds; report variability.
* **SPFresh:** the same hardware/CPU envelope, repeats, tuning-query set and tuning budget. Record SPFresh's pool, search and maintenance settings. Tune it rather than relying on defaults. No SPFresh source changes.

### 4.2 Workload parity and controls

**Preparation:** match `benchmark/SPFresh/sift/*_prepare.sh` and the shared layout helpers exactly.

* Query: the initial base prefix, with matching filtered GT.
* Insert: IDs `[BASE_POINTS, BASE_POINTS + INSERT_POINTS)`.
* Delete: IDs `[BASE_POINTS - DELETE_POINTS, BASE_POINTS)`.
* Mixed round `r` (zero-based): live set `[0, BASE_POINTS + (r+1) × INSERT_CAP)` minus `[BASE_POINTS - DELETE_POINTS, BASE_POINTS - DELETE_POINTS + (r+1) × DELETE_CAP)`.

Use the shared `compute_split`, `compute_update_layout`, counted-ID helpers, `resolve_incremental_gt` and `generate_round_groundtruths`. Keep SPFresh's equal insert/delete round caps for paired runs.

**Ground truth:** use `baselines/DiskANN/tests/utils/incremental_groundtruth.cpp`. Keep its failure on insufficient deep-`K` survivors; regenerate a deeper cache rather than padding.

**Scoring:** reuse `calculate_recall` and BufANN's field conventions, normalizing its percentage to SPFresh's tie-aware fraction. Record `nlist` and its source, `nprobe`, rerank factors, PQ chunks, trainer and router modes, PCA `d'`, thread allocation and cache policy. Keep post-round recall separate from concurrent QPS.

**Controls:** identical data, queries, update IDs and caps, recall `k`, hardware/CPU budget and warmup policy. SPFresh's insert/append/delete pools are configured independently, with search during updates on and reassignment off by default. Record `LoadAllVectors` and total RSS.

### 4.3 Additional experiments

* **E1 — Skewed insert stream.** The default insert tail matches the base distribution, the easiest case for frozen centroids. Prepare a dataset variant whose insert tail is concentrated in a few regions (for example, tail vectors drawn from a small set of base clusters) and regenerate its ground truth with the same helpers. Run IVF and SPFresh on the insert and mixed workloads.
* **E2 — Posting-list growth.** Log posting-list length distributions (max, p99, mean) at each round boundary for the default and skewed streams, alongside recall and latency.
* **E3 — Equal memory budgets.** Compare IVF and SPFresh at one or two fixed memory budgets, accounting for IVF's resident PQ codes, lists, router and buffer pool against SPFresh's resident centroid index and caches.

### 4.4 Evaluation sequence

1. Tiny synthetic fixtures: correctness and lifecycle tests.
2. SIFT1M float: all four workloads, checkpoint/reopen, QPS–recall sweep.
3. uint8 SIFT10M: large-`nlist` builds, `nlist` sweep, weighted mixed mode.
4. SIFT100M only if inputs and resources are configured; no billion-scale requirement.
5. **Declared `nlist` grid:** about `sqrt(N)`, 10K, 50K, 100K and 200K, extended toward 500K–1M only where W9 ANN training fits measured time/memory and `nlist <= training_count`. Record sample adequacy, occupancy and infeasible candidates. W8's `m/nlist >= 32` rule applies to calibration candidates, **not** every manual feasibility build; it permits at most 28,125 lists with a 900K calibration sample (fewer if held-out rows are reserved). Record excluded candidates; do not silently relax the rule for high-`nlist` predictions. Sweep `nprobe` and rerank factors on tuning queries for matched-recall curves. Don't require the optimum to exceed `sqrt(N)` or to match the paper's 200K.
6. **Ablations:** W9 rebuilt/reuse/seeded training controls; W10 router modes on identical centroids, lists, PQ and heap; required W8 predictions vs. measured routing/scan cost and end-to-end latency.
7. **Final comparison (M5):** IVF with its W8-selected `nlist` and production router, the manual sweep optimum as a separately labeled control, and the vanilla-router configuration alongside where space allows, paired with tuned SPFresh on the default workloads and E1–E3. Freeze/report selection provenance and any declared manual fallback if calibration excludes candidates or fails; a fallback does not waive the W8 implementation/validation study.

### 4.5 Report figures (4-page budget)

1. Predicted vs. measured `T1`, `T2` and `T1+T2` over `nlist`, with selected/observed optima and end-to-end latency alongside (cost-model Figure 6 analogue); optional W9 quality/runtime panel.
2. Static QPS vs. recall@10: IVF (production and vanilla router) vs. SPFresh.
3. Update workloads: recall and latency per round, default and skewed streams (E1), with list-length summaries (E2).
4. W10 router-mode ablation (may be cut for space).
5. Table: build time, memory (including E3 budgets), disk footprint and update throughput.

## 5. Delivery order and status

| Milestone | Depends on | Exit condition |
|---|---|---|
| M0: contracts and scaffolding | — | W7 skeleton + W3 fixtures (**done**) |
| M1: build and routing | M0; M7 for production | Functional W1/W2 build/reload (**done**); initial SIFT1M 1024-list rebuilt-control build measured; W9 builder integration implemented; production: feasible large-`nlist` SIFT1M/SIFT10M builds |
| M2: query-only IVF | M1 | W4 fixtures (**done**); initial SIFT1M query point measured; production: SIFT1M/SIFT10M sweeps with telemetry |
| M3: mutable IVF | M2 | W5 fixtures (**done**); production: insert/delete measurements |
| M4: concurrent IVF | M3 | W6 fixtures (**done**); production: mixed-workload measurements |
| M6: optimized routing (S08) | M1 functional | Functional hnswlib/PCA work done; initial six-mode SIFT1M outputs agree; fixed-centroid matched-recall router measurements remain |
| M7: ANN K-Means (S09) | M1 functional, M6 functional | Required W9 mechanisms, existing scalar/bounded exact checks, native-builder integration and large-`nlist` quality/time measurements; no GEMM prerequisite |
| M8: cost model (S10; required, functional work implemented) | M1 production, M6, M7 | Predictor/tool/provenance fixtures implemented; qualified inputs, frozen real calibration and prediction-vs-measurement study remain open |
| M5: final comparison (S07) | M1–M4 production, M6, M7, M8 | Protocol and model/manual-control `nlist` selection provenance saved; paired, tuned SPFresh runs on default workloads and E1–E3 |
| M9: report | M5 | Figures from §4.5 and a 4-page write-up |

The paired SPFresh comparison runs only at M5. Before then, confirm that SPFresh builds and completes a SIFT1M smoke run on the target node, without recording comparison numbers, so M5 isn't blocked by baseline setup.

### Current functional boundary

* **S04:** immutable lists and query composition with batched reranking; persistence, CLI/query scripts and fixtures pass. Initial SIFT1M build/query run completed; tuned sweeps and production trainer qualification remain pending. See [s04-lists-query-components.md](s04-lists-query-components.md).
* **S05:** mutable lists/locator, private heap copies, tombstones, quiescent checkpoints and reopen; sequential mutation CLI and insert/delete scripts pass fixtures. See [MUTABLE_INDEX.md](MUTABLE_INDEX.md) and [s05-cli-and-mutation-harness.md](s05-cli-and-mutation-harness.md).
* **S06:** weighted and fixed 8+8 rounds with quiescent scoring; failure-path and sanitizer checks pass. See [s06-mixed-workloads.md](s06-mixed-workloads.md).
* **S08:** revised hnswlib patch series, frozen bottom-layer refinement,
  single-pass zero-in-degree reinsertion, PCA traversal and full-centroid rerank
  implemented and functionally verified. See
  [s08-native-routing.md](s08-native-routing.md). Superseded native-graph work is
  removed; six modes also ran on the SIFT1M smoke bundle, but matched-recall
  production component measurements remain pending.
* **S09:** ANN reuse/seeding, random-partition initialization, builder/cache/manifest
  selection and bounded diagnostic ablations implemented in Windows `ivf-only`;
  mounted verification passed (Release 11/11, native targeted 5/5, ASan/UBSan
  3/3 and scoped TSan race checks with disclosed runtime/deadlock-detector caveats).
  User native-copy handoff/rebuild and initial W9 1024-list SIFT1M run passed;
  the large-scale quality/time/downstream gate remains open.
* **S10:** W8 estimator/calibration/report plumbing are implemented with mounted
  small fixtures; user native handoff/rebuild passed 6/6 targeted checks and a
  frozen SIFT1M pilot passed its audit. Report-only shell integration gap, required
  qualified SIFT10M calibration/prediction study and M8 remain
  pending; see progress and [s10-cost-model.md](s10-cost-model.md). Manual sweeps are not W8.
* **Pending:** large-scale W9 quality/time/downstream measurements, large-`nlist`
  and real update/mixed qualification (M1–M4), S10/M8, then S07.

### 2026-10-09 SIFT1M run: what it closes

Evidence is under `/home/omega/bufANN-lindorm-ivf/build/wsl-sift1m-20261009-114223/`
(`plan.json`, `environment.sh`, `summary.json`, `verification.json`, logs and
native records); detailed interpretation is in [implementation-progress.md](implementation-progress.md).

* **Closed locally:** SIFT1M input/PQ/GT provisioning for the initial query set; the graph-free native build/query/reopen smoke using this repo's helpers/scripts; exact agreement between native and unchanged SPFresh scoring. No canonical Vamana index was needed.
* **Partially advanced:** W1/W2/W3/W4/W7 real-data integration and W10 mode/component loading. At 1024 lists, all eight trials produced identical neighbors with 91.07% Recall@10; native build was 412.30 s and vanilla median throughput was 1309.96 QPS. These are initial tuning-set measurements, not full milestone closure.
* **Not closed by that run:** W9, any large-`nlist` feasibility/benefit claim, training SSE/occupancy or router-agreement studies, SIFT1M insert/delete/mixed/checkpoint qualification, SIFT10M, matched-recall W10 speedups, the final CPU/cache protocol, SPFresh readiness/comparison, E1–E3 or the report. W8 was unimplemented at that boundary; its now-implemented functional work does not close real prediction validation. `exact` in these query records means exact **centroid routing**, not exact K-Means or exhaustive vector search.
* **New measurement caution:** fetch accounts for 87.3% of vanilla summed query-stage wall time (including contention), while PCA retains 28/128 dimensions. Report fetch/rerank separately alongside W8's paper-defined `T1+T2`; do not extend the predictor or infer PCA/high-`nlist` gains from this one setting.

The subsequent W9 reuse-seeded/random-partition run is saved separately under
`build/wsl-s09-sift1m-20261009-141638/`: 373.99 s build, 242.45 s training,
476.04 MiB peak build RSS, 92.42% Recall@10 and 486.02 QPS vanilla median.
All eight query outputs agree; final training has zero empty clusters and six
bottom-unreachable nodes. This closes native S09 real-data integration at 1024
lists, not the large-`nlist` gate or matched training-control study. The earlier
sample-initialized run is not its controlled speed/quality baseline. See progress
for occupancy, timing caveats and remaining experiments.

**Current scope done when** W1–W10 pass their functional and measurement gates, including W8/S10/M8 prediction validation, the M5 comparison and E1–E3 are complete, the §4.5 figures exist, existing BufANN/SPFresh entry points work unchanged, and another user can reproduce calibration, model/manual-control `nlist` selection, build, prepare and run from documented settings. Prediction misses are disclosed results; they do not waive the study or require an improved model. Add IVF commands to `README.md` once validated.

## 6. Future work

**Out of scope, not deferred:** 4-bit SIMD fast-scan PQ and residual PQ re-encoding.

**Scope history (2026-10-09):** the user temporarily deferred W8/S10/M8 and
subsequently restored it. The predictor, calibration/report plumbing and
prediction-validation figure are now required in §3/§4, not future work.

**Deferred:** crash-safe WAL and safe physical reclamation (summary §13.4).

ANN K-Means is no longer future work: W9/S09 is required in §3. Bounded exact
GEMM is optional later reference work, not a prerequisite for S09, large-`nlist`
experiments, S10 or project completion. WAL and reclamation remain deferred.
