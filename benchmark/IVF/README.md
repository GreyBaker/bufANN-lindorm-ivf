# IVF benchmark guide

These scripts prepare and run IVF query-only, insertion, deletion and mixed-update
workloads on Linux/WSL. Run the commands below from the repository root.

## 1. Compile the IVF tools

For the Ubuntu-packaged dependencies installed by the [quickstart](#quickstart-sift),
use a separate build tree and these paths:

```bash
export IVF_BUILD_DIR="$PWD/build/ivf-benchmark"
bash benchmark/IVF/build.sh \
  -DMKL_ROOT=/usr \
  -DMKL_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu \
  -DINTEL_OMP_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
```

The build uses installed MKL, Intel OpenMP and libaio, and writes the drivers and
diagnostic tools to `$IVF_BUILD_DIR/tests/`. Make runtime libraries discoverable
through the loader or `LD_LIBRARY_PATH`. `IVF_BUILD_JOBS` defaults to 4.
Ubuntu packages install under `/usr`; an empty `/opt` is normal. Only use
`/opt/intel/oneapi/...` paths if you separately installed Intel oneAPI, substituting
its actual MKL/OpenMP directories rather than a presumed version.
Routing fetches a pinned upstream comparison; the modified IVF headers are
vendored. See the [dependency guide](../../dependencies/hnswlib/README.md).

## 2. Configure a dataset

Configure `benchmark/datasets/<DATASET>.sh` as described in the
[dataset guide](../../README.md#2-configure-a-dataset). Stage matching vector and
query `.bin` files before query preparation. `DATA_TYPE`, `DIM`, `NPTS_FULL`,
`NPTS_BASE` and `UPDATE_POINTS` must match your inputs. The first `NPTS_BASE` rows
form the initial index; insertions use the remaining tail. Source-row IDs stay
consistent across IVF vectors, PQ codes, mutations and ground truth. There is no
requirement to use the same files or download pipeline as SPFresh.

## 3. Generate ground truth

For uint8 squared-L2 prefix/tail-update workloads, use the
[one-command GT workflow](#4-configure-and-run-all-four-workloads) after configuring the
actual update plan. The SIFT quickstart below integrates it in step 4. The legacy
helper below remains available for independently generated inputs:

```bash
bash benchmark/gen_deep_gt.sh sift1m
```

Preparation filters `GT_BIN_KCACHED` for active IDs. Alternatively, `IVF_GT_FILE`
can supply base GT for queries or post-update GT for sequential mutations; mixed
runs derive per-round GT. Insufficient candidates fail.

## 4. Supply PQ data

Set `IVF_PQ_PREFIX` to the prefix of existing `_pq_pivots.bin` and
`_pq_compressed.bin` files. Global, non-residual PQ8 codes must cover **all
`NPTS_FULL` rows**. To extend base-only codes, copy the pivots to a fresh prefix:

```bash
build_pq_codes_standalone TYPE DATA_BIN NEW_PREFIX ACTUAL_PQ_CHUNKS 0
```

Preparation never retrains PQ. Keep the IVF codebook unchanged across runs;
no DiskANN/Vamana graph is needed.

## 5. Prepare and run IVF

Choose a centroid count and resource budget (illustrative values below):

```bash
export DATASET=sift1m
export IVF_PQ_PREFIX=/absolute/path/to/existing-full-source-pq/prefix
export IVF_NLIST=1024 IVF_MEMORY_BUDGET_GB=4 IVF_BUILD_THREADS=20
export IVF_SAMPLE_RATE=1 IVF_TRAINER_MODE=lv-plus IVF_MAX_REPS=15
export IVF_INIT=random_partition IVF_TRAINING_BREADTH=64 IVF_SEED=0
export IVF_TRAINING_RERANK_FACTOR=3 IVF_TRAINING_REFINE_BREADTH=128
export IVF_OPTIMIZED_MODES=1 IVF_ASSIGNMENT_ROUTER_MODE=pca-refined
export IVF_BUILD_ROUTER_RERANK_FACTOR=3 IVF_PCA_VARIANCE=0.95 IVF_REFINE_EF=128
export IVF_QUERY_BREADTH=128 IVF_INSERT_BREADTH=64
export IVF_ROUTER_MODE=pca-refined IVF_ROUTER_RERANK_FACTOR=3 IVF_ROUTER_BREADTH=128

bash benchmark/IVF/sift/query_prepare.sh
IVF_NPROBE=16 IVF_RERANK_FACTOR=4 QUERY_THREADS=4 \
  bash benchmark/IVF/sift/query.sh

UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete.sh

UPDATE_PERCENT=10 UPDATE_BATCH=10000 bash benchmark/IVF/sift/update_prepare.sh
UPDATE_PERCENT=10 UPDATE_BATCH=10000 IVF_SCHEDULER=update_driven NTHREADS=16 \
  QUERY_THREADS=4 bash benchmark/IVF/sift/update.sh
```

Sequential mutations use one worker. Mixed rounds have equal insert/delete caps.
The shell default, `update_driven`, divides `NTHREADS` between mutation and query
workers and searches until mutations complete, with no query-ratio/count quota.
`QUERY_THREADS` controls post-update scoring, not that split.
Recall is scored after workers join, separately from concurrent query timing.

Query defaults: `IVF_NPROBE=1`,
`IVF_RERANK_FACTOR=2`, `QUERY_THREADS=1` and the profile's `RECALL_AT`.
The example explicitly selects full-base LV+ training and optimized final routing;
it does not change CLI/harness defaults. See the ablation settings below for other
training and router choices. The full-scale quickstart handles SIFT100M's pass cap.

Work files live in `eval_tempfiles/<dataset>/IVF/<workload>/`; reusable immutable
bundles live in `eval_cached/<dataset>/IVF/`. Runs retain logs, `result.json` and
`neighbors.bin`; mutations also retain `mutation_result.json`,
and mixed runs retain per-round results and a summary. Cache hits restore a build;
they do not measure a new one. Source, settings and tool identities are checked.

Existing paths, including failed runs, are not overwritten. Use a fresh
`EVAL_TEMPFILES` root to prepare again or `IVF_RUN_TAG` to repeat from the same base.
Keep inputs trusted and immutable. Only `hnswlib-0.8.0-stock1` bundles are accepted;
rebuild older bundles rather than relabeling them.

## Purpose

This backend provides a single-node, Lindorm-inspired IVF path alongside the
separate BufANN and SPFresh baselines. It combines centroid routing, existing
global PQ8 scanning and heap-backed vector fetching, with frozen routing across
updates. The experiments below isolate training reuse/seeding, LV/LV+ assignment,
router choices and centroid-count prediction before comparison with separately
obtained baseline measurements.
Functional tests and example settings do not establish production quality or
paper speedups. Reserve final queries until calibration and tuning are frozen.

## Ablation instructions

Use fresh paths, declared work/memory limits and tuning queries only. **Reserve
final queries until calibration and tuning are frozen.** Keep PQ, source rows and
unrelated settings fixed; repeat measurements and retain failures/exclusions.
Plans, identities and logs are retained; summaries are not production qualification.

### Settings and frozen routing

For bounded scheduler ablations, explicitly select `IVF_SCHEDULER=weighted` or
`fixed_8_8`. These retain the finite `QUERY_POINTS` override or legacy
`QUERY_RATIO`-based count (default 80). `fixed_8_8` requires `NTHREADS=16`.
The native CLI retains its legacy `weighted` default; the shell runner selects
`update_driven`. That mode rejects ratio/count/seed overrides rather than hiding
a quota. Its latency mean covers all queries; p99 uses a seeded reservoir of at
most 65,536 completion-order samples, labeled in the result. Actual query counts
and overlaps may be zero for very short batches. Already-admitted queries join
before checkpointing and scoring.

CLI/harness training defaults are `IVF_TRAINER_MODE=ann-reuse-seeded`,
`IVF_INIT=random_partition`, `IVF_TRAINING_BREADTH=64`, `IVF_SAMPLE_RATE=1`,
`IVF_MAX_REPS=15` and `IVF_SEED=0`.
The operational examples and quickstart explicitly override the trainer to
`lv-plus` while keeping `IVF_SAMPLE_RATE=1`: every initial-base row participates
in training. Sampling examples below are optional ablations, not quickstart steps.

`IVF_ASSIGNMENT_ROUTER_MODE` freezes the base/insert router (default `hnswlib`);
`IVF_ROUTER_MODE` overrides queries. `exact` enumerates centroids; `patched-full`,
`refined`, `pca` and `pca-refined` require `IVF_OPTIMIZED_MODES=1` at preparation.
Centroids, routers and final PCA stay frozen across updates. Centroid controls
`IVF_ROUTER_BREADTH`/`IVF_ROUTER_RERANK_FACTOR` are distinct from full-vector
`IVF_RERANK_FACTOR`. See native `--help` for more settings.
`IVF_BUILD_ROUTER_RERANK_FACTOR` sets the persisted base/insert centroid factor;
`IVF_ROUTER_RERANK_FACTOR` overrides the query centroid factor. Selecting LV+
alone does not enable optimized final routing; the recipe selects `pca-refined`
for both base/insertion assignment and queries, with optimized artifacts enabled
before preparation.

### Training ablations

```bash
source benchmark/datasets/sift1m.sh
python3 benchmark/IVF/qualify.py train \
  --data "$DATA_BIN" --dtype "$DATA_TYPE" --dim "$DIM" \
  --full-count "$NPTS_FULL" --base-count "$NPTS_BASE" \
  --trainer "$IVF_BUILD_DIR/tests/ivf_training_diagnostics" \
  --output "$PWD/eval_training/sift1m-comparison-01" \
  --modes exact,ann-rebuild,ann-reuse,ann-reuse-seeded,ann-reuse-count,ann-reuse-seeded-count,lv,lv-plus \
  --init random_partition --training-breadth 64 \
  --training-rerank-factor 3 --training-refine-breadth 128 \
  --nlist 256 --sample-rate 0.01 --iterations 3 --seed 0 --threads 4 \
  --memory-budget-gb 1 --exact-work-limit 20000000 \
  --assignment-work-limit 100000 --evaluation-rows 1000 \
  --evaluation-work-limit 300000 --routing-work-limit 1000000000 --timeout 1800
```

The modes compare exact double-L2 assignment, rebuilt HNSW, reused graphs with or
without seeding, nearest-one count controls, and LV/LV+. LV/LV+ share reused
full-coordinate graphs, initial-only refinement and 95%-variance PCA refit at each
joined assignment boundary. **LV+ changes only previous-assignment seeding**;
keep its other settings identical to LV. Training search settings are independent
of final routing. Moving PCA/refinement schedules are interpretations; fixed pass
caps and hnswlib substitution are not exact paper reproduction.

Modes use the same deterministic base sample in separate processes. Inspect SSE,
occupancy/recovery, connectivity, time, RSS and distance counts. SSE uses training
rows, not holdouts; objective work differs from traversal, which rebuilt HNSW does
not count. Exact training costs `sample_count * (iterations + 1) * nlist`
distances. If over budget, omit `exact`: bounded evaluation remains, but
exact-trained reference ratios are unavailable. Never bypass budget failures.

### Base-only frozen calibration

```bash
python3 benchmark/IVF/cost_model.py \
  --tool "$IVF_BUILD_DIR/tests/ivf_cost_model" --output /new/path/calibration-r10 \
  --data "$DATA_BIN" --pq-prefix "$IVF_PQ_PREFIX" \
  --data-type "$DATA_TYPE" --dim "$DIM" --base-count "$NPTS_BASE" \
  --sample-count 32768 --synthetic-queries 100 \
  --nlist-grid 256,512,1024 --topk 10 --target-recall .90 \
  --trainer-mode lv-plus --training-breadth 64 \
  --build-threads 4 --memory-budget-gb 4 --min-sample-per-centroid 32
```

Calibration uses disjoint seeded base sample/query rows and exact double-L2 GT,
never official queries. Predicted `T1+T2` covers routing and PQ scan/selection,
not ADC-table setup, fetching or exact reranking: **not an end-to-end optimum**.
Eligibility requires `sample_count >= nlist * min_sample_per_centroid`.
Default 32 is project policy, not a paper mandate or proof of adequacy;
inspect ratios/exclusions and test sensitivity with positive minima.

Prepare a full-base index from the frozen report:

```bash
export IVF_COST_MODEL_REPORT=/new/path/calibration-r10/report.json
unset IVF_NLIST
export IVF_SAMPLE_RATE=1 IVF_TRAINER_MODE=lv-plus IVF_INIT=random_partition
export IVF_TRAINING_BREADTH=64 IVF_BUILD_THREADS=4 IVF_MEMORY_BUDGET_GB=4
bash benchmark/IVF/sift/query_prepare.sh
```

Match report settings and source/PQ identities; explicit `IVF_NLIST` must agree.
Selection is recorded in `nlist_selection.json`. Freeze a report per recall/top-k
target before validation. Never edit reports, admit excluded candidates or refit
after validation. Manual sweeps do not validate predictions.

### Fixed-router diagnostics

Prepare a bundle with `IVF_OPTIMIZED_MODES=1`, then compare all six routers on its
unchanged centroids/PCA. Ensure probes fit the index:

```bash
python3 benchmark/IVF/router_sweep.py \
  --diagnostic "$IVF_BUILD_DIR/tests/ivf_router_diagnostics" \
  --index-prefix /path/to/index/prefix --query-file /path/to/tuning-query.bin \
  --data-type float --output /new/path/router-ablation --scope tuning \
  --nprobes 1,10,32 --breadths 32,64,128 --recall-targets 0.90,0.95,0.99 \
  --query-rows 100 --query-threads 1 --router-rerank-factor 3 \
  --memory-budget-gb 4 --evaluation-work-limit 100000000 \
  --repeats 3 --timeout-s 1800
```

This measures **centroid recall and router throughput**, not end-to-end ANN.
Repeats are averaged before selecting the fastest observed breadth meeting each
minimum recall target; this is not exhaustive tuning or identical-recall
interpolation. Timed search includes centroid-match scoring but excludes exact
GT/loading/checks. Warmups and hashing warm caches. Exact reference enumeration
costs `query_rows * nlist` distances; retain connectivity and missed targets too.

### Repeated query sweeps

```bash
DATASET=sift1m bash benchmark/IVF/sift/query_sweep.sh \
  --tag repeated-01 --nprobes 8,16,32 --rerank-factors 2,4,8 --threads 1,4 \
  --repeats 3 --warmups 1 --memory-budget-gb 4 --buffer-frames 64 --timeout 1800
```

Trials reuse the index/PQ/GT; records go to `query/runs/`, plans/summaries to
`query/sweeps/<tag>/`. Separate-process warmups do not warm later buffer pools.
`--cpus` pins without CPU isolation. Recall gates use percent Recall@K;
run-percentile aggregates are not pooled query percentiles. Gates do not establish
matched-recall baseline comparisons.

### Measurement cautions and verification

- Query timing excludes preparation/open/checks/scoring/export; lifetime RSS does
  not. Build timing includes router reload, not Python hashing/cache restoration.
- Cache hits measure restoration, not builds. Hashing/open/copies warm caches;
  heap reads count buffer-pool reads, not device I/O. No cold-device claim is made.
- Memory budgets guard workspace estimates, not hard RSS caps. Record machine,
  storage, tool/runtime and OpenMP/MKL settings; advance `IVF_CACHE_TAG` or change
  cache roots after untracked environment changes.
- Timeouts retain partial evidence; failed gates exit nonzero. No resume or crash
  recovery is provided. Checkpoints require joined workers and are not durable
  transactions; deleted IDs cannot be reused.
- Heap writes retain direct I/O, use aligned page snapshots and require complete
  page writes. A failed/short write makes that session unusable and prevents a
  successful checkpoint; it is not automatically repaired. If a slot mismatch
  is confirmed by a read-only inspection of the heap, preserve the failed
  artifacts and reconstruct into fresh work/cache roots after fixing the cause.
  Rebuilding the driver does not repair an already inconsistent heap.
- Immutable streaming heap construction fills each page before advancing, rather
  than cycling over 256 partially filled pages with the default 64 buffer frames.
  This avoids bulk-load page thrashing without enlarging the query buffer budget.
  Individual appends and mutable insertions retain their existing allocation
  policy. Source-row IDs and file framing are unchanged; physical slot placement
  can differ. Rebuild the builder and its library for new preparations; this
  cannot accelerate a process already running, and there is no partial-build
  resume or supported heap-only reconstruction command.
- Native builds log `[IVF build]` phase boundaries. At one million or more rows,
  training also logs graph/PCA, assignment and mean timings per pass, and list/
  heap streaming reports every million completed rows. These diagnostics are
  included in build wall time; row counters are not durable checkpoints.
- `IVF_BUILD_THREADS` controls both training assignments and final posting-list
  routing. A sequential reader feeds bounded blocks to frozen-router workers;
  histogram/PQ grouping and heap construction remain serial. Completed row logs
  occur at joined block boundaries (approximately every million routing rows).
  Effective routing workers are capped by available blocks and reported in build
  JSON. Worker scratch, input buffers and stack allowances are included in memory
  admission. Standalone list-library callers remain serial.
- Saved posting-list buffers and the final router are released before heap
  population. The finished heap's locator arrays are released before final
  hashing.

Dependency-isolated functional checks:

```bash
cmake -S tests/ivf -B build/ivf-contracts -DCMAKE_BUILD_TYPE=Release \
  -DIVF_BUILD_ROUTING=ON
cmake --build build/ivf-contracts -j4
ctest --test-dir build/ivf-contracts --output-on-failure
```

Isolated drivers only validate inputs; workloads/calibration require the full Linux
build. For full tests, enable `-DIVF_BUILD_CONTRACT_TESTS=ON`, build test executables
and run CTest there. Scalar sanitizer results establish neither SIMD safety
(upstream has known prefetch issues) nor full-system race safety. Sampling,
prediction accuracy and comparisons with external SPFresh results still need
separate studies.

The full Linux build includes `ivf_direct_io`, a synthetic persistence/fault test
that enforces strict write alignment even on permissive filesystems. It checks
eviction, explicit/background flushing, reopen, failed-checkpoint admission,
write errors, short writes and interruption retries without requiring an ext3
server. These fixtures do not establish crash durability or full-scale results.

# Work plan: update performance and measurement alignment

**Status: U0 local inventory recorded; U1 initial increment under qualification.
Remote comparison admission remains pending; U2 is partial, U3/U4 passed their
scoped implementation gates; U5 and U6's input/timing segment passed their local
functional gates. Remaining U6 work and U7–U8 are pending.** This plan addresses serial
sequential insertion, suspected mutable-page churn, globally serialized vector
fetching, and measurement discrepancies against local BufANN/SPFresh. It supplements
earlier qualification work without closing W1–W10 requirements or establishing
production readiness. Operational commands still describe current behavior, not
the proposed interfaces.

## Scope and invariants

- Change only IVF, its harness/tests and this README. Local BufANN/SPFresh sources
  are read-only references: do not edit, build, run or deploy their infrastructure.
  Preserve independently obtained baseline results, failed evidence, and existing
  source/query/PQ/GT files. Do not overwrite, repair in place, or relabel old runs.
- Preserve full indexed/live populations, original source-row IDs, all final
  queries, matching exact GT, frozen centroids/routers/PQ, +16 list growth and
  swap-with-last deletion. No SPFresh splitting/reassignment, retraining on updates,
  padded truth, omitted queries or final-query-driven tuning.
- Lindorm §4.4 specifies routing, nearest-centroid assignment, quantization and
  list append, not one insertion worker or 64 local buffer frames. Its WAL-based
  commitment is not implemented by our checkpoint mechanism. Do not claim that
  foreground publication/checkpoint completion is a Lindorm-equivalent transaction.
- Retain full-training LV+, optimized routing, 200K centroids, PQ64 and declared
  ANN settings; preserve `hnswlib-0.8.0-stock1` compatibility. Exact GEMM remains
  optional; sampling adequacy and deferred PQ/scanner changes remain open.
- Do not modify an active process or its binaries/artifacts. New code/settings
  require new run identities and unused evidence roots, without timestamps or
  cleanup of previous attempts. This plan authorizes no automatic commit,
  deployment, restart, privileged action or baseline rerun.

## Milestones and acceptance gates

### U0 — Freeze evidence and comparison contracts (first)

Inventory the effective current IVF and available baseline settings: source/query/
GT/PQ and binary/library identities, CPU/storage/RAM, foreground/background workers,
cache capacities, input staging, maintenance, scoring and timing scopes. Source
defaults are not evidence of actual server settings. Use read-only process/thread
and I/O samples to locate `D` waits in opening/copying, hashing, insertion, flushing
or checkpointing; CPU percentage alone cannot establish the cause. Invasive tracing
of the active run requires separate approval.

Define separate foreground-insert, completion/maintenance, standalone post-update
query and concurrent mixed comparisons. Declare total CPU/RAM allowances and cache
protocols; equal frame counts or parameter names do not establish equivalence.
Historical unmatched runs characterize their configurations, not matched-resource
or paper-speedup claims.

**Gate:** retained evidence inventory, explicit metric definitions, unresolved
provenance gaps and a contract for each comparison. Never fabricate missing metrics.

### U1 — Add phase/progress and diagnostic visibility (before optimization)

Instrument opening/private copies, validation/hashing, source staging, insertion,
checkpointing and post-update search. Add bounded-frequency completed-operation
counts, elapsed time and rates, plus routing/PQ, heap-wait and storage diagnostics
where measurable. Distinguish dirty evictions/page reads/writes/flush bytes from
device IOPS. Record effective workers, pool capacity and timing policies. Claimed
or submitted work is not completed publication; ETA is advisory, not persistence.

Files: `tests/ivf_cli_execution.cpp`, `src/bufann/ivf_heap.cpp`,
`benchmark/IVF/common.py`, `harness.py` and `mixed.py`.

**Gate:** counts advance only after success; terminal and owned logs receive
incremental output; failures retain phase/count evidence without a successful final
result. Scratch/output frequency are bounded; measure and disclose overhead rather
than silently subtracting it from timings.

### U2 — Resolve resource defaults and read-failure handling

Use the quickstart's **262,144 frames / 1 GiB page payload** as the full-scale
candidate, subject to composed admission. Make prepare/run/repetition recipes
agree, including the full-scale sweep that still requests 64 frames. Retain small
pools for pressure fixtures; do not blindly make a 1-GiB pool the default of a CLI
with a 1-GiB total budget. Document effective profile defaults and reject incompatible
budgets before execution. Consider separating runtime capacity from build identity
where safe, without altering saved manifests or calling restoration a fresh build.
Ensure GT selection/coverage succeeds before the workload loop; a separately shown
GT command must not leave preparation runnable before that gate.

Audit reads alongside strict writes: a real error or unexpected short read of a
materialized page must not silently become zero coordinates. Preserve legitimate
newly grown-page handling and interruption retries; failures must poison the
session and prevent successful checkpoints. Do not reproduce the separate BufANN
checkout's ignored-write behavior.

Files: `benchmark/IVF/settings.py`, `harness.py`, `mixed.py`,
`tests/ivf_cli_execution.cpp`, `src/bufann/inplace_backend.cpp` and `ivf_heap.cpp`.
Shared-backend fixes need scoped compatibility tests and must preserve non-IVF
callers' allocation/search policies.

**Gate:** resource examples agree; estimates account for frames/metadata and future
worker scratch; synthetic read/write/short-I/O faults fail closed. A larger pool
alone is not evidence that allocation is fixed.

### U3 — Make mutable IVF allocation page-local (depends on U1/U2)

Reuse the existing page-filling seam for IVF mutable insertion, leaving generic
backend/BufANN callers on their policy. Preserve IDs, coordinates, PQ codes, list
ownership and readiness/publication ordering; only physical slot placement changes.
Cover deletion-created holes, aborted insertions, page boundaries, growth and reopen.

Files: `src/bufann/ivf_heap.cpp`, `ivf_index.cpp` and the backend allocation seam.
Prefer this bounded change over buffered I/O or worker-local allocators, which need
separate justification and qualification.

**Gate:** logical results/recovery invariants remain correct; physical I/O and dirty
eviction evidence demonstrate locality improvement. Evaluate allocation at fixed
pool sizes on synthetic/independent tuning inputs before combining thread changes.

### U4 — Remove unnecessary global fetch serialization (depends on U1/U2)

Review `_copy_mutex` in `ivf_heap.cpp`: it covers whole read batches and blocks
independent fetches and mutation publication. Design narrower ledger/lifecycle
protection, per-page guards and bounded aggregate pin admission. Immutable query
reads must not require a global exclusive I/O lock; mutable reads must still reject
absent/retired IDs and copy the correct coordinate record.

Do not simply delete the mutex or use unrestricted shared locking: multiple batches
can jointly exhaust shard frames, especially with tiny pools. Preserve duplicate-
page handling, publication ordering, cancellation and quiescent checkpoints.
Worker-local pages or asynchronous I/O are later alternatives, not prerequisites.

**Gate:** independent fetches demonstrably overlap; tiny/skewed pools have no pin
leaks/deadlocks; concurrent insert/delete/query tests and existing same-ID/fault
protections pass. Measure scaling instead of inferring it from worker creation.

### U5 — Add bounded parallel sequential insertion (depends on U2–U4)

Use existing joined-worker/failure-propagation patterns and concurrent
`MutableIndex::insert`. Add an explicit insertion-worker setting, record effective
workers and retain one-worker mode as a control. Workers need independent input
state and routing/PQ scratch. Admit buffers/stacks/work before launch; validate IDs,
coordinates and limits before mutation. On failure stop admission, join admitted
workers and publish no successful checkpoint/result for a partial workload.

Keep delete-only behavior and mixed scheduling unchanged in this milestone. Do
not reinterpret `NTHREADS` or silently alter half-mutation/half-query allocation.
Parallel order can differ; require logical correctness and a declared tie policy,
not byte-identical parallel checkpoint families. Heap/list contention may limit
speedup even after adding workers.

**U5 implementation segment:** native `--insert_threads 1..256` applies only to
`insert_only`/`search_after_update`; default one is retained. The sequential harness
passes independent `IVF_INSERT_THREADS` (default one), never `NTHREADS`. Query-only,
delete-only, mixed and validation-only calls reject the native option. Effective
workers are capped by input count (zero for empty input); one worker runs on the
caller thread. Each worker has a separately opened input stream and row buffer;
existing `MutableIndex::insert` owns independent routing/PQ scratch. Coordinates,
ID framing/uniqueness/ranges, live-ID and insert/delete-disjointness checks run
before mutations. Retired-ID no-reuse is also enforced by insertion reservations;
such failure can leave earlier admitted mutations in the failed-work heap.
Readers/buffers are allocated
before the delete phase; all insertion threads launch behind a gate before any
insertion admission. Callback/launch failure stops new admission, joins admitted
workers, propagates the first captured error and publishes no successful
checkpoint/result. Admitted inserts may remain in the owned failed-work heap.

Workspace admission includes worker streams/rows/mutation scratch, conservative
Linux stack allowances (at least 8 MiB per launched worker) and worst-case +16 list
growth/replacement storage before mutation. These are conservative estimates, not
measured peak RSS, a full-scale guarantee or crash durability. New metrics declare
requested/effective workers, peak admitted callbacks (including lock/I/O waits),
worker/growth allowance and pre-launch runtime thread count. The insertion timer
includes worker launch/join, row reads, successful mutation and periodic progress;
source identity/coordinate preflight, reader setup, checkpoint and GT are outside.
Delete timing/scheduling and mixed worker allocation are unchanged. Parallel
publication order is unspecified; exact distance/ID tie ordering is unchanged.

Scheduler qualification passed 50 normal, ten ASan/UBSan and ten TSan executions
in `build/u5-check/helper-run02/`, with halt-on-error, no suppressions, a synchronized
TSan negative control and three detected deliberate-race trials. Four OFF
scheduler/CLI/Python/CMake tests passed in `build/u5-check/off-run02/`; the earlier
OFF attempt omitted the CLI contract executable and is retained. The first native
attempt failed to compile a new thread-count diagnostic, before runtime tests;
it is retained in `build/u5-check/native-run01/`. The fix uses the existing Linux
directory-iteration pattern without changing mixed execution.
The second native attempt built successfully and passed the scheduler and all six
float worker/pool insertion/reopen controls, but its later 20 MiB serial resource
control was below the existing composed-open estimate. That fixture-only allowance
is now 0.08 GiB: above the recorded ~43–45 MB serial estimate and below the
~112–113 MB eight-worker estimate, without changing ANN/population settings or
weakening native resource admission. Failed `build/u5-check/native-run02/` is
retained. The corrected fixture passed for all three dtypes using unchanged
runtime binaries and fresh inputs/output roots in `build/u5-check/fixture-run01/`.
Final native qualification passed in `build/u5-check/native-run03/summary.json`:
both targeted tests, twenty scheduler repetitions, three all-dtype native fixture
repetitions and all fifteen broader regressions. This covers scheduler bounds,
reader/routing/publication/progress exception seams and partial-launch joins;
all-dtype native 1/2/8-worker, 2/64-frame controls; full-population exact query/reopen
oracles; empty inputs; resource/flag refusals; failed retired-ID insertion with no
successful publication; and unchanged source/base artifacts. Final reconciliation
is retained in `build/u5-check/final-run01/summary.json`.

U5 passed its **local implementation/functional gate**. Its scheduler-only sanitizer
tests do not qualify the full native routing/HNSW/SIMD path; the 56-insert synthetic
controls are not a full-scale insertion throughput or resource claim. Retain one
worker as the default/control. Selection of an operational parallel worker count
remains pending independent, resource-matched qualification (U2 remains partial).
No worker default is promoted, no failed evidence is discarded, and nothing is
deployed. To exercise the opt-in on a fresh already-prepared sequential insert
work tree, use `IVF_INSERT_THREADS=8` with the existing insertion run recipe; native
calls use `--insert_threads 8`. Query/delete-only/mixed harness runs do not use this
setting; `NTHREADS`, mixed allocation and frozen ANN settings are unchanged.

Files: `tests/ivf_cli_execution.cpp`, native CLI validation,
`benchmark/IVF/harness.py` and `tests/ivf/` mutation/scheduler/integration fixtures.

**Gate:** one/multiple-worker correctness, launch/operation failure joins, accurate
counts/budgets, checkpoint/reopen equivalence and resource reporting. Choose the
operational worker count using independent qualification, not final-query results.

### U6 — Align staging, timing and cache protocols (depends on U0/U1/U5)

Name insertion input policies explicitly. BufANN stages insert coordinates and
local SPFresh loads full vectors before timing; IVF currently reads coordinates
inside its loop. A pre-staged mode must admit its memory and report loading time;
a bounded streaming mode retains and labels source I/O in its wall-time rate.
Do not normalize throughput by subtracting overlapping stage times. Preserve
legacy metric definitions or version changed semantics explicitly.

Separate foreground publication, checkpoint/flush completion and total workflow
cost. SPFresh's queue drain is not automatically filesystem durability. Add optional
same-process, base-derived warmup with an explicit policy and zero-warmup control.
Do not tune on final queries or claim separate-process application-cache warmup.

Preserve current copy/hash/checkpoint-reopen isolation as a safety mode. Evaluate an
optional persistent mutable session for post-update queries/mixed rounds only if
immutable source protection, required checkpoints and failed evidence remain intact.
Do not weaken recovery checks to improve QPS. Reject unsupported sequential round-
cap expectations or implement them explicitly.

**Gate:** named/versioned metric scopes, retained staging/completion costs and
declared cache/process boundaries. GT preparation/scoring stay outside timed
queries. Persistent versus reopened execution is an explicit experiment, not a
silent reinterpretation of historical measurements.

**U6 input/timing segment:** sequential native insertion now names
`--insert_input_policy streaming|staged`, default **streaming**. The sequential
insert harness forwards `IVF_INSERT_INPUT_POLICY` with the same default; query,
delete and mixed harnesses do not apply it, and those native workloads reject the
option. Streaming retains the U5 foreground denominator including bounded,
worker-owned source-row reads; `insert_timing_contract_version=1` labels that
legacy scope. Opt-in staged mode validates/loads **all requested insert rows**
before mutation, admits their exact unpadded coordinate payload plus allocation
allowance alongside workers/list growth, then shares read-only staged coordinates
until joined insertion completes. It uses version 2 and excludes source-row reads
from `insert_time_s`; worker launch/join, publication, heap I/O and progress remain
inside. No overlapping subinterval is subtracted from either denominator.

Staging is not full-source preload and is not automatically equivalent to another
system's input policy. Full-source integrity checks before/after updates and the
separate verified checkpoint remain required. `insert_staged_coordinate_bytes`
declares payload, not total workspace/RSS. `insert_input_preparation_time_s` aliases
`coordinate_validation_or_staging_time_s`; **do not add both**. Validation/loading,
foreground publication and checkpoint/flush-copy-verification costs remain
separate. Checkpoint completion is still not a WAL/crash-durability claim.

`driver_workflow_time_s` directly measures driver dispatch through completed work
(including applicable open, validation, staging, checkpoint, GT load/scoring and
neighbor export), before canonical-record publication. It excludes argument parse,
process startup/exit and the final canonical-record write; it is not a sum of
potentially nested stage metrics. Query foreground boundaries/recall are unchanged;
GT preparation/loading/scoring remain outside timed queries. New cache/process
labels distinguish the existing native combined live-session query from a newly
opened/private-copy query; open validation may touch pages and OS caches are not
controlled. Explicit warmup remains zero in this segment.

The sequential harness retains `execution_protocol.json` adjacent to unchanged
canonical native metrics. It measures parent-side process wall costs (spawn/wait,
native record output and log streaming) and a separate run-entry-to-native-result-
validation workflow wall interval. The latter excludes preparation scripts, the
protocol write and outer-shell exports. Mutation/checkpoint then query remain
**two distinct driver processes** with a fresh private query copy; no application
cache is claimed to transfer across them. Query-only runs have one driver process.
The protocol file is no-overwrite evidence and is written only after successful
native query-record validation, never a fabricated query-QPS/recall record.

The staging helper passed forty ordinary and ten isolated ASan/UBSan executions
with halt-on-error in `build/u6-input-check/helper-run01/`; this qualifies only the
std-only sizing, validation/order and callback-unwind seam. Five OFF
staging/scheduler/CLI/Python/CMake checks passed in `build/u6-input-check/off-run01/`.
Native qualification in `build/u6-input-check/native-run01/summary.json` passed
both targeted tests, twenty helper repetitions, three all-dtype native policy
fixture repetitions and all seventeen broader regressions. This covers all
requested synthetic insert rows, full exact query/reopen equivalence across
streaming/staged input and 1/8-worker 2/64-frame controls, empty inputs, memory and
flag refusals, nonfinite-row failure before mutation, and process/timing/protocol
collision evidence. Source/base fingerprints and canonical native metric equality
were preserved. Final reconciliation is retained in
`build/u6-input-check/final-run01/summary.json`.

The **input/timing/process-evidence segment passed its local functional gate**,
not a full-scale performance/resource or native ANN/HNSW/SIMD/race gate. No
input/worker default is promoted; no baseline infrastructure, ANN settings,
populations, final queries or remote process are changed. U6 as a whole remains
incomplete: optional base-derived same-process
warmup, persistent-versus-reopened experiments, sequential round-cap handling,
broader observability and remote comparison admission are not closed here.

### U7 — Qualify scoring and cross-system interpretation (depends on U0)

Establish an independent named scorer for completed live states. Record percentage
versus fraction units and chosen-ID versus boundary-tie policy. Local SPFresh
mutation scoring matches distances with a tolerance; its separate query helper and
IVF/BufANN use ID-based truth expansion. Ten-column derived truth is not boundary-
complete, and these policies must not be described as identical.

Validate with synthetic ties/duplicate distances and independent exact live-state
truth. Boundary-complete GT or exact-distance auditing requires an explicit
implementation/compute budget, not padding or automatic runtime fallback. Rescore
existing baseline artifacts read-only where sufficient; otherwise mark the proposed
comparison unavailable, without modifying/rerunning baseline infrastructure.

Keep IVF update-driven, BufANN weighted/quota and SPFresh asynchronous-search
workloads separately labeled. Equal mutation totals are not equivalent concurrent
workloads. Independently obtained measurements need not share identical caches or
repetitions, but their provenance and policies must support the claimed comparison.

**Gate:** reference scorer fixtures pass; compared reference scores use the same
units/tie rules and declared workload/timing contract. Original scores/files remain
unchanged alongside any new interpretation.

### Execution record — initial U0/U1 increment

U0's local source-contract inventory is retained with fingerprints under
`build/driver-progress-check/`. User-reported DB-server query aggregates are
retained in `build/driver-progress-check/remote-query-observation.json`: 10,000
queries, eight workers, approximately 685 seconds wall time; fetching averages
536.837 ms/query and 98.04% of summed stage time, versus 547.557 ms total average
latency. These are per-query wall times including waits, not CPU measurements or
additive run durations. Approximately 5,368 seconds of accumulated fetching can
occur within that concurrent run. This supports prioritizing fetch contention/I/O
diagnostics but does not separate `_copy_mutex` wait from device/frame waits.
Raw metrics, effective buffer capacity/binary identities and insertion PID/I/O
samples are still unavailable; do not equate the two observations' configurations.
U0 is not complete and no matched-resource comparison is admitted yet.

| Contract | Current IVF | Local BufANN | Local SPFresh |
|---|---|---|---|
| Sequential insertion workers | 1 | `INSERT_THREADS`, defaults to `NTHREADS` | `SPFRESH_INSERT_THREADS`, defaults to `nproc`, plus maintenance pools |
| Cache capacity | Quickstart explicitly 1 GiB; fallback still 64 frames | Default 1 GiB page pool | RocksDB default 3 GiB block-cache capacity plus write buffers |
| Insertion source coordinates | Streamed inside foreground timer | Staged before timer | Full source loaded before timer |
| Post-update query process | Checkpoint/private-copy reopen | Live process/cache retained, warmup supported | Live process/cache retained |
| Mixed scheduling | Update-driven dedicated workers | Weighted finite quota plus maintenance | Separate update pools and asynchronous search batches |
| Native recall | Percent, ID-based GT expansion | Percent, ID-based GT expansion | Mutation: fraction/distance matching; query helper: fraction/ID matching |

These are source contracts, not confirmed settings or performance of historical
remote baseline runs. Cache capacities refer to different data structures, not
equal total RSS or equivalent memory allowances.

The first U1 implementation adds `[IVF driver]` phase boundaries and sequential
completed-count/rate reporting, heap page/dirty-eviction/flush counters, and explicit
source-validation/staging timing fields. The Python harness flushes each received
line to the terminal and owned log. Periodic reports require at least one second
and 1,024 additional completed operations; they add no heartbeat thread and may
stop while a native operation is blocked. Phase failure logs retain the last
successful count; final canonical success metrics still require workload success.

The initial native scalar-routing increment passed eight scoped CTests and ten
reporter repetitions. Its retained synthetic insertion fixture completed 8,128
inserts; an injected page-write error reported eight completed inserts, retained
the partial private heap and published no success result/checkpoint. The first
fixture attempt correctly failed the private/checkpoint path-overlap guard and is
preserved separately. Evidence: `build/driver-progress-check/run01/final-summary.json`.
Reporter-only ASan/UBSan passed; Windows Python helpers passed five tests with one
Linux-only skip. Windows C++ qualification is blocked by an unavailable compiler.
These checks establish neither full-scale overhead nor SIMD/race qualification.

The next U1 increment adds `fetch_adapter_mutex_wait_us_total` and
`fetch_backend_pin_us_total` to standalone/post-update query records. These are
**subintervals of `fetch_us_total`**, not extra stages to add to the existing stage
sum. Divide each by the scored query count to obtain its per-query average. Mutex
timing measures acquisition wait; backend pin timing includes lookups, frame waits,
eviction and read I/O, not pure device time. The remaining fetch time includes
coordinate copies/guards and other adapter overhead. No globally shared counter
delta is used for per-query attribution; the synchronization itself is unchanged.
Deterministic timed-acquisition helper tests pass under ASan/UBSan. Native
scalar-routing wiring passed nine scoped CTests and ten repetitions each of the
reporter and acquisition helpers in `build/driver-progress-check/run02/`. The
8,128-insert and injected-write-failure checks passed again, including partial
completed counts, no success publication and unchanged immutable inputs.
The isolated acquisition helper also passed Clang 20 TSan with a synchronized
negative control and three detected deliberate-race trials. No suppressions or
global ASLR changes were used. This is helper-only evidence, not a resolution of
the existing HNSW construction findings or native heap/SIMD race qualification.
The dependency-isolated OFF attempt in
`build/driver-progress-check/helpers-off-run01/` exposed a missing explicit include
directory on the timing-test target; that failed evidence is retained. After the
registration-only fix, fresh OFF checks passed three tests in
`build/driver-progress-check/helpers-off-run02/`; native ON registration and CMake
scope checks passed four tests in
`build/driver-progress-check/registration-run01/`. The latter records final source
fingerprints and confirms all runtime sources still match the nine-test native
attempt. No full-scale overhead, SIMD or full-system race qualification is claimed.

Existing foreground timers and source-I/O inclusion remain unchanged. Periodic
reporting is inside those timers; boundary reporting is outside. Heap counter
deltas exclude source reads and checkpoint file-copy/hash/metadata I/O; mixed
foreground counters include concurrent queries. These are backend page operations
and accumulated wait times, not device IOPS or CPU time. New fields are additive,
not normalized baseline throughput or durability claims.

This increment labels opening/private-copy/verification and checkpoint/flush/copy
as combined phases; it does not yet isolate their internal substeps or add periodic
mixed/query counts, per-insert routing/PQ/lock histograms, or a full-scale overhead
qualification. U1 remains partial; U2 has started with the scoped changes below;
U3–U8 are pending. Allocation, worker counts,
cache budgets, scoring, source IDs and ANN settings are unchanged by this increment.

The first U2 increment rejects synchronous `pread` errors/short/zero reads instead
of padding them with zeros, retries `EINTR`, retains a terminal pool failure and
releases the failed loader's pin while waking waiters. Newly grown sparse pages
remain valid: `grow_heap` extends the file before publishing them, so their full
zero-page reads need no EOF exception. IVF single-coordinate read failures also
poison its heap state and refuse a checkpoint. Allocation/search policies and
the batched-read implementation are unchanged.

The Python orchestration now shares an early per-frame budget lower-bound gate;
the full-scale repetition recipe matches the 262,144-frame candidate. The
quickstart selects/checks GT before entering any workload preparation loop, with
separate assignment/export so coverage/generation failure stops preparation.
Nine Python helper tests passed on Linux; Windows passed seven with two Linux-only
skips. The first native synchronous-fault invocation passed, but its tenth repeat
failed a fixture assertion that sampled an optimistic contender pin twice; broader
regressions did not run. That evidence remains in `build/resource-read-check/run01/`.
The fixture now captures a single observation and distinguishes a contender's
arrival from confirmed condition-variable parking. Fresh scalar-native qualification
in `build/resource-read-check/run02/summary.json` passed the fault suite's initial
invocation plus 100 repetitions and all eleven broader regressions. These cover
heap/query/mutable-session persistence, strict writes, CLI/query/mixed orchestration,
preparation guards and GT management. Original `run01` failure evidence is retained;
no baseline workload was run. No batched-read, SIMD, race or full-scale safety claim
follows from this synchronous-path result.

**U2 remains partial:** the initial synchronous-path increment did not validate
libaio completion byte counts. The subsequent prerequisite segment below adds and
qualifies exact completion/error/draining behavior in the stated scope; it is not
a full-system storage-safety result. The composed-resource estimate,
full-scale memory/overhead evidence and runtime/build-cache identity separation
also remain separate qualification/design work. Do not promote this increment as
a complete storage-safety or full-scale acceptance result.

**Segmented continuation through U3/U4:** an existing, uncommitted exact-AIO
implementation is being preserved and qualified as the read-safety prerequisite.
It validates completion identity/bytes/secondary status, drains accepted partial
submissions, retries interruption and synchronously destroys a context on unsafe
collection failure before unwinding buffers. If quiescing the kernel fails, it
fails-stop rather than freeing buffers still owned by the kernel. Legacy readers
keep their existing behavior; the in-place pool opts in. Pool callers now copy
registered contexts while holding the map lock, avoiding references invalidated
by concurrent registration. Batch-copy exceptions poison the IVF heap state.
The exact-AIO helper passed ten isolated ASan/UBSan executions. The first native
build exposed the robin-map iterator mutation API mismatch; its failed evidence
is retained in `build/u3-u4-check/aio-run01/`. After correcting that assignment,
Native prerequisite qualification in `build/u3-u4-check/aio-run02/summary.json`
passed three targeted tests, ten repetitions each and all eleven broader
regressions. Both original failure evidence and legacy-reader scope are retained.
This supersedes the
unimplemented-reader audit above only within the eventual qualified scope.

The major U3/U4 risks are read-failure unwind/cancellation, aggregate pin exhaustion
in tiny or skewed shard pools, lock-order/deadlock/starvation, and mutable visibility
or record-reuse errors. Qualification must demonstrate safe bounded overlap, not
just worker creation. Shared-backend scope, full-scale memory/I/O behavior, existing
HNSW sanitizer findings and unmatched baseline/GT provenance remain acceptance
risks. Checkpoints are still not WAL/crash durability; no live process or immutable
artifact is replaced by these segments.

**U3 segment:** mutable `HeapStorage::prepare_insert` now opts into the existing
caller-serialized page-filling seam; generic `append`/backend allocation remain
round-robin. It is still a private, graph-free IVF heap with its own buffer pool,
not a graph rebuild or a new immutable format. Aborted retries keep the same slot;
logical retirement keeps committed records/IDs and creates no reusable physical
hole. The new fixture compares 8,192 complete rows across float/int8/uint8 at fixed
64/256-frame pools, inspects physical owner placement, checks all coordinates,
growth, retained deletion and aborted retry, and reports foreground backend
page-read/dirty-eviction deltas separately from elapsed time. Its round-robin
control is the local heap library, not a baseline benchmark execution. Native
qualification in `build/u3-u4-check/u3-run01/summary.json` passed four targeted
tests, ten repetitions each and all eleven broader regressions, including
checkpoint/reopen. Retained measurements in
`build/u3-u4-check/u3-measurement-run01/summary.json` show at 64 frames: float
backend reads 8,192 → 1,171 and dirty evictions 8,128 → 1,107; int8/uint8 reads
8,192 → 274 and dirty evictions 8,128 → 210. The 256-frame controls also passed
and show smaller gains. These are synthetic backend page operations, not device
IOPS or a predicted full-scale speedup.

**U4 segment:** coordinate fetches now take short ledger snapshots/revalidation
locks, perform batched buffer-pool I/O without that lock, and copy under per-page
read guards. Mutable preparation has a separate serialized writer lock and does
not hold the ledger lock during page work. This depends on the current no-reuse,
no-relocation contract for published/retired IDs; it must not be transplanted into
the reference repo's reclaiming/tag-reinsertion behavior.

All read/write page work shares FIFO aggregate pin credits: capacity is the
smallest shard's frame count minus one (at least one); each query batch uses at
most half that capacity (at least one), capped at 256. Logical requests, including
duplicates/missing IDs, reserve conservative credits. Frames are destroyed before
their lease; a terminal failure wakes queued admissions. Tiny pools intentionally
serialize rather than exhausting frames. This changes physical fetch batching,
not the ANN candidate budget, probes, rerank count, workers or scoring.

The backend review also found stale loading-frame pin/reuse and counter-reset
races hidden by the old adapter serialization. Loading waiters now retain their
pin under the page-table lock; flushing waiters reclassify without retaining an
update pin. Loader metadata/ownership is initialized before table publication,
and ownership is additive so late failed optimistic pins are not overwritten.
Generic allocation/search policies are unchanged. `fetch_pin_admission_wait_us_total`
is an additional **nested fetch subinterval**, not a stage to add to the stage sum;
adapter-mutex time now covers the short ledger locks, separately from admission.

The std-only admission controller passed isolated ASan/UBSan and controlled TSan
checks (synchronized negative control and three detected intentional-race trials,
no suppressions/global ASLR changes). The first native U4 attempt passed six
targeted tests and ten repetitions each; ten of eleven broader regressions passed.
The remaining query fixture hard-coded the former physical batch count; its exact
hit oracle remains strict while batch-count expectations now derive from the
admitted limit. That failed evidence remains in `build/u3-u4-check/u4-run01/`.

Final review also preserves cancellation ordering for an aborted retry by sharing
the serialized writer lock, and poisons admission before releasing owned credits
on failures. New fixtures cover a queued **resident** read receiving the original
cold-read error and a cancellation queued behind blocked retry I/O. Writer-waiter
diagnostics count queued/acquiring sections, not completed mutations. Final native
qualification in `build/u3-u4-check/u4-run02/summary.json` passed six targeted tests,
ten repetitions each, and all eleven broader regressions. Heap-only TSan's first
attempt failed OpenMP discovery before compiling/running; its evidence is retained.
A fresh attempt uses the existing LLVM 18 OpenMP header (fallback include only)
with Clang 20 and the existing oneAPI runtime; no packages/privileges or global
settings are changed. Heap-only concurrent-fetch/admission tests passed both TSan
(`build/u3-u4-check/heap-thread-run02/`) and ASan/UBSan
(`build/u3-u4-check/heap-address-run01/`) on matching source hashes, without
suppressions. Five OFF helper/orchestration/CMake-scope tests passed in
`build/u3-u4-check/off-run01/`. This does not resolve HNSW construction findings or
qualify the complete native ANN/SIMD path. Controlled delayed-completion overlap establishes
independent backend batches, not device speedup. Warm-cache component scaling is
measured separately with identical batch limits and completed request counts;
full-scale performance, race/SIMD acceptance and crash durability remain outside
this milestone.

**Measured performance limit:** `build/u3-u4-check/u4-measurement-run02/summary.json`
retains 48 trials: float/uint8, 1/2/4/8 workers, serialized/bounded-concurrent control,
three repetitions, identical batch limits and 1,024 completed fetch requests per
trial. The 64-frame warm-cache fixture is not an ANN benchmark and includes thread
launch/join. Median uint8 fetches/s were:

| Workers | Serialized control | Bounded concurrent |
| --- | ---: | ---: |
| 1 | 611,274 | 644,018 |
| 2 | 308,127 | 780,723 |
| 4 | 306,199 | 28,604 |
| 8 | 220,432 | 18,130 |

Float shows the same high-worker regression. At 64 frames, the conservative
worst-shard capacity is 15 credits and a full batch reserves seven, allowing only
two such calls concurrently. Queue/wakeup overhead and short-lock contention are
not separately attributed by this measurement. This is a real limitation, not
evidence to discard or silently subtract.
It does not predict cold/device-bound scaling or performance at the 262,144-frame
candidate, where eight 256-request batches fit within the aggregate capacity.
U3/U4 are complete **within the recorded local functional/heap-sanitizer scope**;
no universal speedup, operational default promotion, matched-baseline result,
full-scale/race/SIMD acceptance or deployment follows. Subsequent admission-policy
tuning must use independent inputs and fresh qualification, not final queries.

### U8 — Qualification and staged rollout (after applicable gates)

1. Run scoped regressions across dtypes/routers, tiny/normal pools, growth/page
   boundaries, same/different-list inserts, concurrent queries/deletes, cancellation,
   budget admission, direct-I/O faults and checkpoint/reopen.
2. Reconfigure the stale sanitizer tree before claiming new ASan/UBSan coverage.
   Run scoped scalar TSan with working negative/positive controls and preserve
   unsuppressed reports. Existing HNSW construction lock-order findings remain open
   blockers for broad race-clean claims; no blanket suppression, stock modification
   or global ASLR change. SIMD/full-system safety remain separate qualifications.
3. On synthetic/independent tuning inputs, measure serial/parallel and round-robin/
   page-local controls at separately declared pool sizes. Retain stage times, counts,
   RSS scope, lock waits, page/flush counters, available device observations, repeated
   measurements and failures. No universal speedup threshold is assumed.
4. Freeze implementation, resources, scoring and workload policies before final
   evaluation. Advance from scoped tests to fresh SIFT10M, then SIFT100M with the
   unchanged full populations/all final queries. Preserve earlier completed and
   failed runs; do not overwrite, retroactively relabel or discard them.

**Release gate:** applicable correctness/resource/fault tests pass; critical findings
are resolved or block the affected claim; actual artifacts match declared identities;
normal/repetition recipes agree. Record executed evidence and remaining limits here.
Functional success alone establishes neither paper-equivalent throughput/durability
nor cold-cache behavior or production acceptance.

# Quickstart: SIFT
SIFT IVF testing from a new Linux computer.

This guide starts with a new **Ubuntu 24.04 x86-64 computer**, downloads the
project and IVF inputs, and runs full-scale query, insert, delete and mixed-update tests.
SPFresh is tested through its pre-existing infrastructure, not this guide.
You need sudo access, internet access and an AVX2-capable CPU.

Prefer **16 physical CPU cores, 128–256 GB RAM and at least 1 TB free SSD space**.
Index copies and retained checkpoints take much more space than the vector download
(1.28 GB for SIFT10M; 12.8 GB for SIFT100M). The resource recommendation is
for SIFT100M. Expect long-running jobs, especially training and mixed updates.

| Test | SIFT10M | SIFT100M |
|---|---|---|
| Query | 9M base, all 10,000 queries | 90M base, all 10,000 queries |
| Insert | Insert 1M → 10M live | Insert 10M → 100M live |
| Delete | Delete 1M → 8M live | Delete 10M → 80M live |
| Mixed update | Insert 1M and delete 1M | Insert 10M and delete 10M |

Run commands in order, in the **same Bash terminal**. Stop if any command fails.
Use fresh paths rather than overwriting existing or failed runs.

## 1. Install dependencies and get the project

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo add-apt-repository -y multiverse
sudo apt update

sudo apt install build-essential cmake git curl python3 tmux \
  libboost-dev libaio-dev libgoogle-perftools-dev \
  libmkl-full-dev libomp-dev
```

Accept the MKL license if prompted. This uses Ubuntu-packaged MKL and its
`iomp5`-compatible OpenMP runtime, not a separate oneAPI installation. Record
these versions when comparing performance with other builds.

For remote work, start a persistent terminal before continuing.

```bash
git clone --recurse-submodules https://github.com/GreyBaker/bufANN-lindorm-ivf.git "$HOME/project/ivf-project"
cd "$HOME/project/ivf-project"

# Only needed if IVF is inside a larger repository.
if [ -d ivf-only ]; then cd ivf-only; fi

export IVF_BUILD_DIR="$PWD/build/ivf-benchmark"
```

You should now see `CMakeLists.txt` and `benchmark/IVF/`. Run the remaining
commands from this directory.

## 2. Compile the tools

```bash
(
  set -euo pipefail
  bash benchmark/IVF/build.sh \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DMKL_ROOT=/usr \
    -DMKL_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu \
    -DINTEL_OMP_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
  cmake --build "$IVF_BUILD_DIR" --target bufann_gt_computer -j4
)
```

Only build the PQ helpers if generating new PQ inputs; skip this and step 3's
generation commands when reusing already verified inputs. The helper build is
separate from IVF and does not build a DiskANN graph. Warnings may occur; a
nonzero exit is a failure, not a guaranteed successful build.
```bash
cmake -S baselines/DiskANN -B baselines/DiskANN/build -DCMAKE_BUILD_TYPE=Release
cmake --build baselines/DiskANN/build \
  --target build_pq_pivot_standalone build_pq_codes_standalone -j4
```

These paths assume Ubuntu's packaged MKL layout.

## 3. Prepare the inputs once

Choose either profile. This recipe takes the **first 10M or 100M uint8 vectors
from BIGANN/SIFT1B**, not a random sample. Other valid subsets can be supplied
through the dataset profile; regenerate matching GT and PQ for their exact row
order. 

You'll want at least the eval and data drives to have significant disk space.

Once generated, these artifacts could be uploaded and downloaded instead of regenerated. `DATASET=sift10m` is also valid.
```bash
export DATASET=sift100m; export DATA_ROOT="$HOME/project/ann-data/$DATASET"; export DATA_BIN="$DATA_ROOT/base.bin" QUERY_BIN="$DATA_ROOT/query.10K.bin"
source "benchmark/datasets/$DATASET.sh"
printf 'Dataset=%s total=%s base=%s updates=%s\n' \
  "$DATASET" "$NPTS_FULL" "$NPTS_BASE" "$UPDATE_POINTS"

export OMP_NUM_THREADS=20 MKL_NUM_THREADS=1 IVF_PQ_PREFIX="$DATA_ROOT/pq/pq64"

(
  set -euo pipefail
  # PQ helpers can overwrite outputs; refuse existing PQ before downloading.
  for output in "${IVF_PQ_PREFIX}"_*; do
    [[ ! -e "$output" && ! -L "$output" ]] || {
      echo "ERROR: input artifact exists; reuse it or choose a fresh prefix: $output" >&2; exit 1;
    }
  done
  python3 benchmark/IVF/download_sift.py \
    --rows "$NPTS_FULL" --base "$DATA_BIN" --query "$QUERY_BIN"

  # Full-source PQ, including the insertion tail; no DiskANN graph.
  mkdir -p "$DATA_ROOT/pq"
  baselines/DiskANN/build/tests/build_pq_pivot_standalone \
    uint8 "$DATA_BIN" "$IVF_PQ_PREFIX" 64
  baselines/DiskANN/build/tests/build_pq_codes_standalone \
    uint8 "$DATA_BIN" "$IVF_PQ_PREFIX" 64 0
)
```

Download and PQ generation can take substantial time. GT is handled automatically
in step 4, after the workload plan is configured. The downloader checks
headers/ranges, refuses existing outputs and retains partial files; it does not
resume. Reuse completed inputs unchanged across runs; skip their generation when
using an existing verified dataset. The PQ helper's final `0` encodes all rows.

The recipe uses 64 eight-bit chunks: 64 bytes (512 bits) per vector. Lindorm's SIFT setting uses 128 four-bit chunks, also 64 bytes/vector;
equal storage does not make the quantizer/scanner implementations equivalent.

For this recipe, SIFT100M's initial base remains **90M**, not 900K. A GT log with
`base_trunc=900000` indicates a different indexed population, not training-only
sampling. Correct the loaded profile and choose a fresh work root instead of
expanding GT to compensate for an accidentally reduced base.

IVF filters GT for each live set and rejects insufficient surviving candidates.
This recipe measures **Recall@10**; Recall@100 needs deeper GT and fresh preparation.
Keep final queries out of tuning: freeze settings using base-only calibration or
a separate tuning set before scoring them.

## 4. Configure and run all four workloads

**Training limit:** SIFT100M uses **10 passes on all 90M base rows**, plus final
assignment: 990M searches, below the native 1B guard. The default 15 passes would
need 1.44B and fail; there is no environment override. SIFT10M fits with 15 passes.
This guard applies to **LV+ as well as the other trainers**. These are fixed pass
caps, not convergence-based stopping or paper-mandated iteration counts. Full
training can be expensive; LV+ does not reduce the number of sampled rows here.

**Separate PCA/routing allowance:** at 200K centroids and 128 dimensions, one
PCA covariance calculation counts `200000 * 128 * 129 / 2 = 1651200000`
operations, already above the inherited 1B routing-work default. The recipe
declares a finite 3B allowance for covariance plus solver/validation headroom.
It applies to training PCA and optimized final-router work; it does **not**
raise the separate 1B training-assignment cap or guarantee that every graph
operation fits. Keep guards enabled and retain any subsequent work-limit failure.

```bash
# Reload the same profile the workload scripts will use, before deriving counts.
source "benchmark/datasets/$DATASET.sh"
# Refuse a modified/smaller profile rather than silently training/indexing it.
case "$DATASET" in
  sift100m) expected_full=100000000; expected_base=90000000; expected_updates=10000000 ;;
  sift10m) expected_full=10000000; expected_base=9000000; expected_updates=1000000 ;;
  *) echo "ERROR: quickstart requires sift10m or sift100m" >&2; exit 1 ;;
esac
[[ "$DATA_TYPE" = uint8 && "$DIM" = 128 && "$METRIC" = l2 &&
   "$NPTS_FULL" = "$expected_full" && "$NPTS_BASE" = "$expected_base" &&
   "$UPDATE_POINTS" = "$expected_updates" ]] || {
  echo "ERROR: loaded profile does not match the full-scale quickstart" >&2; exit 1;
}
# Remove any smaller-workload overrides from earlier experiments.
unset INSERT_POINTS DELETE_POINTS QUERY_POINTS UPDATE_QUERY_BATCH QUERY_RATIO INSERT_CAP DELETE_CAP IVF_RUN_TAG IVF_COST_MODEL_REPORT IVF_SCHEDULER_SEED

# Train on every initial-base row; keep the indexed population unchanged.
if [ "$DATASET" = sift100m ]; then
  export IVF_MAX_REPS=10
else
  export IVF_MAX_REPS=15
fi
export IVF_SAMPLE_RATE=1
export IVF_TRAINER_MODE=lv-plus IVF_INIT=random_partition IVF_TRAINING_BREADTH=64 IVF_SEED=0 IVF_BUILD_THREADS=20 IVF_MEMORY_BUDGET_GB=96
export IVF_TRAINING_RERANK_FACTOR=3 IVF_TRAINING_REFINE_BREADTH=128

# Explicit starting configuration, not a qualified recall optimum.
export IVF_NLIST=200000 IVF_NPROBE=32 IVF_RERANK_FACTOR=3 QUERY_THREADS=8 RECALL_AT=10
export IVF_ROUTING_WORK_LIMIT=3000000000
# Enable optimized final routing separately from LV+ training.
export IVF_OPTIMIZED_MODES=1 IVF_ASSIGNMENT_ROUTER_MODE=pca-refined IVF_BUILD_ROUTER_RERANK_FACTOR=3 IVF_PCA_VARIANCE=0.95 IVF_REFINE_EF=128 IVF_QUERY_BREADTH=128 IVF_INSERT_BREADTH=64
export IVF_ROUTER_MODE=pca-refined IVF_ROUTER_RERANK_FACTOR=3 IVF_ROUTER_BREADTH=128

# Full profile update counts; ten mixed rounds in either dataset.
export UPDATE_PERCENT=10 UPDATE_BATCH=$((UPDATE_POINTS / 10)) IVF_SCHEDULER=update_driven NTHREADS=16 EVAL_TEMPFILES="$HOME/project/ann-eval/run01" EVAL_CACHED="$HOME/project/ann-eval/cache"
export IVF_BUFFER_FRAMES=262144
# Inspect the actual destinations; a large data volume does not ensure space
# on a work/cache filesystem mounted elsewhere. No existing evidence is deleted.
mkdir -p "$(dirname "$EVAL_TEMPFILES")" "$EVAL_CACHED"
df -h "$DATA_ROOT" "$(dirname "$EVAL_TEMPFILES")" "$EVAL_CACHED"

# Each workload gets an independent prepared base; stop on any failure.
(
  set -euo pipefail
  # Reuse covered GT; recommend/generate only if needed. Failure stops the loop.
  GT_BIN_KCACHED=$(python3 benchmark/scripts/ensure_gt.py \
    --base-file "$DATA_BIN" --query-file "$QUERY_BIN" \
    --full-count "$NPTS_FULL" --base-count "$NPTS_BASE" \
    --insert-count "$UPDATE_POINTS" --delete-count "$UPDATE_POINTS" \
    --mixed-rounds 10 --mixed-insert-cap "$UPDATE_BATCH" --mixed-delete-cap "$UPDATE_BATCH" \
    --k-out "$RECALL_AT" --expected-queries 10000 --num-threads 20 --generate)
  export GT_BIN_KCACHED
  for workload in query insert delete update; do
    bash "benchmark/IVF/sift/${workload}_prepare.sh"
    bash "benchmark/IVF/sift/${workload}.sh"
  done
)
```

`ensure_gt.py` ignores nonexistent shell paths, finds
existing caches, and checks every query against every declared live population.
It reuses a passing cache unchanged. Otherwise it recommends a sufficient depth,
generates exact full-source GT with the existing helper, and checks the new file
before returning its path. With no cache, it starts at K=1000; that is a starting
depth, not a coverage guarantee. Remove `--generate` to check/recommend without
allowing expensive generation. `IVF_BUILD_DIR` locates the generator; override
with `--gt-generator`. Use `--cached-gt` to prefer a custom existing cache.

No timestamps, manual K extraction, or report-path bookkeeping are needed.
Caches, reports and generator logs are preserved under `$DATA_ROOT/.gt/run-0001/`,
`run-0002/`, etc.; rerunning rechecks and reuses completed caches (underflowing ones
can seed recommendation). Failed/partial generation outputs are never selected.
Existing GT, PQ and index files are never overwritten. A malformed cache is an
error, not permission to bypass validation. On failure, inspect that run's
`error.json` and `generate-k*.log`, fix the cause, and rerun the same command.
Exit 3 means generation is needed; 1/2 means an error. No failure returns a GT path.
Keep the shell assignment and `export` separate as shown: combining them can mask
failure and allow preparation to proceed without validated GT.

Recommendation scans require NumPy and original uint8 squared-L2 inputs in at most
128 dimensions. Defaults admit 32 scan queries and 256 MiB estimated scan workspace;
adjust with `--max-scan-queries`, `--chunk-rows`, or `--memory-budget-mb` if needed.
These are not limits on the native GT generator's RAM. Generation may take substantial
time/memory. Tagged 10K-query K=1000 GT is **120,000,008 bytes**; larger K needs more
space. If other wrappers score intermediate insert/delete states, add the actual
`--insert-round-cap`/`--delete-round-cap`. Inputs and counts must match the intended
workload; existing exact-cache provenance remains your responsibility.

GT runs entirely before index preparation and QPS measurement, without changing
ANN settings, timing or scoring policy. Deeper shared GT does not make ten-column
derived GT fully tie-aware. Preserve completed results; retry failed workloads in
a fresh `EVAL_TEMPFILES` root with the same verified `EVAL_CACHED` bundle.

Memory budgets check workspace estimates, not hard RAM caps. Keep these settings
fixed across preparation and execution. Cache hits restore an index, not measure
a new build; use a fresh cache root when measuring fresh construction.
The full-scale recipe explicitly requests 262,144 frames (1 GiB page payload).
CLI/harness fallbacks remain 64 frames and a 1 GiB total allowance for small runs;
they are not full-scale profile defaults. Python preparation/runtime admission
rejects a budget smaller than the native per-frame allowance (payload plus
4,096 bytes/frame); native composed admission still accounts for lists, locators,
router and worker scratch. Passing this cheap lower-bound check is not a total-RAM
guarantee. No existing manifests/cache identities are rewritten.
Keep cache storage enabled for the normal four-workload loop: identical source,
PQ, builder/library and build settings allow insert/delete/update preparation to
restore the immutable base instead of repeating training. Restoration uses real
verified copies, not shared writable heaps; it still takes I/O and disk space.
`EVAL_CACHE_NO_STORE=1` prevents storing a new bundle but does **not** disable cache
hits. Use an unused cache root (or a new `IVF_CACHE_TAG`) for a deliberately fresh
build. Do not rebuild binaries or change build settings between those workload
preparations if the goal is reuse.

Each workload starts independently from the same profile base. Insertions
from the insert test do **not** carry into the delete test.
IVF sequential insert/delete use one mutation worker; `QUERY_THREADS` controls later
scoring. Centroids and routers stay frozen across updates.

These settings run **10 rounds**: each inserts/deletes 100K rows for SIFT10M or
1M rows for SIFT100M. There is **no required query ratio or total query quota**.
IVF's `update_driven` mode replays the query pool until all mutations complete,
joins admitted queries, then scores all 10,000 queries after each round. Record
the actual query count and overlap. This workload does not require reproducing
the external SPFresh tester's scheduler or scoring cadence.

## 5. Read results and repeat

Outputs are under `$EVAL_TEMPFILES/$DATASET/IVF/` in `query/`, `insert/`,
`delete/` and `update/`. Retain logs, structured results and input identities.


For three query repetitions at the same frozen settings:

```bash
bash benchmark/IVF/sift/query_sweep.sh \
  --tag full-scale-repeated \
  --nprobes 32 --rerank-factors 3 --threads 8 \
  --repeats 3 --warmups 1 \
  --memory-budget-gb 96 --buffer-frames 262144 --timeout 86400
```

For another complete run, choose a fresh work root and repeat the loop in step 4
without changing the frozen settings:

```bash
unset IVF_RUN_TAG
export EVAL_TEMPFILES="/project/ann-eval/run02"
# Repeat only the workload loop from step 4.
```

Choose a new root for each repetition so preparation starts from the same IVF
base. Do not delete failed evidence or change settings in response to final-set
results. Tagged reruns from an existing IVF prepared base are also supported;
the fresh-root recipe makes each repetition explicit.
