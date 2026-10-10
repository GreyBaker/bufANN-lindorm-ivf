# Native IVF build and benchmark guide

The single-node IVF backend supports build/reload, queries, sequential
insert/delete and concurrent weighted or fixed 8+8 mixed rounds on Linux/WSL.
It routes through centroids and reuses global PQ8 codes; no vector graph or
DiskANN/Vamana conversion is required. BufANN/SPFresh remain separate baselines.
These tools and small functional fixtures **do not qualify production settings
or establish paper speedups**. Supply your own inputs, declare resource/quality
budgets, and keep reserved final queries out of calibration and tuning.

## Build with installed system dependencies

Use an existing full routing-enabled build via `IVF_BUILD_DIR`, or configure a
separate tree (adjust these oneAPI paths for your installation):

```bash
export IVF_BUILD_DIR="$PWD/build/ivf-benchmark"
bash benchmark/IVF/build.sh \
  -DMKL_ROOT=/opt/intel/oneapi/mkl/2026.1 \
  -DMKL_LIBRARY_DIR=/opt/intel/oneapi/mkl/2026.1/lib \
  -DINTEL_OMP_LIBRARY_DIR=/opt/intel/oneapi/compiler/2026.1/lib
```

This enables `IVF_BUILD_ROUTING=ON` and builds `ivf_build`, `ivf_driver`,
`ivf_cost_model`, `ivf_training_diagnostics` and `ivf_router_diagnostics` under
`$IVF_BUILD_DIR/tests/`. The default tree is `build/ivf-benchmark`; set
`IVF_BUILD_JOBS` to change the default 4 jobs. The full native build retains
administrator-installed MKL (ILP64, Intel-threaded), iomp5 and libaio linkage.
Prepare/run scripts consume binaries: they never install or configure dependencies.
The shared `benchmark/scripts/build.sh` dependency-prefix build is separate.
IVF/dependency CMake setup is in `cmake/IVF.cmake` and `cmake/Hnswlib.cmake`;
see [hnswlib documentation](../../dependencies/hnswlib/README.md) for the vendored
IVF version and checksum-pinned, unmodified upstream comparison. No build-time
patching or header generation is needed.

The Python entry points are `harness.py` (query/sequential workloads), `mixed.py`
(mixed rounds), `qualify.py` (training/query experiments), `cost_model.py`
(calibration) and `router_sweep.py` (fixed-router experiments). Shared local-file
and process helpers live in `common.py`; build defaults and frozen-report policy
live in `settings.py`. New training/calibration/query-sweep evidence records the
shared module identities; legacy build/cache setting fields remain unchanged.

## Input/PQ/GT contracts

- Load `benchmark/datasets/<DATASET>.sh`. Query preparation requires existing
  data/query `.bin` files; mutation preparation uses shared `ensure_dataset`
  staging. `DATA_BIN` must contain exactly `NPTS_FULL` rows of `DIM` coordinates
  of `DATA_TYPE`. Only `[0, NPTS_BASE)` forms the initial index. Profile splits
  and external source-row IDs are never inferred or remapped.
- `IVF_PQ_PREFIX` supplies current five-offset `<prefix>_pq_pivots.bin` and
  **full-source** `<prefix>_pq_compressed.bin`. PQ is global, non-residual, 8-bit,
  and not retrained by prepare. Canonical graph codes often cover only the base.
  If needed, copy existing pivots to a **new** prefix and run
  `build_pq_codes_standalone TYPE DATA_BIN NEW_PREFIX ACTUAL_PQ_CHUNKS 0`.
  Do not overwrite the canonical prefix or change the comparison codebook.
- Base GT defaults to `GT_BIN_KCACHED` filtered by the unchanged shared
  `resolve_incremental_gt ... --base_trunc NPTS_BASE`. `IVF_GT_FILE` instead
  supplies matching caller-owned GT, linked read-only. GT contains IDs/distances
  and optional tags equal to the ID plane; nonidentity tag mappings, nonfinite
  values and inadequate deep-cache candidates fail rather than fabricating recall.

## Prepare and run

```bash
export DATASET=sift1m
export IVF_BUILD_DIR="$PWD/build/ivf-benchmark"
export IVF_PQ_PREFIX=/absolute/path/to/existing-full-source-pq/prefix
export IVF_NLIST=1024  # example only, not qualified tuning
export IVF_MEMORY_BUDGET_GB=4 IVF_BUILD_THREADS=4
bash benchmark/IVF/sift/query_prepare.sh
IVF_NPROBE=16 IVF_RERANK_FACTOR=4 QUERY_THREADS=4 \
  bash benchmark/IVF/sift/query.sh
# query_run.sh is an equivalent alias.
IVF_RUN_TAG=probe32 IVF_NPROBE=32 QUERY_THREADS=4 \
  bash benchmark/IVF/sift/query.sh
```

Prepare creates `eval_tempfiles/<dataset>/IVF/query/`: read-only input links,
real private `index/<dataset><suffix>` bundle files, then `prepared.json`.
Existing/partial work trees are refused and preserved; select a fresh
`EVAL_TEMPFILES` root to re-prepare. Run checks profile/query/GT/manifest identities;
native open validates component contents. Working index files may not symlink
to cache. Outputs are `run.log`, `native_result.json`, canonical `result.json`
and interleaved SPFresh `neighbors.bin`; shared metrics extraction is cross-checked
against native output. Failures do not publish empty successful results.
Existing run files are refused; safe `IVF_RUN_TAG` values (letters, numbers,
hyphens, underscores) place new output in `query/runs/<tag>/`.

### Settings and frozen routing

Build environment variables are `IVF_` plus uppercase CLI fields. Defaults:

| Fields | Defaults |
|---|---|
| `NLIST`, `SAMPLE_RATE`, `MAX_REPS`, `SEED` | explicit nlist, 1, 15, 0 |
| `BUILD_THREADS`, `MEMORY_BUDGET_GB`, `PAGE_SIZE`, `BUFFER_FRAMES` | 1, 1, 4096, 64 |
| `ROUTER_DEGREE`, `CONSTRUCTION_BREADTH`, `QUERY_BREADTH`, `INSERT_BREADTH`, `ROUTER_SEED` | 16, 200, 128, 64, 100 |
| `TRAINER_MODE`, `INIT`, `TRAINING_BREADTH` | `ann-reuse-seeded`, `random_partition`, 64 |
| `OPTIMIZED_MODES`, `REFINE_EF`, `PCA_VARIANCE`, `BUILD_ROUTER_RERANK_FACTOR`, `ROUTING_WORK_LIMIT` | 0, 128, .95, 3, 1000000000 |

Run defaults: `IVF_NPROBE=1`, `IVF_RERANK_FACTOR=2`, `QUERY_THREADS=1`,
profile `RECALL_AT` (normally 10), `IVF_BUFFER_FRAMES=64`.
`IVF_QUERY_MEMORY_BUDGET_GB` falls back to `IVF_MEMORY_BUDGET_GB`, then 1.
`IVF_ROUTER_BREADTH` overrides persisted query breadth without rebuilding.
Query thread settings do not imply a mixed-workload worker split.

Query router modes are `hnswlib`, `exact`, `patched-full`, `refined`, `pca`,
`pca-refined`; the latter four need `IVF_OPTIMIZED_MODES=1` artifacts.
`IVF_ASSIGNMENT_ROUTER_MODE` freezes the base/insert default (`hnswlib`);
`IVF_ROUTER_MODE` overrides queries only and does not invalidate the build cache.
`IVF_ROUTER_RERANK_FACTOR` is a query-only centroid-rerank override, distinct
from `IVF_RERANK_FACTOR` (full-vector candidates after PQ scanning).
Construction/pruning use full coordinates. Refinement replaces bottom out-edges
from a frozen search/prune pass and reinserts zero-in-degree nodes once; global
connectivity is not guaranteed. PCA traversal reranks in full coordinates.
Final centroids/routers/PCA are frozen across mutations and query overrides.
An explicit query `--router_mode default` resolves to the persisted mode, as
does omitting the flag. Readers and writers require producer
`hnswlib-0.8.0-stock1`, including mutable checkpoint origins. Historical
`hnswlib-0.8.0-prefetch1` and `hnswlib-0.8.0-nomv` bundles are rejected: rebuild
into a fresh prefix or use the original tool for historical replay. Do not
relabel old artifacts/reports, including stock builds made before the label fix.
Framing versions remain supported independently of the producer policy.

### Sequential insert/delete

```bash
UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete.sh
```

Use the same build/PQ/profile environment. Shared split/count helpers preserve
SPFresh's base, insert-tail and delete-base-tail IDs; `INSERT_POINTS`/`DELETE_POINTS`
override counts only. GT must match the **post-update** live prefix; insufficient
survivors fail. Each pair owns `.../IVF/{insert,delete}/`, restores only an immutable
base, mutates a new private heap, checkpoints, then reopens another private copy
for scoring. `mutation_result.json` and final-query `result.json` are separate,
cross-checked canonical records. Tags preserve repeats from the same base.
One sequential mutation worker is used; `QUERY_THREADS` controls later scoring,
not concurrent queries. Native `search_after_update` is also sequential.

### Concurrent mixed rounds

```bash
UPDATE_PERCENT=10 UPDATE_BATCH=10000 bash benchmark/IVF/sift/update_prepare.sh
UPDATE_PERCENT=10 UPDATE_BATCH=10000 IVF_SCHEDULER=weighted NTHREADS=16 \
  QUERY_RATIO=80 QUERY_THREADS=4 bash benchmark/IVF/sift/update.sh
UPDATE_PERCENT=10 UPDATE_BATCH=10000 IVF_SCHEDULER=fixed_8_8 NTHREADS=16 \
  IVF_RUN_TAG=fixed QUERY_THREADS=4 bash benchmark/IVF/sift/update.sh
```

Shared SPFresh layout helpers supply caps/counts/round GT, including equal
insert/delete caps. Weighted mode uses the remaining-operation scheduler;
fixed mode has eight mutation and eight query workers, not graph-maintenance
threads. `QUERY_POINTS` follows the shared total/per-round policy; otherwise
`QUERY_RATIO=80`. `QUERY_THREADS` controls post-round scoring only. Each round
joins, validates/checkpoints and scores against matching GT; the next process
reopens a new private heap. Concurrent query timing/overlap has no synchronized
recall against a moving set; recall is measured separately after the barrier.
Root `result.json` is the final round; `summary.json` retains all rounds.
For direct native flags, see `ivf_driver --help` (`--workload mixed`).

Posting lists grow by append/+16 slots and use locator/swap deletion. Deleted
IDs cannot be reused. Checkpoints retain tombstones/aborted physical records and
require quiescent workers; they provide no slot reclamation, coordinated WAL,
crash recovery, atomic/fsync publication or transactional durability.

### Cache and measurement scopes

`eval_cached/<dataset>/IVF/<sha256-key>/` binds exact settings/profile, full
vector/pivot/code identities, builder/library bytes and optional `IVF_CACHE_TAG`.
Components are hashed before/after real copies; identity/checksum failures refuse
reuse. Nonempty `EVAL_CACHE_NO_STORE` disables new stores, not restores. Hits
measure restoration, **not** new builds. Identities are not authentication or
external PQ-training provenance; keep inputs trusted/immutable. Artifacts are
machine-local, and runtime-library/ABI/toolchain tracking is not exhaustive:
advance `IVF_CACHE_TAG` or change cache roots when that environment changes.

Hashing/copies warm caches; native open hashes the entire heap. `build_time_s`
spans native build and its post-build router reload, excluding Python hashing/cache work. `query_time_s`
excludes preparation, checks, open, scoring and export. Lifetime peak RSS includes
these phases. Backend heap-read counters are buffer-pool reads, not physical
device I/O. Cache policy is uncontrolled (not cold-device throughput); heap
pin/copy admission remains serialized. Memory budgets guard conservative workspace
estimates, not hard RSS caps. Record machine/storage/compiler/runtime settings,
including `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, for performance measurements.

## Training and bounded ablations

Select `IVF_TRAINER_MODE` (or native `--trainer_mode`):

| Mode | Assignment policy |
|---|---|
| `exact` | Scalar double-L2 enumeration, work-bounded; no GEMM requirement |
| `hnsw` / `ann-rebuild` | Rebuilt full-coordinate HNSW, breadth-sized double rescoring (`hnsw` is a diagnostic alias) |
| `ann-reuse` / `ann-reuse-seeded` | Reused full-coordinate graph, legacy rescoring; optional previous-assignment seed |
| `ann-reuse-count` / `ann-reuse-seeded-count` | Reused full-coordinate graph returns nearest-one under its float metric; double assigned objective counted separately |
| `lv` / `lv-plus` | Reused/refined graph, 95%-variance PCA assignment, bounded full-coordinate rerank; LV+ adds only previous-assignment seeding |

CLI/harness defaults are in the settings table; programmatic `BuildConfig` keeps
legacy `ann-rebuild`/`sample` defaults. `IVF_INIT=sample|random_partition` and
`IVF_TRAINING_BREADTH` enter cache identity independently of final routing.
LV/LV+ require `random_partition`: full-coordinate old-neighbor refresh, one
initial bottom refinement/isolated-node reinsertion, PCA refit/reprojection at
every joined assignment boundary (including the final one), and first/invalid-seed
fallback. `IVF_TRAINING_RERANK_FACTOR=3` bounds returned full-rerank candidates;
`IVF_TRAINING_REFINE_BREADTH=128` controls initial refinement. Matching native
flags use lowercase underscores; nondefault overrides require LV/LV+.
These are independent of final router settings and do not require optimized
final-router artifacts. Fixed pass caps, hnswlib substitution and interpreted
moving-PCA/initial-only-refinement schedules limit exact paper reproduction.

`qualify.py train` uses identical deterministic base-only reservoirs, initialization,
seed, ordered means, empty recovery and pass caps. Each mode runs in a fresh
process with independent lifetime RSS. Example from the repository root:

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

Omit `exact` at scales beyond its budget; bounded exact evaluation still runs
but is **not** an exact-trained reference, so reference ratios remain null.
Keep all settings except seeding identical for LV vs LV+; separate count-policy
controls from PCA effects. Repeat training in separate output paths/processes.
Training breadth falls back to diagnostic `--insert-breadth` if omitted;
LV rerank/refinement arguments apply only to LV modes in a multi-mode run.

Exact training costs `sample_count * (iterations + 1) * nlist` vector distances;
exact evaluation costs `min(evaluation_rows, sample_count) * nlist`. Oversized
reference work fails before modes launch. These limits count vector distances/
searches, not coordinates or CPU time; `--routing-work-limit` separately bounds
each refinement/PCA setup's coordinate/solver work. Do not bypass budget failures.
`--max-sse-ratio` optionally gates evaluation SSE against a selected `exact`
trainer (required): freeze the threshold first. Both-zero objectives give ratio 1;
exact-only zero gives null and fails a supplied gate. `--max-peak-rss-mb` is a
post-run lifetime-RSS gate, not an allocation limit.

Plans/logs and `.sample_ids.bin`, `.centroids.bin`, `.assignments.bin`,
`.occupancy.bin` are retained (two-uint32-header matrices; float centroids,
uint32 other values). Sample hashes must match; source/executable identities are
checked before/after modes. Exact double nearest-centroid SSE and tie-aware
agreement use identical evenly spaced reservoir slots: **training-sample**, not
holdout or base-list quality. Approximate assigned SSE is separate. Inspect
occupancy/recovery and unrepaired connectivity, not timing alone. Graph-distance
counts cover assignment traversal, exclude construction/reinsertion, and are
unavailable for vanilla rebuilt HNSW. Returned/reranked/objective counts differ.
All graph/PCA setup is inside `training_time_s`; setup fields are graph-time
subsets, worker fields sum worker seconds rather than wall time.
`evaluation_time_s` is separate; `process_wall_s` spans the child, not Python hashing.

Legacy trainers write `IVFINDEX3` with trainer/init/breadth and an optimized flag;
count/LV trainers write `IVFINDEX4`, also binding search policy and independent
rerank/refinement settings. Existing v1/v2 bundles reopen with their actual legacy
rebuilt/sample/insert-breadth identity. Mutable checkpoints preserve origin
version/settings; model selection is metadata, not another native index format.

## Base-only frozen calibration

`cost_model.py` orchestrates native `T1+T2` using base rows and disjoint synthetic
queries, never official queries. It freezes source/PQ/tool identities, samples,
exact GT, routers, eligible/excluded candidates and predictions. `T1` is mean
measured centroid routing; `T2` is radius occupancy scaled by `N/m` and a profiled
scalar-PQ8 scan/selection coefficient using real base codes. Double-L2 sample GT
retains repeated GT-cluster IDs and inclusive radius ties. ADC-table construction,
fetch and exact rerank are excluded: this is not an end-to-end-QPS optimum or
prediction-validation study. PQ4/SIMD small-list penalties are not modeled.

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

Choose counts that fit the base. Eligibility is
`sample_count >= nlist * min_sample_per_centroid`. Default 32 is **project policy,
not a paper requirement or SIMD batch size**. Declared positive minima down to 1
permit sampling-sensitivity studies, not proof of adequacy. Version-2 reports
record minima, ratios, exclusions, below-32 warnings and trainer search policy;
version-1 reports retain implicit minimum 32 and legacy trainer contracts.
Do not edit frozen reports to admit excluded candidates or refit predictions.

`IVF_COST_MODEL_REPORT` supplies the chosen nlist for query, mutation and mixed
shell preparation. `IVF_NLIST` may be omitted; if also supplied, it must match:

```bash
export IVF_COST_MODEL_REPORT=/new/path/calibration-r10/report.json
unset IVF_NLIST
# Match every report trainer/router/resource setting; full builds require sample rate 1.
export IVF_SAMPLE_RATE=1 IVF_TRAINER_MODE=lv-plus IVF_INIT=random_partition
export IVF_TRAINING_BREADTH=64 IVF_BUILD_THREADS=4 IVF_MEMORY_BUDGET_GB=4
bash benchmark/IVF/sift/query_prepare.sh
```

Source/PQ/settings mismatches and conflicting manual choices fail. Preparation/
cache metadata and run `nlist_selection.json` bind report path/hash; preserve the
report unchanged. Freeze separate reports per recall/top-k target before validation.
Compare predicted/measured routing and scan costs, absolute misses and selected-
candidate ratios separately from end-to-end latency. Manual builds need no report;
manual sweeps do not validate predictive selection.

## Fixed-router diagnostics

`ivf_router_diagnostics` and `router_sweep.py` compare frozen centroid routers,
not end-to-end ANN recall/QPS. Prepare `IVF_OPTIMIZED_MODES=1` for all six modes;
diagnostics do not rebuild/refit centroids/PCA. Ensure probes fit the index:

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

Exact reference enumeration costs `query_rows * nlist` distances. Records retain
exclusive settings/plan/summary files, per-trial process logs (including failed
or timed-out children), input/tool/helper hashes and available/selected affinity.
Use `--cpus` for explicit pinning. Hashing warms inputs; cache state is otherwise
uncontrolled. No output is overwritten, and failed partial evidence is retained.
Native records retain
centroid recall, identities, projection/graph/rerank times, distance availability,
PCA dimension/variance, connectivity and lifetime RSS. Timed worker search includes
centroid-match scoring, excludes exact GT/load/checks. A mode-query warmup and prior
validation warm caches. Repeats are averaged before selecting the fastest observed
breadth meeting each **minimum** recall target: not identical-recall interpolation
or exhaustive tuning. Preserve failures/missed targets; bottom reachability and
zero in-degree do not establish strong connectivity.

## Repeated query sweeps

On the prepared bundle, each trial invokes the unchanged single-run path:

```bash
DATASET=sift1m IVF_BUILD_DIR="$PWD/build/ivf-benchmark" \
  bash benchmark/IVF/sift/query_sweep.sh \
  --tag repeated-01 --nprobes 8,16,32 --rerank-factors 2,4,8 --threads 1,4 \
  --repeats 3 --warmups 1 --memory-budget-gb 4 --buffer-frames 64 --timeout 1800
```

`RECALL_AT` and profile/work resolution match `query.sh`; lists/centroids/PQ/GT
and persisted breadth do not change. Explicit settings are checked against native
records; unsupported settings fail. Maximum 4096 trials. Plans/orchestration logs,
completion records and final summary live in `query/sweeps/<tag>/`; canonical
records/logs/neighbors remain in unique `query/runs/` directories. Input/manifest/
prepared/driver/script/profile identities must stay fixed; every trial validates
components. Existing destinations, including failures, are never resumed/deleted.

Warmups are separate discarded processes: they warm OS caches, not the next
process's buffer pool. Ordered configurations/warmups/repeats can bias results;
frequency, contention and other processes are uncontrolled. Optional `--cpus`
pins descendants to comma-separated IDs within current affinity using installed
`taskset`; it does not isolate CPUs. Small sets may use fewer active query workers.
Aggregates are min/median/max across runs for recall/QPS, query/open time, mean/
p50/p95/p99 latency, lifetime RSS, backend reads, workspace and active workers;
medians of run percentiles are **not pooled-query percentiles**. `process_wall_s`
includes the script's hashing/open/GT/scoring/export; orchestrator rehashes add
untimed I/O. No cold-cache or device-I/O claim is made.
`--min-recall` gates percent Recall@K (default 0, no meaningful target);
`--max-peak-rss-mb` gates maximum measured RSS. Neither is matched-recall baseline
selection or automatic acceptance; paired baseline studies need their own protocol.

Training/query plans precede work; summaries are written last after required
children/identity checks. Failures/timeouts preserve partial output without a final
summary; timeouts terminate descendant process groups. Completed measurements
failing declared gates retain `gates_passed: false` and exit nonzero. There is no
rollback, atomic/fsync publication or resume guarantee. Summaries explicitly set
`production_qualified: false`.

## Functional verification

Dependency-isolated checks need no native MKL/libaio/BufANN heap dependencies:

```bash
cmake -S tests/ivf -B build/ivf-contracts -DCMAKE_BUILD_TYPE=Release \
  -DIVF_BUILD_ROUTING=ON
cmake --build build/ivf-contracts -j4
ctest --test-dir build/ivf-contracts --output-on-failure
```

Routing configuration fetches the checksum-pinned, unmodified hnswlib comparison
into the build tree; the modified IVF implementation is checked in under
`dependencies/hnswlib/include/ivf_patched/`. Earlier comparison measurements used
a safety-corrected upstream copy; retain that distinction in new studies.
Isolated `ivf_build`/`ivf_driver` validate only; native workloads/calibration require
a full Linux build. Enable tests by adding `-DIVF_BUILD_CONTRACT_TESTS=ON` to the
full build command above. Build all test executables before full CTest:

```bash
cmake --build "$IVF_BUILD_DIR" -j4
ctest --test-dir "$IVF_BUILD_DIR" -N
ctest --test-dir "$IVF_BUILD_DIR" --output-on-failure
# Focused training/calibration/CLI subset:
ctest --test-dir "$IVF_BUILD_DIR" --output-on-failure \
  -R '^(ivf_ann_training|ivf_training_diagnostics|ivf_ann_training_cli|ivf_cost_model_cli|ivf_cli_execution)$'
```

Make MKL/Intel OpenMP runtime libraries discoverable via loader configuration or
`LD_LIBRARY_PATH`. Fixtures cover dtypes, joined determinism, base-only sampling,
LV/PCA/candidate oracles, bounded work/memory refusal, exact SSE/artifacts, report
minima 32/20/1 and historical compatibility, frozen provenance, build/reopen,
mutation/mixed scoring, caches/no-overwrite/identity failures, repeats and gates.
Additional checks cover sibling CMake target visibility, routing builds without
Python discovery, artifact claim/hash behavior, report-only shell preparation,
retired-producer rejection, stock-only checkpoint roundtrips and router-sweep
failure/timeout evidence.
Optional `ivf_trained_helper_integration` requires both existing standalone helpers
configured with `IVF_PQ_PIVOT_HELPER` and `IVF_PQ_CODES_HELPER`; it checks trained
encoder/ADC parity and script results for all dtypes (4096 rows, 13 dimensions,
three unequal chunks). Retain configure/build/test logs and tool/runtime identities.
Scoped scalar sanitizers avoid stock upstream's known SIMD prefetch lookahead
issues; they do not qualify the unmodified SIMD comparison, instrument external
helpers/system libraries or establish full-system race qualification.
Production quality, latency/RSS/concurrency,
sampling adequacy and prediction accuracy still require separate measurements.
