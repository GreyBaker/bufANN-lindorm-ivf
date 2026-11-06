# IVF benchmark guide

Query-only benchmark runs fork the prepared immutable base into a private mutable session before warmup or query timing, without inserting, deleting, or creating a checkpoint. Native `search_only` without `--private_heap_path` still supports immutable search; supplying a fresh private path with immutable input selects the no-update fork.

These scripts prepare and run IVF query-only, insertion, deletion and mixed-update
workloads on Linux/WSL. Run the commands below from the repository root.

**Choose a workflow:** [full-scale SIFT setup](#quickstart-sift),
[all four workloads](#4-configure-and-run-all-four-workloads),
[wide query grid](#quickstart-sift100m-wide-query-only-grid),
[reuse/refinement](#follow-up-reuse-the-first-run-and-refine-100k-400k), or
[recall diagnostics](#investigating-the-approximately-99559-recall-ceiling).
The numbered sections immediately below describe the lower-level scripts for
custom datasets and ablations; the quickstart supplies its own fixed preset.

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

Choose a centroid count and resource budget before evaluating final queries.
The SIFT1M example below uses the configuration exercised by the local WSL run;
it is not the SIFT100M configuration or a change to library/harness defaults:

```bash
export DATASET=sift1m
export IVF_PQ_PREFIX=/absolute/path/to/existing-full-source-pq64/prefix
export IVF_NLIST=10000 IVF_MEMORY_BUDGET_GB=6 IVF_BUILD_THREADS=20
export IVF_QUERY_MEMORY_BUDGET_GB=6 IVF_BUFFER_FRAMES=262144
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=1
export IVF_SAMPLE_RATE=1 IVF_TRAINER_MODE=lv-plus IVF_MAX_REPS=15
export IVF_INIT=random_partition IVF_TRAINING_BREADTH=64 IVF_SEED=0
export IVF_TRAINING_RERANK_FACTOR=3 IVF_TRAINING_REFINE_BREADTH=128
export IVF_OPTIMIZED_MODES=1 IVF_ASSIGNMENT_ROUTER_MODE=pca-refined
export IVF_BUILD_ROUTER_RERANK_FACTOR=3 IVF_PCA_VARIANCE=0.95 IVF_REFINE_EF=128
export IVF_QUERY_BREADTH=128 IVF_INSERT_BREADTH=64
export IVF_ROUTER_MODE=pca-refined IVF_ROUTER_RERANK_FACTOR=3 IVF_ROUTER_BREADTH=128

bash benchmark/IVF/sift/query_prepare.sh
export IVF_NPROBE=32 IVF_RERANK_FACTOR=3 QUERY_THREADS=8
bash benchmark/IVF/sift/query.sh

UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete.sh

UPDATE_PERCENT=10 UPDATE_BATCH=10000 bash benchmark/IVF/sift/update_prepare.sh
UPDATE_PERCENT=10 UPDATE_BATCH=10000 IVF_SCHEDULER=update_driven NTHREADS=16 \
  QUERY_THREADS=8 bash benchmark/IVF/sift/update.sh
```

Sequential insertion defaults to one worker; bounded parallel insertion is opt-in
with `IVF_INSERT_THREADS`. Sequential deletion stays serial. Mixed rounds have
equal insert/delete caps and their own worker allocation, unchanged by insertion
settings.
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

### Insertion workers, input staging and same-process warmup

These execution controls do not retrain centroids, routers, PCA or PQ and do not
change source-row IDs, mutation volumes, ANN candidate budgets or final queries.

| Shell setting | Native driver option | Default | Supported scope |
|---|---|---|---|
| `IVF_INSERT_THREADS=1..256` | `--insert_threads N` | `1` | Sequential `insert_only`/`search_after_update`; effective workers are capped by insert count |
| `IVF_INSERT_INPUT_POLICY=streaming\|staged` | `--insert_input_policy POLICY` | `streaming` | Sequential insertion only; mixed input staging is separate and unchanged |
| `IVF_WARMUP_QUERIES=0..100000` | `--warmup_queries N` | `0` | Query-producing calls; once before the continuous workload in mixed runs |

Native options require `--validate_inputs 0`. Insertion options are rejected for
query-only, delete-only and mixed native calls; warmup is rejected for mutation-only
calls. The shell harnesses ignore insertion settings outside sequential insertion.
`NTHREADS` remains the mixed foreground pool size, not an insertion-worker alias.
`UPDATE_BATCH` controls mixed round caps, not sequential insertion/deletion caps.

- **Streaming:** validate every requested coordinate before mutation, then use
  bounded, worker-owned readers/buffers inside the insertion timer. The existing
  foreground scope is labeled `insert_timing_contract_version=1`.
- **Staged:** admit and load all requested insert coordinates before mutation;
  joined workers consume a read-only staged array. Payload is
  `insert_count * dim * sizeof(dtype)`, reported as
  `insert_staged_coordinate_bytes`. It is additional to worker/router/PQ scratch,
  stacks, lists and pool admission—not a total-RSS estimate. Version `2` excludes
  source-row loading from `insert_time_s`, while heap I/O, publication, worker
  launch/join and periodic progress remain included. Full-source before/after
  integrity checks and a separate verified checkpoint are retained.
- **Failures:** all insertion threads launch behind an admission gate. A callback
  or launch failure stops new admission and joins admitted work before rethrowing;
  a partial mutation workload publishes no successful checkpoint/result. Its owned
  heap and failed evidence are retained rather than repaired or overwritten.
- **Warmup:** inputs come only from frozen base-trained full-coordinate centroids,
  using `floor(i * nlist / requested)` (`frozen_base_centroid_grid_v1`). Integer
  dtypes round halves away from zero and clamp to their range. No final queries or
  GT select the inputs. Every requested warmup executes, even when centroid IDs
  repeat. Warmup is serial on the caller, on the same index object as timed queries,
   after mutation/checkpoint for sequential workloads, but before all mixed rounds.
   Its results are discarded, never
  scored/exported or counted in final query statistics or mixed foreground counts.
  It has its own `warmup_time_s`, outside `query_time_s` but inside driver workflow
  wall time. New query workers do not inherit caller TLS; this is not an OS-cache,
  whole-pool warming or representative-sampling guarantee. A warmup failure prevents
  final query success output; an already completed checkpoint is preserved.

After preparing an independent insert work tree, optional controls can use fresh
tags. Declare them before evaluation; do not select workers or warmup counts by
tuning on final-query results:

```bash
# Same staged policy: compare worker counts without also changing source I/O.
IVF_RUN_TAG=staged_one IVF_INSERT_THREADS=1 IVF_INSERT_INPUT_POLICY=staged \
  IVF_WARMUP_QUERIES=0 bash benchmark/IVF/sift/insert.sh
IVF_RUN_TAG=staged_eight IVF_INSERT_THREADS=8 IVF_INSERT_INPUT_POLICY=staged \
  IVF_WARMUP_QUERIES=0 bash benchmark/IVF/sift/insert.sh
# Warmup happens inside the final query driver, not in a throwaway process.
IVF_RUN_TAG=centroid_warmup IVF_INSERT_THREADS=8 IVF_INSERT_INPUT_POLICY=staged \
  IVF_WARMUP_QUERIES=64 bash benchmark/IVF/sift/insert.sh
```

The existing separate-process `query_sweep.sh --warmups` is **not** this capability:
it cannot transfer a process-owned buffer pool to the later timed driver.

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
before final stable-state scoring. Mixed execution does not export checkpoints.

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

Trials reuse the index/PQ/GT; records go to `<dataset>/IVF/query/runs/`,
plans/summaries to `<dataset>/IVF/query/sweeps/<tag>/`.
Separate-process warmups do not warm later buffer pools.
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
  Mutable IVF preparation also fills pages locally, with backend allocation/writes
  caller-serialized even when routing/PQ insertion workers are parallel. Generic
  appends/backend allocation remain round-robin. Aborted retries keep their physical
  slot; deletion retains committed IDs/records without physical slot reuse.
  Source-row IDs and file framing are unchanged; physical slot placement
  can differ. Rebuild the builder and its library for new preparations; this
  cannot accelerate a process already running, and there is no partial-build
  resume or supported heap-only reconstruction command.
- The private, graph-free IVF heap owns its buffer pool. Batched coordinate fetches
  use short ledger locks, per-page read guards and FIFO aggregate pin admission
  shared with scalar reads and writes. Capacity is the smallest shard's frame count
  minus one (at least one); query batches use at most half of it (at least one),
  capped at 256. Pins release before admission credits. Tiny/skewed pools stay
  bounded, but admission/lock contention can reduce high-worker warm-cache
  throughput; this is not a universal concurrency speedup. Terminal read/AIO
  failures poison the heap/pool, wake admission waiters and refuse checkpoints;
  strict full-read/completion validation does not apply a blanket behavior change
  to unrelated legacy readers.
- `fetch_adapter_mutex_wait_us_total`, `fetch_pin_admission_wait_us_total` and
  `fetch_backend_pin_us_total` are **nested fetch wall-time subintervals**. Do not
  add them again to stage totals or interpret them as CPU/device utilization.
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

### Maintaining shared code without merging different contracts

Use the existing small helpers when adding IVF behavior; keep workload policy at
the call site rather than creating another orchestration layer:

| Shared owner | Responsibility |
|---|---|
| `benchmark/IVF/common.py` | File identities, counted source-row ranges, exclusive-path checks, bounded JSON and process primitives |
| `benchmark/IVF/settings.py` | Declared settings, admission and query flag formatting; callers validate warmup before any mutation |
| `benchmark/IVF/harness.py` | Bundle ownership and common native-open arguments; sequential workload sequencing stays explicit |
| `benchmark/IVF/common.sh` | Fresh work-directory claims, run-tag output selection and query/sequential prefix GT staging |
| `benchmark/IVF/experiment.sh` | SIFT100M recipe settings, fixed grid loops, GT preflight and successful private-heap cleanup |
| `benchmark/IVF/sift/_profile.sh` | Dataset/tool shell bootstrap; sequential and mixed arithmetic remain in separate profiles |
| `benchmark/scripts/check_gt_coverage.py` | Ordered live-state plans, GT plane layout and cached-row scratch estimates |
| `include/bufann/ivf_numeric_internal.h` | Unchecked ordered float ADC, double exact-L2 and distance/ID ordering; caller validation remains local |
| `src/bufann/ivf_random_internal.h` | Deterministic bounded draws; training/calibration sampling strategies stay separate |
| `src/bufann/ivf_artifacts.cpp` | Byte-copy mechanics; callers still own claims, stream checks, close and independent verification |
| `src/bufann/ivf_index.cpp` | Ordered component digest operations below each manifest family's explicit schema |
| `tests/ivf_training_settings.h` | Identical build/calibration training-option parsing; diagnostic defaults remain separate |
| `tests/ivf_cli_execution.cpp` | Native checkpoint progress, counter snapshots and timing shared by sequential/mixed updates |
| `tests/ivf/ivf_test_support.py` | Test-only binary framing and shared synthetic populations, with no import-time I/O or production oracle dependency |

Some apparent repetition is intentional. Do not merge immutable/mutable visibility
checks or memory policies, streaming/staged timing boundaries, sequential/mixed
schedulers, diagnostic defaults, atomic GT-record publication or heap poison/unwind
paths. Ordered manifest component helpers preserve each format's separate version
and suffix schema. Hashes before/after operations are integrity boundaries, not
redundant reads to cache away. Test fixture sharing must not share mutable files or
replace independent distance/recall oracles with production calculations. Keep the
README commands copyable; experiment logic belongs in tested scripts, not Markdown.


# Quickstart: SIFT
SIFT IVF testing from a new Linux computer.

This guide starts with a new **Ubuntu 24.04 x86-64 computer**, downloads the
project and IVF inputs, and runs full-scale query, insert, delete and mixed-update tests.
SPFresh is tested through its pre-existing infrastructure, not this guide.
You need sudo access, internet access and an AVX2-capable CPU.

Prefer **16 physical CPU cores, 128–256 GB RAM and at least 1 TB free SSD space**.
Index copies and retained checkpoints take much more space than the vector download
(12.8 GB for SIFT100M). Expect long-running jobs, especially training and mixed updates.

| Test | SIFT100M |
|---|---|
| Query | 90M base, all 10,000 queries |
| Insert | Insert 10M → 100M live |
| Delete | Delete 10M → 80M live |
| Mixed update | Insert 10M and delete 10M |

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
  libmkl-full-dev libomp-dev python3-numpy
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
an isolated installed-library wrapper, separate from IVF and existing baseline
build trees; it does not build a DiskANN graph or SPFresh infrastructure.
Choose a fresh helper build path; do not overwrite binaries used by active runs.
Warnings may occur; a nonzero exit is a failure, not a guaranteed successful build.
```bash
export IVF_PQ_HELPER_DIR="$PWD/build/ivf-pq-helpers-01"
cmake -S tests/ivf/pq_helpers -B "$IVF_PQ_HELPER_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/usr \
  -DMKL_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu \
  -DINTEL_OMP_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
cmake --build "$IVF_PQ_HELPER_DIR" \
  --target build_pq_pivot_standalone build_pq_codes_standalone -j4
```

These paths assume Ubuntu's packaged MKL layout.

## 3. Prepare the inputs once

This recipe takes the **first 100M uint8 vectors
from BIGANN/SIFT1B**, not a random sample. Other valid subsets can be supplied
through the dataset profile; regenerate matching GT and PQ for their exact row
order. 

You'll want at least the eval and data drives to have significant disk space.

Once generated, these artifacts can be uploaded and downloaded instead of regenerated.
```bash
export ANN_DATA_ROOT="$HOME/project/ann-data"
export DATASET=sift100m
export DATA_ROOT="$ANN_DATA_ROOT/$DATASET"
export DATA_BIN="$DATA_ROOT/base.bin" QUERY_BIN="$DATA_ROOT/query.10K.bin"
export GT_ARTIFACT_DIR="$DATA_ROOT/groundtruth"
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
  "$IVF_PQ_HELPER_DIR/build_pq_pivot_standalone" \
    uint8 "$DATA_BIN" "$IVF_PQ_PREFIX" 64
  "$IVF_PQ_HELPER_DIR/build_pq_codes_standalone" \
    uint8 "$DATA_BIN" "$IVF_PQ_PREFIX" 64 0
)
```

Download and PQ generation can take substantial time. GT is handled automatically
in step 4, after the workload plan is configured. The downloader checks
headers/ranges, refuses existing outputs and retains partial files; it does not
resume. Reuse completed inputs unchanged across runs; skip their generation when
using an existing verified dataset. The PQ helper's final `0` encodes all rows.

### Storage layout and reuse

```text
ann-data/sift100m/
  base.bin, query.10K.bin
  pq/pq64_*                         # shared pivots and full-source codes
  groundtruth/
    .gt/run-*/                      # deep GT, coverage evidence and generator logs
    derived/<identity>/truth.bin    # shared live-prefix / mutation / round GT
ann-eval/<experiment>/
  sift100m/IVF/                     # dataset/runner/workload
    query/
      data/, groundtruth/           # read-only links to shared inputs/truth
      index/                        # complete prepared base (retain)
      runs/<tag>/                   # results, neighbors, logs; private heaps disposable
    insert/
    delete/
    update/
```

The grid adds `nlist-<N>/` above the dataset directory. `EVAL_CACHED`
continues to hold verified trained-index caches, not shared GT or generated PQ.
`ANN_DATA_ROOT` defaults to `$HOME/project/ann-data`; dataset-specific roots and
explicit input paths still override it. The generic `benchmark/scripts/build_pq.sh`
now defaults to `ann-data/<dataset>/pq/pq32-base<NPTS_BASE>` (override `PQ_PREFIX`
and `PQ_CHUNKS`). Its base-only codes are **not** the full-source codes needed by
the IVF insertion recipe above; retain the explicit `IVF_PQ_PREFIX` selection.
Derived GT is reused across nlist values, experiments and runners only when its
deep-GT/query/tool hashes, source identity, output width and live-set filters
match. Source vectors must remain immutable: their resolved path, size and stat
identity are recorded rather than rehashing 100M rows for every GT derivation.
The exact full-source GT's provenance remains the caller's responsibility.
Eval cleanup does not delete shared GT; `ann-data` is not a permitted destructive
eval root. PQ components inside an index/private session are still bundle-owned,
not substitutes for the shared full-source PQ inputs.

Outputs use the `dataset/runner/workload` layout. No existing experiment
directories are moved or rewritten. Keep existing paths and link targets intact, including
the actual GT binary named in `selected_gt.txt`. The GT preflight also discovers
old `$DATA_ROOT/.gt/` runs without regenerating them. New GT goes under
`$GT_ARTIFACT_DIR` (default `ann-data/<dataset>/groundtruth`), even when reusing
an old base. Do not rename old prepared trees: saved metadata contains absolute
paths. No migration is required to continue the previous experiment.

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

For a **query-only grid over nlist, nprobe and router breadth**, use the
[copyable SIFT100M wide-grid instructions](#quickstart-sift100m-wide-query-only-grid)
**instead of this step** after completing steps 1–3. Both recipes use the checked-in
`experiment.sh` runner; this README is the only experiment document required.

**Training limit:** every main recipe uses **15 passes on all 90M base rows**,
plus final assignment: **1.44B searches**, admitted by the separate native
**3B assignment-search guard**. Conflicting `IVF_MAX_REPS` overrides are rejected
by the main runner; individual workload/native interfaces remain configurable.
This guard applies to **LV+ as well as the other trainers**. These are fixed pass
caps, not convergence-based stopping or paper-mandated iteration counts. Full
training can be expensive; LV+ does not reduce the number of sampled rows here.

**Separate PCA/routing allowance:** at 200K centroids and 128 dimensions, one
PCA covariance calculation counts `200000 * 128 * 129 / 2 = 1651200000`
operations, already above the inherited 1B routing-work default. The recipe
declares a finite 3B allowance for covariance plus solver/validation headroom.
It applies to training PCA and optimized final-router work; it does **not**
replace the separate 3B training-assignment cap or guarantee that every graph
operation fits. Keep guards enabled and retain any subsequent work-limit failure.

```bash
export EVAL_TEMPFILES="$HOME/project/ann-eval/run01"  # must not exist
export EVAL_CACHED="$HOME/project/ann-eval/cache"
export IVF_MEMORY_BUDGET_GB=96 IVF_QUERY_MEMORY_BUDGET_GB=96
export IVF_MAX_REPS=15
export IVF_INSERT_THREADS=8 IVF_INSERT_INPUT_POLICY=staged IVF_WARMUP_QUERIES=0
unset IVF_GT_FILE
bash benchmark/IVF/experiment.sh workloads --dry-run
```

Review the preview, then execute:

```bash
bash benchmark/IVF/experiment.sh workloads
```

The SIFT100M runner checks the profile and shared GT,
then prepares/runs **query, insert, delete, update**, stopping at the first failure.
It clears stale count, scheduler-seed, run-tag and cost-model overrides. Effective
settings are recorded in `settings.env`; subprocesses do not change your shell exports.
The existing prepare/run scripts perform input, cache and native-result validation.

| Recipe setting | Value |
|---|---|
| Training | Full-base LV+, 20 workers, seed 0; 15 passes plus final assignment |
| Routing | PCA-refined for assignment/query; PCA variance 0.95; both rerank factors 3 |
| Workload search | nlist 200K, nprobe 64, router breadth 128, Recall@10 |
| Resources | 96 GiB build/query allowances, 262144 frames, 8 query workers |
| Updates | 10% tail inserts/base deletes; 10 mixed rounds; update-driven, 16 foreground workers |
| Sequential insertion | 8 workers, staged inputs; override with `IVF_INSERT_THREADS=1 IVF_INSERT_INPUT_POLICY=streaming` |
| Warmup | 0 for workloads; 64 for grids, unless explicitly overridden |

Resources and execution-policy exports above remain overridable; main recipes fix 15 passes.
Other training/search settings are fixed; use the individual workload scripts for
other experiments. `IVF_TIMEOUT_S` defaults to 86400 seconds **per script or GT
command**, using GNU `timeout`. There are no total/per-index deadline managers.
Check free space on both input and work/cache filesystems before starting.

`ensure_gt.py` ignores nonexistent shell paths, finds
existing caches, and checks every query against every declared live population.
It reuses a passing cache unchanged. Otherwise it recommends a sufficient depth,
generates exact full-source GT with the existing helper, and checks the new file
before returning its path. With no cache, it starts at K=1000; that is a starting
depth, not a coverage guarantee. Set `IVF_GENERATE_GT=0` for `experiment.sh` to
check/recommend without permitting an exact scan. For custom GT plans, call
`ensure_gt.py --help`; `--gt-generator` selects a helper and `--cached-gt` prefers
an existing cache. `IVF_BUILD_DIR` supplies the default generator location.

No timestamps, manual K extraction, or report-path bookkeeping are needed.
Caches, reports and generator logs are preserved under `$GT_ARTIFACT_DIR/.gt/run-0001/`,
`run-0002/`, etc.; rerunning rechecks and reuses completed caches (underflowing ones
can seed recommendation). Failed/partial generation outputs are never selected.
Existing GT, PQ and index files are never overwritten. A malformed cache is an
error, not permission to bypass validation. On failure, inspect that run's
`error.json` and `generate-k*.log`, fix the cause, and rerun the same command.
`ensure_gt.py` exits 3 when generation is needed and 1/2 on errors. No failure
returns a GT path; the experiment runner exits nonzero and retains the phase log.
The runner checks the GT command's exit status and returned path before preparing
any index; a failed GT gate cannot fall through to benchmark execution.

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
Sequential insertion uses `IVF_INSERT_THREADS` (one by default); deletion remains
serial. `QUERY_THREADS` controls later scoring. These insertion controls do not
change the mixed worker split selected by `NTHREADS`. Centroids, PCA and PQ stay
frozen across updates; the execution controls do not change ANN search budgets.

If selecting the optional staged-input policy, all requested insertion coordinates
must pass memory admission before mutation. For this uint8/128D recipe the staged
payload is 1.28 GB for SIFT100M's 10M inserts,
in addition to pool/list/router/worker memory. Its version-2 insertion timing
excludes source loading; streaming version 1 includes worker reads. Do not compare
the two foreground rates as an isolated worker speedup. Keep preparation and
checkpoint costs alongside mutation throughput.

`IVF_WARMUP_QUERIES=64`, if explicitly selected, runs 64 serial frozen-centroid
queries on the same index object before the timed final pass for sequential/query
runs. In mixed runs it instead executes once before the continuous workload.
It uses neither
final queries nor GT, discards results and reports separate warmup time. It does
not count toward the 10,000 scored queries or mixed foreground queries, and does
not transfer an application cache between the mutation and query processes.
Declare a fixed policy before evaluation; zero warmup is not a cold-cache guarantee.
See [Insertion workers, input staging and same-process warmup](#insertion-workers-input-staging-and-same-process-warmup)
for native options, failure behavior and matched-policy examples.

These settings run **10 rounds**, each inserting/deleting 1M rows.
There is **no required query ratio or total query quota**.
IVF's `update_driven` mode replays the query pool until all mutations complete
across all rounds, joins admitted queries, then scores all 10,000 queries once.
One native process, mutable session and worker pool span the entire workload.
Mutation rounds stay ordered; query workers can cross boundaries. Coordinates
use one reusable largest-round staging buffer; initial staging is outside
`mixed_wall_s`, later-round staging is included while queries continue. There is
no checkpoint, reopen, worker restart, intermediate warmup or per-round recall.
Record
the actual query count and overlap. This workload does not require reproducing
the external SPFresh tester's scheduler or scoring cadence.

## 5. Read results and repeat

Outputs are under `$EVAL_TEMPFILES/$DATASET/IVF/` in `query/`, `insert/`,
`delete/` and `update/`.
Retain logs, structured results and input identities.
For sequential/query runs, inspect `execution_protocol.json` for the declared
input/worker/warmup policy and separate process wall costs; `result.json` retains
the native scored-query metrics unchanged. Mutation completion costs live in
`mutation_result.json`. The driver writes these metrics directly; no duplicate
native/extracted JSON files are produced. Mixed runs retain
`rounds/round_<r>.json` accounting records (no intermediate recall), one native
root `result.json`, final `neighbors.bin`, one private heap, `round_schedule.txt`,
`run.log`, and a version-3 `summary.json`. Query attribution to rounds is by
completion, including queries crossing a boundary. `mixed_wall_s` is the single
continuous interval, not summed worker time. Root final-pass `query_qps` is not
concurrent foreground QPS. No mutable checkpoint is created by mixed execution;
sequential checkpoint APIs remain pending the next implementation phase. Keep failed logs and
owned heaps; a failed mutation session must not be checkpointed or reused.

For three query repetitions at the same frozen settings:

```bash
bash benchmark/IVF/sift/query_sweep.sh \
  --tag full-scale-repeated \
  --nprobes 32 --rerank-factors 3 --threads 8 \
  --repeats 3 --warmups 0 \
  --memory-budget-gb 96 --buffer-frames 262144 --timeout 86400
```

This inherits the fixed `IVF_WARMUP_QUERIES` from step 4. With its default `0`,
there is no explicit centroid warmup. Sweep `--warmups 1`, if separately declared,
would add a discarded **whole driver process** per configuration; it does not
warm the next process's buffer pool and is not equivalent to centroid warmup.
Keep both policies consistent across repetitions and record them with results.

For another complete run, choose a fresh work root and repeat the recipe in step 4
without changing the frozen settings:

```bash
export EVAL_TEMPFILES="$HOME/project/ann-eval/run02"
bash benchmark/IVF/experiment.sh workloads
```

Choose a new root for each repetition so preparation starts from the same IVF
base. Do not delete failed evidence or change settings in response to final-set
results. Tagged reruns from an existing IVF prepared base are also supported;
the fresh-root recipe makes each repetition explicit.

# Quickstart: SIFT100M wide query-only grid

Use this **instead of step 4** of the SIFT quickstart after completing its
[dependencies/build](#1-install-dependencies-and-get-the-project) and
[SIFT100M input preparation](#3-prepare-the-inputs-once). Rebuild the IVF tools
if they predate the query-only mutable-session change. Run from the repository
root in the same persistent Bash terminal. No insert, delete or mixed workload
is prepared or executed.

This experiment prepares an independent index at each of the following **nlist
values, in this exact order**:

```text
100, 1000, 10000, 100000, 200000, 300000
```

For each index, search a wide, fixed powers-of-two probe grid, never exceeding
that index's list count:

| nlist | nprobe values | Query passes |
|---:|---|---:|
| 100 | 1, 2, 4, 8, 16, 32, 64 | 7 |
| 1,000 | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 | 10 |
| 10,000; 100,000; 200,000; 300,000 | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 | 14 each |

There are **73 query passes**, each scoring all 10,000 queries, and six index
preparations/cache restorations. The initial corpus remains **90M of the 100M
source rows**, exactly as in the ordinary SIFT100M quickstart; this does not
silently switch to a 100M-live index. PQ64 and checked GT are reused across
indexes, but centroids and routers are trained independently at each nlist.

Fixed settings are full-base LV+ training with 15 passes, PCA-refined query/assignment
routing, PCA variance 0.95, both rerank factors 3, requested router breadth 128,
eight query workers, and 64 serial centroid warmup queries per pass. Effective
query traversal breadth is `max(128, min(nlist, 3*nprobe))`, so high probes also
increase router effort. This is not an exact reproduction of Lindorm's PQ4/SIMD
implementation or a matched-recall comparison by itself.

Query-only execution forks each immutable base into a **private mutable session
without updates**. Opening/copying and warmup stay outside native query timing;
they remain included in process/workflow timing and timeout budgets. After a pass
exits successfully, its results are validated and its CSV row is flushed, then
its private `query.heap` and `query.heap.pq.bin` are deleted. Results, neighbors,
logs, prepared bases, and verified cache bundles are retained. Failed passes keep
their artifacts; cleanup never recursively removes directories.

## Wide-grid setup and preview

Retain the verified `DATA_BIN`, `QUERY_BIN`, `IVF_PQ_PREFIX`, and `IVF_BUILD_DIR`
from steps 1–3. Choose a fresh absolute work root. Each command has a one-day
timeout by default (`IVF_TIMEOUT_S`); the complete grid may take days.

```bash
export DATASET=sift100m
export EVAL_TEMPFILES="$HOME/project/ann-eval/sift100m-wide-grid-01"  # must not exist
export EVAL_CACHED="$HOME/project/ann-eval/cache"
export IVF_MEMORY_BUDGET_GB=96 IVF_QUERY_MEMORY_BUDGET_GB=96
export IVF_MAX_REPS=15
export IVF_BUFFER_FRAMES=262144 QUERY_THREADS=8 IVF_WARMUP_QUERIES=64
unset IVF_GT_FILE
bash benchmark/IVF/experiment.sh sift100m-wide --dry-run
```

Review the preview, then execute (this can take days):

```bash
bash benchmark/IVF/experiment.sh sift100m-wide
```

### Follow-up: reuse the first run and refine 100K-400K

For a second, exploratory run after the wide grid, use the commands below instead
of the wide-grid commands. Keep the previous run intact;
successful private query heaps need not exist, but its prepared base indexes,
`prepared.json` files and `selected_gt.txt` must still exist. Use the same verified
source/query/PQ inputs and retain the original base bundles unchanged.

```bash
export DATASET=sift100m
export IVF_GRID_PREVIOUS_ROOT="$HOME/project/ann-eval/sift100m-wide-grid-01"
export EVAL_TEMPFILES="$HOME/project/ann-eval/sift100m-refine-grid-01"  # fresh
export EVAL_CACHED="$HOME/project/ann-eval/cache"
# Keep the SAME DATA_BIN, QUERY_BIN, IVF_PQ_PREFIX and trained-base settings.
export IVF_MEMORY_BUDGET_GB=96 IVF_QUERY_MEMORY_BUDGET_GB=96
export IVF_MAX_REPS=15
export IVF_BUFFER_FRAMES=262144 QUERY_THREADS=8 IVF_WARMUP_QUERIES=64
unset IVF_GT_FILE
bash benchmark/IVF/experiment.sh sift100m-refine --dry-run
```

Review the preview, then execute:

```bash
bash benchmark/IVF/experiment.sh sift100m-refine
```

This mode executes **80 passes**, in nlist order **100K, 200K, 300K, 400K**:

- Probes at every nlist: **16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192,
  224, 256, 384, 512**. This tightens the region around 90% and 95% recall and
  retains low/high-effort anchors. A target not reached is reported, not assumed.
- Requested breadth **128** everywhere; additionally **64 at probes 16/32**,
  **256 at probes 48/64**, and **512 at probe 128**. These produce distinct
  effective breadths. In particular, 64 and 128 are identical at probes >=48,
  and 256 is ineffective once `3*nprobe >=256`; those duplicate trials are omitted.
- Both rerank factors remain **3**, PCA variance remains **0.95**, and all query
  passes still score the same 10,000 queries. Do not silently change the vector
  shortlist or scoring definition while comparing nlist or breadth.

**Reuse:** the first three indexes are queried directly from their existing
prepared work trees, not retrained or re-copied into new preparations. New tagged
query outputs live under those prepared trees' `runs/` directories; original
results and prepared bases are not overwritten. Tags incorporate the new root's
identity. Only 400K is prepared in the new root, with normal verified-cache reuse
if a matching build already exists. PQ and source files are reused unchanged.
Centroids/graphs/PCA cannot be reused between different nlist values.

The previous `selected_gt.txt` supplies the full-source GT cache. Coverage is
rechecked **without `--generate`**: missing/insufficient GT stops this follow-up
rather than triggering a new full exact scan. Existing per-base derived GT is
reused for old indexes; 400K derives equivalent base GT from the shared cache.
Neither reuse nor deeper full-source GT guarantees that the derived ten-column
scoring file contains all boundary ties (see saturation discussion below).

The 400K build uses a finite **5B routing-work allowance** because PCA covariance
alone costs `400000 * 128 * 129 / 2 = 3302400000` operations. Existing bases retain
their original settings and identities; their 3B allowance is not edited. All
training guards remain enabled. Rebuild/restore is allowed only for the new 400K
index; missing previous bases cause a failure, not an accidental retraining job.

The new root holds `settings.env`, `grid.tsv`, `results.csv`, logs, and `heap-cleanup.log`.
Use each CSV row's `output_dir` to find its native result: reused-index outputs
are deliberately outside the new root. Successful private heaps are deleted as
before, and originals/caches are preserved. This is post-observation exploratory
tuning, not an independent held-out validation of the original sweep.

### Runner options and retained evidence

`bash benchmark/IVF/experiment.sh --help` shows the three supported recipes.
The optional second argument `--dry-run` previews the schedule without reading
inputs or creating paths. Paths and resources use environment variables, like the
BufANN/SPFresh scripts. There are no legacy modes, compatibility options, nested
deadline controllers or extra JSON plan/summary formats. No grid includes nlist 1M.

Keep `settings.env`, `grid.tsv`, `results.csv`, `selected_gt.txt` and logs. The CSV
contains native nlist/probe/breadth, Recall@10, QPS, query time and output path.
Native `result.json` and `execution_protocol.json` remain in each run directory.
The runner stops on failure and preserves its files; inspect the last command's
log and retry in a fresh root. It checks previous `prepared.json` files exist;
you are responsible for choosing matching previous experiment inputs/settings.
The lower-level harness still verifies prepared query/GT/index identities.
Freeze the settings and warmup policy before evaluation.

### Investigating the approximately 99.559% recall ceiling

The first wide grid approaches 99.559% at several different list counts. Over
10,000 queries at k=10, this corresponds to about **441 missing hit slots out of
100,000**, not necessarily 441 failed queries. A stable aggregate does not prove
the same individual neighbors were missed at each setting.

Three explanations must be separated:

1. **PQ shortlist loss:** the scanner retains only `ceil(10 * 3) = 30` approximate
   candidates, even when it scans millions of vectors. Exact reranking cannot
   recover a true neighbor discarded before fetching. Full list coverage does
   not imply exact search when this shortlist is bounded.
2. **List coverage/routing loss:** near-saturation is not proof that every useful
   list was probed. Use an all-lists diagnostic to remove this approximation.
3. **Scoring ties:** `common.sh::ivf_stage_prefix_truth` normally derives GT with
   exactly `RECALL_AT=10` columns. The native scorer expands ties only into extra
   columns actually present. It cannot accept alternative equally distant IDs
   missing from a ten-column file, despite the tie-aware policy label. With uint8
   SIFT and duplicate/equidistant descriptors, this is a credible explanation for
   at least part of the ceiling. It is not established without auditing results.

Exact uint8 squared distances in 128 dimensions are below 2^24, so the float32
distance export can represent those integer distances exactly. This makes a
boundary-distance/tie audit useful before blaming floating-point precision.

#### First: rescore existing neighbors, without running ANN again

Run the existing-GT auditor on a saturated result; no heaps or vectors are read,
and no new truth is generated:

```bash
previous="${IVF_GRID_PREVIOUS_ROOT:?Set the original wide-grid root}"
deep_gt=$(cat "$previous/selected_gt.txt")
base_work="$previous/nlist-100/sift100m/IVF/query"
python3 benchmark/IVF/audit_recall.py \
  --deep-gt "$deep_gt" --base-count 90000000 --k 10 \
  --results "$base_work/runs/np64_rb128/neighbors.bin"
```

Repeat for another saturated index, such as 10K/probe2048. The auditor filters
full-source GT to IDs below 90M and compares strict first-ten-ID recall with
expanded boundary-tie recall. It requires a stored distance strictly beyond the
live kth distance to prove all boundary ties were included; otherwise expanded
recall is reported as unavailable, not as an exact corrected score. Source-row
ordering, query ordering, and the exact GT's provenance remain your responsibility.
This audits existing GT; it does not establish its correctness by recomputation.

- If corrected recall reaches 100%, the apparent ceiling was scoring-related
  for that result, not necessarily a PQ accuracy limit.
- If returned distances exceed the exact kth distance after tie correction,
  genuine retrieval misses remain.
- IDs missing from the expanded truth despite a distance within the cutoff
  require investigation of GT coverage, IDs, or result-distance consistency.
- Keep old native scores unchanged. Label rescored values and scoring policy
  separately; the follow-up nlist grid deliberately retains the original scoring
  definition so its main results remain comparable.

#### Second: bounded all-lists/vector-rerank diagnostic

This is **optional**, separate from the 80-point grid, and deliberately varies
vector rerank factor as a diagnostic rather than as the fixed-factor main
configuration. It reuses the old 100-list base and existing deeper GT. It searches
the first 256 queries at **nprobe=100**, returning the top ten after fetching 30,
100, or 300 PQ candidates. Probing all lists bypasses approximate centroid
selection and removes assignment/list-coverage loss for the live base.

```bash
python3 benchmark/IVF/recall_diagnostic.py \
  --previous-root "${IVF_GRID_PREVIOUS_ROOT:?Set the original wide-grid root}" \
  --query "$QUERY_BIN" \
  --output "$HOME/project/ann-eval/sift100m-recall-audit-01"  # must not exist
```

The runner validates the original query identity, creates a prefix subset, and
retains native results, neighbors, audit reports and logs under `rf3/`, `rf10/`
and `rf30/`. It enforces the all-lists scan count and removes only successful
private query heaps. Failed passes keep their files. `--query-count` defaults to
256; `--timeout-s` defaults to 21600 per pass. Native scoring/results are not rewritten.

This diagnostic still scans all 90M codes per query and can take minutes or more;
the 256-query subset bounds cost but may contain few saturation misses. The runner
checks `scanned_codes == query_count * 90000000`. A no-miss subset is not
proof that the complete 10K query set has no residual error.

If increasing the shortlist improves **tie-corrected, all-lists recall**, PQ
shortlisting contributes to the error. If strict ID recall stalls but corrected
recall is already 100%, rerank tuning cannot resolve the scoring artifact. If
errors remain at larger factors, inspect the missed queries before assuming that
even larger probing or router breadth can help. No native search/scoring code or
existing benchmark records are rewritten by either diagnostic.
