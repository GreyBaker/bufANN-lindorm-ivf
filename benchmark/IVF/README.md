# IVF benchmark guide

Query-only benchmark runs fork the prepared immutable base into a private mutable session before warmup or query timing, without inserting, deleting, or creating a checkpoint. Native `search_only` without `--private_heap_path` still supports immutable search; supplying a fresh private path with immutable input selects the no-update fork.

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
| `IVF_WARMUP_QUERIES=0..100000` | `--warmup_queries N` | `0` | Query-producing calls; in mixed runs only the post-checkpoint final query pass |

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
  after mutation/checkpoint when applicable. Its results are discarded, never
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
grid guide's terminal blocks copyable; they are not executable Markdown templates.


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
**instead of this step** after completing steps 1–3. No separate orchestrator
script is needed; use a fresh work root and declare the grid/time budgets first.

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
export IVF_QUERY_MEMORY_BUDGET_GB=96 OMP_NUM_THREADS=20 MKL_NUM_THREADS=1
export IVF_TRAINING_RERANK_FACTOR=3 IVF_TRAINING_REFINE_BREADTH=128

# Expected starting configuration
export IVF_NLIST=200000 IVF_NPROBE=64 IVF_RERANK_FACTOR=3 QUERY_THREADS=8 RECALL_AT=10
export IVF_ROUTING_WORK_LIMIT=3000000000
# Enable optimized final routing separately from LV+ training.
export IVF_OPTIMIZED_MODES=1 IVF_ASSIGNMENT_ROUTER_MODE=pca-refined IVF_BUILD_ROUTER_RERANK_FACTOR=3 IVF_PCA_VARIANCE=0.95 IVF_REFINE_EF=128 IVF_QUERY_BREADTH=128 IVF_INSERT_BREADTH=64
export IVF_ROUTER_MODE=pca-refined IVF_ROUTER_RERANK_FACTOR=3 IVF_ROUTER_BREADTH=128

# Full profile update counts; ten mixed rounds in either dataset.
export UPDATE_PERCENT=10 UPDATE_BATCH=$((UPDATE_POINTS / 10)) IVF_SCHEDULER=update_driven NTHREADS=16 EVAL_TEMPFILES="$HOME/project/ann-eval/run01" EVAL_CACHED="$HOME/project/ann-eval/cache"
export IVF_BUFFER_FRAMES=262144
# Freeze execution policy before running final queries; preserve default behavior.
# export IVF_INSERT_THREADS=1 IVF_INSERT_INPUT_POLICY=streaming IVF_WARMUP_QUERIES=0
# Optional alternative, declared in advance:
export IVF_INSERT_THREADS=8 IVF_INSERT_INPUT_POLICY=staged IVF_WARMUP_QUERIES=0
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
Sequential insertion uses `IVF_INSERT_THREADS` (one by default); deletion remains
serial. `QUERY_THREADS` controls later scoring. These insertion controls do not
change the mixed worker split selected by `NTHREADS`. Centroids, PCA and PQ stay
frozen across updates; the execution controls do not change ANN search budgets.

If selecting the optional staged-input policy, all requested insertion coordinates
must pass memory admission before mutation. For this uint8/128D recipe the staged
payload is 128 MB for SIFT10M's 1M inserts or 1.28 GB for SIFT100M's 10M inserts,
in addition to pool/list/router/worker memory. Its version-2 insertion timing
excludes source loading; streaming version 1 includes worker reads. Do not compare
the two foreground rates as an isolated worker speedup. Keep preparation and
checkpoint costs alongside mutation throughput.

`IVF_WARMUP_QUERIES=64`, if explicitly selected, runs 64 serial frozen-centroid
queries on the same index object before each timed final pass. It uses neither
final queries nor GT, discards results and reports separate warmup time. It does
not count toward the 10,000 scored queries or mixed foreground queries, and does
not transfer an application cache between the mutation and query processes.
Declare a fixed policy before evaluation; zero warmup is not a cold-cache guarantee.
See [Insertion workers, input staging and same-process warmup](#insertion-workers-input-staging-and-same-process-warmup)
for native options, failure behavior and matched-policy examples.

These settings run **10 rounds**: each inserts/deletes 100K rows for SIFT10M or
1M rows for SIFT100M. There is **no required query ratio or total query quota**.
IVF's `update_driven` mode replays the query pool until all mutations complete,
joins admitted queries, then scores all 10,000 queries after each round. Record
the actual query count and overlap. This workload does not require reproducing
the external SPFresh tester's scheduler or scoring cadence.

## 5. Read results and repeat

Outputs are under `$EVAL_TEMPFILES/$DATASET/IVF/` in `query/`, `insert/`,
`delete/` and `update/`. Retain logs, structured results and input identities.
For sequential/query runs, inspect `execution_protocol.json` for the declared
input/worker/warmup policy and separate process wall costs; `result.json` retains
the native scored-query metrics unchanged. Mutation completion costs live in
`mutation_result.json`. The driver writes these metrics directly; no duplicate
native/extracted JSON files are produced. Mixed runs retain
`rounds/round_<r>/result.json` and a version-2 `summary.json` with relative
`round_results` paths and aggregate counts/times, not embedded round records or a
duplicate root `result.json`. Their final-pass `query_qps` is not concurrent
foreground QPS. Keep failed logs and
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

For another complete run, choose a fresh work root and repeat the loop in step 4
without changing the frozen settings:

```bash
unset IVF_RUN_TAG
export EVAL_TEMPFILES="$HOME/project/ann-eval/run02"
# Repeat only the workload loop from step 4.
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

Fixed settings are full-base LV+ training with a 10-pass cap, PCA-refined query/assignment
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
from steps 1–3. Choose a fresh absolute work root. The larger deadlines below are
explicit budgets, **not estimates that the full grid will finish within them**.

```bash
export DATASET=sift100m
export EVAL_TEMPFILES="$HOME/project/ann-eval/sift100m-wide-grid-01"  # must not exist
export EVAL_CACHED="$HOME/project/ann-eval/cache"
export IVF_GRID_MODE=sift100m-wide
export IVF_GRID_TOTAL_TIMEOUT_S=604800   # seven days for the complete workflow
export IVF_GRID_NLIST_TIMEOUT_S=86400   # one day per index, including its queries
export IVF_GRID_QUERY_TIMEOUT_S=21600  # six hours per query process/workflow
export IVF_MEMORY_BUDGET_GB=96 IVF_QUERY_MEMORY_BUDGET_GB=96
export IVF_BUFFER_FRAMES=262144 QUERY_THREADS=8 IVF_WARMUP_QUERIES=64
unset IVF_GT_FILE IVF_COST_MODEL_REPORT IVF_RUN_TAG
```

### Usage

Next paste the **entire block from `(` through the closing `)`**, including its
Python here-document. Everything required is in this README; no separate guide
file is read. Optionally set `IVF_GRID_DRY_RUN=1` first to preview the schedule
without creating artifacts; unset it and paste the same block again to execute.
Do not change the plan or warmup policy in response to final-query results.

```bash
(
  set -euo pipefail
  export IVF_GRID_REPO="$PWD"
  export EVAL_TEMPFILES="${EVAL_TEMPFILES:-$HOME/project/ann-eval/nlist-grid-01}"
  export EVAL_CACHED="${EVAL_CACHED:-$HOME/project/ann-eval/cache}"
  export IVF_GRID_DRY_RUN="${IVF_GRID_DRY_RUN:-0}"  # 1 previews without reading inputs
   export IVF_GRID_MODE="${IVF_GRID_MODE:-sift100m-wide}"  # also coarse or expanded
  export IVF_GRID_TOTAL_TIMEOUT_S="${IVF_GRID_TOTAL_TIMEOUT_S:-21600}"
  export IVF_GRID_NLIST_TIMEOUT_S="${IVF_GRID_NLIST_TIMEOUT_S:-7200}"
  export IVF_GRID_QUERY_TIMEOUT_S="${IVF_GRID_QUERY_TIMEOUT_S:-900}"
  export IVF_WARMUP_QUERIES="${IVF_WARMUP_QUERIES:-64}"  # 0 disables centroid warmup
  exec python3 - <<'PY'
import csv
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

env = dict(os.environ)
repo = Path(env['IVF_GRID_REPO'])
mode = env['IVF_GRID_MODE']
if mode not in ('coarse', 'expanded', 'sift100m-wide'):
    raise ValueError('IVF_GRID_MODE must be coarse, expanded or sift100m-wide')
wide = mode == 'sift100m-wide'
if wide and env.get('DATASET', 'sift100m') != 'sift100m':
    raise ValueError('sift100m-wide requires DATASET=sift100m')
if env['IVF_GRID_DRY_RUN'] not in ('0', '1'):
    raise ValueError('IVF_GRID_DRY_RUN must be 0 or 1')
limits = {key: int(env['IVF_GRID_' + key.upper() + '_TIMEOUT_S'])
          for key in ('total', 'nlist', 'query')}
if min(limits.values()) <= 0:
    raise ValueError('timeouts must be positive seconds')
nlists = (100, 1000, 10000, 100000, 200000, 300000)
coarse = {100: (1, 2, 4, 8), 1000: (1, 4, 16, 64), 10000: (4, 16, 64, 256)}
points, omissions = [], []
for nlist in nlists:
    probes = (tuple(2 ** i for i in range(14) if 2 ** i <= nlist) if wide else
              coarse.get(nlist, (16, 64, 256, 1024)) if mode == 'coarse' else
              tuple(2 ** i for i in range(0 if nlist <= 10000 else 4, 11) if 2 ** i < nlist))
    for probe in probes:
        seen = set()
        for breadth in ((128, 256, 512) if mode == 'expanded' and probe <= 128 else (128,)):
            effective = max(breadth, min(nlist, 3 * probe))
            reason = ('extra breadth exceeds nlist' if breadth != 128 and breadth > nlist
                      else 'duplicate effective breadth' if effective in seen else None)
            point = dict(nlist=nlist, nprobe=probe, router_breadth=breadth,
                         effective_router_breadth=effective, tag=f'np{probe}_rb{breadth}')
            if reason:
                omissions.append(dict(**point, reason=reason))
                continue
            seen.add(effective)
            points.append(point)
if env['IVF_GRID_DRY_RUN'] == '1':
    print('nlist\tnprobe\trouter_breadth\teffective_router_breadth')
    for p in points:
        print(*(p[k] for k in ('nlist', 'nprobe', 'router_breadth', 'effective_router_breadth')), sep='\t')
    print(f'{mode}: {len(points)} query passes, {len(nlists)} index preparations; no execution')
    raise SystemExit(0)

if not (repo / 'benchmark/IVF/sift/query_prepare.sh').is_file():
    raise ValueError('paste this block from the IVF repository root')
dataset = env.get('DATASET', 'sift100m')
profiles = {'sift10m': (10000000, 9000000, 1000000, 15),
            'sift100m': (100000000, 90000000, 10000000, 10)}
if dataset not in profiles:
    raise ValueError('this full-scale recipe supports sift10m or sift100m, not float SIFT1M')
full, base, updates, reps = profiles[dataset]
if not env.get('IVF_PQ_PREFIX') or not env.get('EVAL_TEMPFILES'):
    raise ValueError('export IVF_PQ_PREFIX and a fresh absolute EVAL_TEMPFILES root')
if env.get('IVF_GT_FILE'):
    raise ValueError('unset IVF_GT_FILE; this recipe requires coverage-checked GT_BIN_KCACHED')
root = Path(env['EVAL_TEMPFILES'])
cache = Path(env['EVAL_CACHED'])
if not root.is_absolute() or not cache.is_absolute():
    raise ValueError('EVAL_TEMPFILES and EVAL_CACHED must be absolute')
if root.exists() or root.is_symlink():
    raise FileExistsError(f'preserve the existing root and choose a new one: {root}')
for key in ('INSERT_POINTS', 'DELETE_POINTS', 'QUERY_POINTS', 'UPDATE_QUERY_BATCH', 'QUERY_RATIO',
            'INSERT_CAP', 'DELETE_CAP', 'IVF_RUN_TAG', 'IVF_COST_MODEL_REPORT', 'IVF_SCHEDULER_SEED'):
    env.pop(key, None)
data_root = Path(env.get('DATA_ROOT', str(Path.home() / 'project/ann-data' / dataset)))
native = Path(env.get('IVF_BUILD_DIR', str(repo / 'build/ivf-benchmark'))).resolve()
env.update(DATASET=dataset, DATA_BIN=str(Path(env.get('DATA_BIN', str(data_root / 'base.bin'))).resolve()),
           QUERY_BIN=str(Path(env.get('QUERY_BIN', str(data_root / 'query.10K.bin'))).resolve()),
           EVAL_CACHED=str(cache), IVF_BUILD_DIR=str(native),
           IVF_BUILD_BIN=str(Path(env.get('IVF_BUILD_BIN', str(native / 'tests/ivf_build'))).resolve()),
           IVF_DRIVER_BIN=str(Path(env.get('IVF_DRIVER_BIN', str(native / 'tests/ivf_driver'))).resolve()),
           IVF_SAMPLE_RATE='1', IVF_MAX_REPS=str(reps), IVF_TRAINER_MODE='lv-plus',
           IVF_INIT='random_partition', IVF_TRAINING_BREADTH='64', IVF_SEED='0', IVF_BUILD_THREADS='20',
           IVF_TRAINING_RERANK_FACTOR='3', IVF_TRAINING_REFINE_BREADTH='128',
           IVF_MEMORY_BUDGET_GB=env.get('IVF_MEMORY_BUDGET_GB', '96'),
           IVF_QUERY_MEMORY_BUDGET_GB=env.get('IVF_QUERY_MEMORY_BUDGET_GB', env.get('IVF_MEMORY_BUDGET_GB', '96')),
           IVF_BUFFER_FRAMES=env.get('IVF_BUFFER_FRAMES', '262144'), IVF_PAGE_SIZE='4096',
           OMP_NUM_THREADS='20', OMP_THREAD_LIMIT='20', MKL_NUM_THREADS='1',
           QUERY_THREADS=env.get('QUERY_THREADS', '8'), RECALL_AT='10',
           IVF_OPTIMIZED_MODES='1', IVF_ASSIGNMENT_ROUTER_MODE='pca-refined',
           IVF_BUILD_ROUTER_RERANK_FACTOR='3', IVF_PCA_VARIANCE='0.95', IVF_REFINE_EF='128',
           IVF_QUERY_BREADTH='128', IVF_INSERT_BREADTH='64', IVF_ROUTING_WORK_LIMIT='3000000000',
           IVF_ROUTER_MODE='pca-refined', IVF_ROUTER_RERANK_FACTOR='3', IVF_RERANK_FACTOR='3',
           IVF_ROUTER_BREADTH='128', IVF_NPROBE='1')
env['IVF_PQ_PREFIX'] = str(Path(env['IVF_PQ_PREFIX']).resolve())
for key in ('IVF_BUILD_BIN', 'IVF_DRIVER_BIN'):
    if not os.access(env[key], os.X_OK) or not Path(env[key]).is_file():
        raise ValueError(f'missing executable {key}; complete quickstart step 2')
for path, rows, dim in ((Path(env['DATA_BIN']), full, 128), (Path(env['QUERY_BIN']), 10000, 128),
                         (Path(env['IVF_PQ_PREFIX'] + '_pq_compressed.bin'), full, 64)):
    with path.open('rb') as source:
        header = source.read(8)
    if header != struct.pack('<II', rows, dim) or path.stat().st_size != 8 + rows * dim:
        raise ValueError(f'wrong full uint8 population/shape: {path}')
if not Path(env['IVF_PQ_PREFIX'] + '_pq_pivots.bin').is_file():
    raise ValueError('missing PQ pivots')

root.mkdir(parents=True, exist_ok=False)
env['INCREMENTAL_GT_BIN'] = str(root / 'incremental_groundtruth')
started = time.monotonic()
global_deadline = started + limits['total']
rows, phases = [], []

def save(name, value):
    with (root / name).open('x') as out:
        json.dump(value, out, indent=2)
        out.write('\n')

save('plan.json', dict(mode=mode, dataset=dataset, source_rows=full, base_rows=base,
     execution_mode='pasted_bash_subshell', queries_per_point=10000, points=points,
     omissions=omissions, timeouts_s=limits,
     settings={k: v for k, v in env.items() if k.startswith('IVF_') or k in (
         'DATASET', 'DATA_BIN', 'QUERY_BIN', 'EVAL_CACHED', 'BENCH_DATASETS_DIR',
         'INCREMENTAL_GT_BIN', 'QUERY_THREADS', 'RECALL_AT', 'OMP_NUM_THREADS',
         'OMP_THREAD_LIMIT', 'MKL_NUM_THREADS')}, selected_configuration=None,
      cache_policy='separate query drivers; no cache drop; hashing/open/copies may warm caches',
      private_query_heap_policy=('delete_successful_pass_private_heap_and_pq_after_result_recording'
                                 if wide else 'retain'),
     gt_policy='existing conservative quickstart four-workload coverage plan; no mutation execution'))

def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

def interrupted(signum, frame):
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, interrupted)

def run(name, command, child_env, deadline, stdout_path=None):
    remaining = min(deadline, global_deadline) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f'budget exhausted before {name}; phase not started')
    phase_start = time.monotonic()
    record = dict(name=name, command=command, timeout_s=remaining)
    process = None
    try:
        with (root / (name + '.log')).open('x') as log:
            if stdout_path is None:
                process = subprocess.Popen(command, cwd=repo, env=child_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = process.wait(timeout=remaining)
            else:
                with stdout_path.open('x') as output:
                    process = subprocess.Popen(command, cwd=repo, env=child_env,
                        stdout=output, stderr=log, start_new_session=True)
                    code = process.wait(timeout=remaining)
        record['returncode'] = code
        if code:
            raise RuntimeError(f'{name} failed ({code}); see retained log')
    except BaseException as error:
        if process is not None:
            terminate(process)
        record.update(error=repr(error), completed=False)
        raise
    else:
        record['completed'] = True
    finally:
        record['process_wall_s'] = time.monotonic() - phase_start
        phases.append(record)
        with (root / 'phases.jsonl').open('a') as out:
            out.write(json.dumps(record) + '\n')

def cleanup_private_query_heaps(folder, name):
    # The query driver has exited; these are only its disposable private
    # session files, never the prepared input bundle. Validate both first.
    heaps = [folder / 'query.heap', folder / 'query.heap.pq.bin']
    if any(p.is_symlink() or not p.is_file() for p in heaps):
        raise ValueError(f'{name}: expected regular private query heaps for cleanup')
    for heap in heaps:
        size = heap.stat().st_size
        heap.unlink()
        with (root / 'heap-cleanup.jsonl').open('a') as cleanup:
            cleanup.write(json.dumps(dict(phase=name, path=str(heap),
                                          bytes=size, deleted=True)) + '\n')

try:
    profile_check = ('source benchmark/datasets/_load.sh; '
                     '[[ "$DATA_TYPE" = uint8 && "$DIM" = 128 && "$METRIC" = l2 && '
                     f'"$NPTS_FULL" = {full} && "$NPTS_BASE" = {base} && "$UPDATE_POINTS" = {updates} ]]')
    run('profile', ['bash', '-euc', profile_check], dict(env, REPO_DIR=str(repo)), global_deadline)
    gt = ['python3', 'benchmark/scripts/ensure_gt.py', '--base-file', env['DATA_BIN'],
          '--query-file', env['QUERY_BIN'], '--full-count', str(full), '--base-count', str(base),
          '--insert-count', str(updates), '--delete-count', str(updates), '--mixed-rounds', '10',
          '--mixed-insert-cap', str(updates // 10), '--mixed-delete-cap', str(updates // 10),
          '--k-out', '10', '--expected-queries', '10000', '--num-threads', '20', '--generate']
    if env.get('GT_BIN_KCACHED'):
        gt += ['--cached-gt', env['GT_BIN_KCACHED']]
    run('groundtruth', gt, env, global_deadline, root / 'selected_gt.txt')
    selected = (root / 'selected_gt.txt').read_text().strip()
    if '\n' in selected or not selected or not Path(selected).is_file():
        raise ValueError('GT preflight did not return one existing truth path')
    env['GT_BIN_KCACHED'] = selected
    with (root / 'results.csv').open('x', newline='') as table:
        writer = None
        for nlist in nlists:
            deadline = min(global_deadline, time.monotonic() + limits['nlist'])
            child = dict(env, IVF_NLIST=str(nlist), EVAL_TEMPFILES=str(root / f'nlist-{nlist}'))
            print(f'nlist={nlist}: prepare/restore full base', flush=True)
            run(f'nl{nlist}-prepare', ['bash', 'benchmark/IVF/sift/query_prepare.sh'], child, deadline)
            for point in (p for p in points if p['nlist'] == nlist):
                child.update(IVF_NPROBE=str(point['nprobe']), IVF_ROUTER_BREADTH=str(point['router_breadth']),
                             IVF_RUN_TAG=point['tag'])
                name = f'nl{nlist}-{point["tag"]}'
                print(name, flush=True)
                run(name, ['bash', 'benchmark/IVF/sift/query.sh'], child,
                    min(deadline, time.monotonic() + limits['query']))
                folder = Path(child['EVAL_TEMPFILES']) / dataset / 'IVF/query/runs' / point['tag']
                native_record = json.loads((folder / 'result.json').read_text())
                expected = dict(baseline='IVF', phase='query', workload='query', query_count=10000,
                    base_count=base, nlist=nlist, pq_chunks=64, recall_at=10,
                    nprobe=point['nprobe'], router_breadth=point['router_breadth'],
                    router_mode='pca-refined', rerank_factor=3, router_rerank_factor=3,
                     query_threads=int(env['QUERY_THREADS']), explicit_warmup_query_count=int(env['IVF_WARMUP_QUERIES']))
                if wide:
                    expected.update(index_format='mutable', input_index_format='immutable', live_count=base)
                if any(native_record.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'{name}: native result does not match frozen plan')
                if not (math.isfinite(native_record['recall']) and 0 <= native_record['recall'] <= 100
                        and all(math.isfinite(native_record[k]) and native_record[k] > 0
                                for k in ('query_qps', 'query_time_s'))):
                    raise ValueError(f'{name}: invalid native query metrics')
                protocol = json.loads((folder / 'execution_protocol.json').read_text())
                if not protocol['completed'] or protocol['native_driver_process_count'] != 1:
                    raise ValueError(f'{name}: expected one completed standalone query process')
                row = dict(**point, recall_at_10_percent=native_record['recall'],
                    query_qps=native_record['query_qps'], query_time_s=native_record['query_time_s'],
                    query_process_wall_s=protocol['query_process_wall_s'],
                    harness_workflow_wall_s=protocol['harness_workflow_wall_s'],
                    query_script_wall_s=phases[-1]['process_wall_s'], output_dir=str(folder))
                if writer is None:
                    writer = csv.DictWriter(table, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                table.flush()
                rows.append(row)
                if wide:
                    cleanup_private_query_heaps(folder, name)
    if len(rows) != len(points):
        raise ValueError('grid incomplete')
    save('summary.json', dict(completed=True, production_qualified=False, records=rows,
         phases=phases, selected_configuration=None, wall_s=time.monotonic() - started))
    print(f'Completed: {root / "results.csv"}', flush=True)
except BaseException as error:
    save('failure.json', dict(completed=False, error=repr(error), completed_points=len(rows),
         wall_s=time.monotonic() - started, partial_evidence_retained=True))
    raise
PY
)
```
