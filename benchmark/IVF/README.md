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
export IVF_NLIST=1024 IVF_MEMORY_BUDGET_GB=4 IVF_BUILD_THREADS=4

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
See the ablation settings below for training and router overrides.

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

`IVF_ASSIGNMENT_ROUTER_MODE` freezes the base/insert router (default `hnswlib`);
`IVF_ROUTER_MODE` overrides queries. `exact` enumerates centroids; `patched-full`,
`refined`, `pca` and `pca-refined` require `IVF_OPTIMIZED_MODES=1` at preparation.
Centroids, routers and final PCA stay frozen across updates. Centroid controls
`IVF_ROUTER_BREADTH`/`IVF_ROUTER_RERANK_FACTOR` are distinct from full-vector
`IVF_RERANK_FACTOR`. See native `--help` for more settings.

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
git clone --recurse-submodules https://github.com/GreyBaker/bufANN-lindorm-ivf.git "$HOME/ivf-project"
cd "$HOME/ivf-project"

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
  cmake -S baselines/DiskANN -B baselines/DiskANN/build -DCMAKE_BUILD_TYPE=Release
)
```

This one will be angry, but it will complete.
```bash
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
export DATASET=sift100m; export DATA_ROOT="$HOME/ann-data/$DATASET"; export DATA_BIN="$DATA_ROOT/base.bin" QUERY_BIN="$DATA_ROOT/query.10K.bin" GT_BIN_KCACHED="$DATA_ROOT/full-query.gt100.bin"
source "benchmark/datasets/$DATASET.sh"
printf 'Dataset=%s total=%s base=%s updates=%s\n' \
  "$DATASET" "$NPTS_FULL" "$NPTS_BASE" "$UPDATE_POINTS"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 IVF_PQ_PREFIX="$DATA_ROOT/pq/pq64"

(
  set -euo pipefail
  python3 benchmark/IVF/download_sift.py \
    --rows "$NPTS_FULL" --base "$DATA_BIN" --query "$QUERY_BIN"

  # Full-source top-100 GT for every query.
  "$IVF_BUILD_DIR/tests/bufann_gt_computer" \
    --data_type uint8 --base_file "$DATA_BIN" --query_file "$QUERY_BIN" \
    --gt_file "$GT_BIN_KCACHED" --K 100 --num_threads 8

  # Full-source PQ, including the insertion tail; no DiskANN graph.
  mkdir -p "$DATA_ROOT/pq"
  baselines/DiskANN/build/tests/build_pq_pivot_standalone \
    uint8 "$DATA_BIN" "$IVF_PQ_PREFIX" 64
  baselines/DiskANN/build/tests/build_pq_codes_standalone \
    uint8 "$DATA_BIN" "$IVF_PQ_PREFIX" 64 0
)
```

Download, GT and PQ generation can take substantial time. The downloader checks
headers/ranges, refuses existing outputs and retains partial files; it does not
resume. Reuse completed inputs unchanged across runs; skip their generation when
using an existing verified dataset. The PQ helper's final `0` encodes all rows.

IVF filters GT for each live set and rejects insufficient surviving candidates.
This recipe measures **Recall@10**; Recall@100 needs deeper GT and fresh preparation.
Keep final queries out of tuning: freeze settings using base-only calibration or
a separate tuning set before scoring them.

## 4. Configure and run all four workloads

**Training limit:** SIFT100M uses **10 passes on all 90M base rows**, plus final
assignment: 990M searches, below the native 1B guard. The default 15 passes would
need 1.44B and fail; there is no environment override. SIFT10M fits with 15 passes.

```bash
# Remove any smaller-workload overrides from earlier experiments.
unset INSERT_POINTS DELETE_POINTS QUERY_POINTS UPDATE_QUERY_BATCH QUERY_RATIO INSERT_CAP DELETE_CAP IVF_RUN_TAG IVF_COST_MODEL_REPORT IVF_SCHEDULER_SEED

# Train on every initial-base row.
if [ "$DATASET" = sift100m ]; then
  export IVF_MAX_REPS=10
else
  export IVF_MAX_REPS=15
fi
export IVF_SAMPLE_RATE=1 IVF_TRAINER_MODE=ann-reuse-seeded IVF_INIT=random_partition IVF_TRAINING_BREADTH=64 IVF_SEED=0 IVF_BUILD_THREADS=8 IVF_MEMORY_BUDGET_GB=96

# Explicit starting configuration, not a qualified recall optimum.
export IVF_NLIST=10000 IVF_NPROBE=32 IVF_RERANK_FACTOR=4 QUERY_THREADS=8 RECALL_AT=10

# Full profile update counts; ten mixed rounds in either dataset.
export UPDATE_PERCENT=10 UPDATE_BATCH=$((UPDATE_POINTS / 10)) IVF_SCHEDULER=update_driven NTHREADS=16 EVAL_TEMPFILES="$HOME/ann-eval/run01" EVAL_CACHED="$HOME/ann-eval/cache"

# Each workload gets an independent prepared base; stop on any failure.
(
  set -euo pipefail
  for workload in query insert delete update; do
    bash "benchmark/IVF/sift/${workload}_prepare.sh"
    bash "benchmark/IVF/sift/${workload}.sh"
  done
)
```

Memory budgets check workspace estimates, not hard RAM caps. Keep these settings
fixed across preparation and execution. Cache hits restore an index, not measure
a new build; use a fresh cache root when measuring fresh construction.

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
  --nprobes 32 --rerank-factors 4 --threads 8 \
  --repeats 3 --warmups 1 \
  --memory-budget-gb 96 --buffer-frames 64 --timeout 86400
```

For another complete run, choose a fresh work root and repeat the loop in step 4
without changing the frozen settings:

```bash
unset IVF_RUN_TAG
export EVAL_TEMPFILES="$HOME/ann-eval/run02"
# Repeat only the workload loop from step 4.
```

Choose a new root for each repetition so preparation starts from the same IVF
base. Do not delete failed evidence or change settings in response to final-set
results. Tagged reruns from an existing IVF prepared base are also supported;
the fresh-root recipe makes each repetition explicit.
