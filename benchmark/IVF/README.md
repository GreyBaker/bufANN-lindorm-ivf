# Native IVF build, calibration and workload harness

Linux/WSL query-only scripts are implemented and tested with synthetic and
standalone-helper-trained fixtures for `float`, `int8`, and `uint8`. This is not
production SIFT training, recall/QPS or memory qualification. No vector-level
graph, Vamana conversion, distributed service or SPFresh source change is required.
Sequential insert/delete prepare/run pairs and native mutable checkpoint/reopen
are implemented, as are concurrent weighted/fixed 8+8 mixed prepare/run pairs.
Production qualification and paired SPFresh comparison are separate from these
functional features. Configure inputs and dependencies for your own machine;
examples below are not qualified production settings.

`IVF_OPTIMIZED_MODES=1` builds the patched-hnswlib/PCA router family;
`IVF_ASSIGNMENT_ROUTER_MODE=pca-refined` sets the frozen build/insert default.
`IVF_ROUTER_MODE` changes queries only without invalidating that build cache.

## Build with installed system dependencies

Use an already-configured full routing-enabled build via `IVF_BUILD_DIR`, or
configure a separate tree. Example for a oneAPI installation (adjust paths):

```bash
IVF_BUILD_DIR="$PWD/build/ivf-benchmark" bash benchmark/IVF/build.sh \
  -DMKL_ROOT=/opt/intel/oneapi/mkl/2026.1 \
  -DMKL_LIBRARY_DIR=/opt/intel/oneapi/mkl/2026.1/lib \
  -DINTEL_OMP_LIBRARY_DIR=/opt/intel/oneapi/compiler/2026.1/lib
```

The script enables `IVF_BUILD_ROUTING=ON` and builds `ivf_build`/`ivf_driver`,
`ivf_cost_model` and training/router diagnostics without
changing other baseline build trees. Administrator-installed MKL (ILP64, Intel
threaded), iomp5 and libaio linkage is retained. Override paths for your install;
default build directory is `build/ivf-benchmark`, jobs default to 4. Prepare/run
consume existing binaries, **never** silently configure/install dependencies.
The shared legacy `benchmark/scripts/build.sh` dependency-prefix path remains
separate and unqualified; it is not the explicit build above.

## Input/PQ/GT contracts

- Load the unchanged `benchmark/datasets/<DATASET>.sh` profile. The `.bin` data
  and query files must exist for query preparation. Mutation preparation uses
  the shared `ensure_dataset` staging helper; no new downloader/converter is added.
- `DATA_BIN` must frame exactly `NPTS_FULL` rows and `DIM` coordinates of
  `DATA_TYPE`. Only `[0, NPTS_BASE)` becomes the initial index. Profile splits
  and external source-row IDs are not inferred, changed or remapped.
- Supply `IVF_PQ_PREFIX` with current five-offset pivots and **full-source** codes
  (`<prefix>_pq_pivots.bin`, `<prefix>_pq_compressed.bin`). PQ is global,
  non-residual, 8-bit and reused, not retrained by prepare. Canonical graph PQ
  files often encode only the base: these must not be passed as full-source codes.
  If needed, copy the existing pivots to a **new** PQ prefix and run the unchanged
  `build_pq_codes_standalone TYPE DATA_BIN NEW_PREFIX ACTUAL_PQ_CHUNKS 0` helper.
  Do not overwrite the canonical prefix or train a different comparison codebook.
- By default derive base GT from `GT_BIN_KCACHED` using the unchanged shared
  `resolve_incremental_gt ... --base_trunc NPTS_BASE` helper. Alternatively
  `IVF_GT_FILE` points at caller-supplied matching base GT; it is linked read-only.
  GT is IDs/distances plus an optional tags plane; tags must equal the ID plane.
  Nonidentity tag-space mappings are explicitly refused. No GT is rewritten to
  fit IVF. Inadequate/deep-cache or nonfinite GT fails rather than faking recall.

## Prepare and run

```bash
export DATASET=sift1m
export IVF_BUILD_DIR="$PWD/build/ivf-benchmark"
export IVF_PQ_PREFIX=/absolute/path/to/existing-full-source-pq/prefix
export IVF_NLIST=1024  # explicit; example only, not qualified production tuning
export IVF_MEMORY_BUDGET_GB=4
export IVF_BUILD_THREADS=4
bash benchmark/IVF/sift/query_prepare.sh

IVF_NPROBE=16 IVF_RERANK_FACTOR=4 QUERY_THREADS=4 \
  bash benchmark/IVF/sift/query.sh
# query_run.sh is an equivalent alias.
```

Prepare materializes `eval_tempfiles/<dataset>/IVF/query/`: dataset/query/GT links
are read-only inputs; `index/<dataset><suffix>` are **real bundle files**, not
writable links into the cache. `prepared.json` is written only after completion.
Existing work trees are refused, not recursively deleted. To re-prepare, preserve
the old tree and select a fresh `EVAL_TEMPFILES` root. Failed partial trees remain
for inspection; no rollback, atomic publication/fsync, WAL or repair is claimed.

Run consumes the prepared tree in place and writes `run.log`, `native_result.json`,
`result.json` and interleaved SPFresh `neighbors.bin`. The unchanged shared
`write_ann_bench_metrics_json` extracts canonical JSON records and is cross-checked
against native output. Failures do not publish an empty success `result.json`.
Existing run files are refused. Repeat/sweep without discarding earlier evidence:

```bash
IVF_RUN_TAG=probe32 IVF_NPROBE=32 QUERY_THREADS=4 bash benchmark/IVF/sift/query.sh
```

Tagged output is under `query/runs/<tag>/`; safe tags contain letters, numbers,
hyphens or underscores. For automated repeats/setting sweeps, optional CPU pinning
and explicitly separate-process warmups, see [QUALIFICATION.md](QUALIFICATION.md).
These tools do not automatically qualify a configuration. Run checks
profile, query, GT and native-manifest identities; native open validates all
component contents. Working index directories/components may not symlink to cache.

### Settings

Build settings use `IVF_` plus the uppercase CLI field: `IVF_NLIST` (required unless
`IVF_COST_MODEL_REPORT` supplies a frozen choice to the Python harness; the shell
prepare wrappers still require the matching explicit `IVF_NLIST`),
`IVF_SAMPLE_RATE` (1), `IVF_MAX_REPS` (15), `IVF_SEED` (0), `IVF_BUILD_THREADS` (1),
`IVF_MEMORY_BUDGET_GB` (1), `IVF_PAGE_SIZE` (4096), `IVF_BUFFER_FRAMES` (64),
`IVF_ROUTER_DEGREE` (16), `IVF_CONSTRUCTION_BREADTH` (200), `IVF_QUERY_BREADTH` (128),
`IVF_INSERT_BREADTH` (64) and `IVF_ROUTER_SEED` (100). CLI validates all settings.

Run uses `IVF_NPROBE` (1), `IVF_RERANK_FACTOR` (2), `QUERY_THREADS` (1), profile
`RECALL_AT` (normally 10), `IVF_BUFFER_FRAMES` (64), and
`IVF_QUERY_MEMORY_BUDGET_GB` (falls back to `IVF_MEMORY_BUDGET_GB`, then 1).
With no query override, persisted query breadth is used. `IVF_ROUTER_BREADTH`
overrides it for queries without rebuilding. No mutable scheduling/8+8 split
is implied by query thread settings.

### Optimized centroid routing

Set `IVF_OPTIMIZED_MODES=1` at build time to save the patched, refined and PCA
artifacts alongside the ordinary HNSW router. Available query modes are `hnswlib`,
`exact`, `patched-full`, `refined`, `pca`, and `pca-refined`. The latter four require
that artifact family. `IVF_ASSIGNMENT_ROUTER_MODE` selects the frozen base/insert
default; `IVF_ROUTER_MODE` overrides queries only.

`IVF_REFINE_EF` (128), `IVF_PCA_VARIANCE` (.95) and
`IVF_BUILD_ROUTER_RERANK_FACTOR` (3) configure final-router construction/search.
`IVF_ROUTING_WORK_LIMIT` (1000000000) bounds refinement/PCA setup work.
For query-only overrides use `IVF_ROUTER_RERANK_FACTOR` and `IVF_ROUTER_BREADTH`.
These centroid rerank factors are distinct from `IVF_RERANK_FACTOR`, which controls
full-vector candidates after PQ scanning.

Construction and pruning use full coordinates. Refinement replaces bottom-layer
out-edges from a frozen search/prune pass, then reinserts zero-in-degree nodes
once; it does not guarantee global connectivity. PCA traversal uses compressed
centroids followed by full-coordinate reranking. Final router/PCA artifacts are
frozen across insert/delete/mixed workloads; they are not refit by query overrides.
See [QUALIFICATION.md](QUALIFICATION.md#fixed-router-diagnostics) for centroid-only
measurements, or [the patch description](../../dependencies/hnswlib-patches/README.md)
for dependency mechanics.

### Sequential insert/delete

```bash
UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/insert.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete_prepare.sh
UPDATE_PERCENT=10 bash benchmark/IVF/sift/delete.sh
```

Use the same build/PQ/profile environment as above. Shared split/count helpers
preserve SPFresh's initial base, insert-tail and delete-base-tail IDs. Optional
`INSERT_POINTS`/`DELETE_POINTS` override only counts. Default GT is filtered for
the final live prefix with the unchanged helper; `IVF_GT_FILE` must be matching
**post-update** GT. Insufficient survivors fail, never pad truth.

Each pair owns a distinct `.../IVF/{insert,delete}/` tree. Prepare caches/restores
only a real immutable base family. Run updates a new private heap, saves a separate
mutable checkpoint, then reopens a new private copy for post-update scoring.
`mutation_result.json` (insert/delete loops) and `result.json` (final query) are
separate canonical records cross-checked with shared extraction. `IVF_RUN_TAG`
preserves repeated runs starting from the same base; no old work is reset/deleted.
One sequential mutation worker is used; `QUERY_THREADS` controls only the later
query pass, not queries during updates. Native `search_after_update` is also
explicitly sequential, not mixed scheduling. Use `ivf_driver --help` for direct
native flags.

Posting lists use append/+16-slot growth and swap deletion with a locator; deleted
IDs cannot be reused. Checkpoints preserve live IDs, tombstones and aborted heap
records, require joined/quiescent workers, and reopen on new private files. They
do not reclaim physical slots or provide coordinated WAL/crash recovery,
atomic/fsync publication or transactional durability.

### Concurrent mixed rounds

```bash
UPDATE_PERCENT=10 UPDATE_BATCH=10000 bash benchmark/IVF/sift/update_prepare.sh
UPDATE_PERCENT=10 UPDATE_BATCH=10000 IVF_SCHEDULER=weighted NTHREADS=16 \
  QUERY_RATIO=80 QUERY_THREADS=4 bash benchmark/IVF/sift/update.sh
UPDATE_PERCENT=10 UPDATE_BATCH=10000 IVF_SCHEDULER=fixed_8_8 NTHREADS=16 \
  IVF_RUN_TAG=fixed QUERY_THREADS=4 bash benchmark/IVF/sift/update.sh
```

Caps/counts/round GT use the unchanged SPFresh/shared layout helpers, including
equal insert/delete caps for paired scripts. Weighted mode reuses the shared
remaining-operation scheduler; fixed mode has eight dedicated mutation and eight
query workers, not graph-maintenance threads. `QUERY_THREADS` controls only
post-round scoring. `QUERY_POINTS` uses the shared total/per-round policy; otherwise
`QUERY_RATIO` defaults to 80. Every round joins, validates/checkpoints and scores
against its matching GT; the next process reopens onto a new private heap.
Concurrent query timing/overlap is separate from post-barrier recall. Root
`result.json` is the final round record; `summary.json` retains all round records
and always disclaims production qualification. Earlier work is never reset.

For direct native execution use `ivf_driver --workload mixed`; `--help` lists
required inputs, scheduling and checkpoint flags. Concurrent query timings do
not carry synchronized recall against a moving live set; recall is measured
separately after each round's mutation/query workers have joined.

### Cache identity and costs

`eval_cached/<dataset>/IVF/<sha256-key>/` binds exact build settings/profile,
streamed full vector/pivot/code identities, builder and BufANN library bytes,
plus optional `IVF_CACHE_TAG`. Cached components/manifest are hashed before/after
copying; checksum/identity failures refuse reuse rather than overwriting cache.
`EVAL_CACHE_NO_STORE` (any nonempty value) disables new cache stores, not restores.
Cache hits report restore events, **not** new build-time measurements.

Identity is not authentication or external PQ-training provenance. Files must be
trusted/immutable during operations; native artifacts are machine-local. External
runtime-library/ABI/toolchain identity is not exhaustively tracked: advance
`IVF_CACHE_TAG` (or use a new cache root) when that environment changes.

Full-source hashes, heap/component hashing and real cache copies add startup I/O
and warm caches. `build_time_s` measures the native API, not Python hashing/cache
work; `query_time_s` excludes preparation, input/GT checks, open, scoring and
export. Native open itself hashes the entire heap. Cache policy is uncontrolled,
not cold-device throughput. Pin/copy admission remains serialized. Lifetime peak
RSS includes startup/open/scoring/export, not just timed queries. Heap-read
counters are backend buffer-pool reads, not proof of physical device I/O. Memory
budgets check conservative workspace estimates rather than enforcing hard RSS caps.

## Calibration and verification

### Calibration and selection provenance

`benchmark/IVF/cost_model.py` orchestrates the native paper-style `T1+T2` tool
using only base rows and disjoint synthetic queries. It freezes source/PQ/tool
identities, actual samples/GT/routers, eligible/excluded candidates and predictions.
`IVF_COST_MODEL_REPORT=/path/report.json` supplies its chosen `nlist` to preparation;
settings/source mismatches and conflicting manual choices fail. Preparation/cache
metadata and adjacent run `nlist_selection.json` preserve the report path/hash.
Model selection is orchestration metadata, not a new native index format;
manual controls remain available. Trainer-specific formats are described below.

This is functional implementation, **not** production calibration or the required
SIFT10M prediction-validation study. Fetch/rerank are excluded; no end-to-end-QPS
optimum is claimed. The estimator minimizes mean measured centroid routing (`T1`)
plus radius occupancy scaled by `N/m` times a profiled scalar scan/selection
coefficient (`T2`). Sample GT is exact double squared L2; repeated GT-cluster IDs
and inclusive centroid-radius ties are retained. Profiling uses real base PQ
codes; ADC-table construction, fetch and exact rerank are outside the model.

Example calibration from the repository root, using a **new** output directory:

```bash
python3 benchmark/IVF/cost_model.py \
  --tool "$IVF_BUILD_DIR/tests/ivf_cost_model" \
  --output /new/path/calibration-r10 \
  --data "$DATA_BIN" --pq-prefix "$IVF_PQ_PREFIX" \
  --data-type "$DATA_TYPE" --dim "$DIM" --base-count "$NPTS_BASE" \
  --sample-count 32768 --synthetic-queries 100 \
  --nlist-grid 256,512,1024 --topk 10 --target-recall .90 \
  --trainer-mode lv-plus --training-breadth 64 \
  --build-threads 4 --memory-budget-gb 4 --min-sample-per-centroid 32
```

Choose sample/query counts that fit your base prefix. Eligibility requires
`sample_count >= nlist * min_sample_per_centroid`. The default **32 is a project
policy, not a Lindorm paper requirement or its SIMD batch size**. Positive minima
down to 1 allow explicitly declared sampling-sensitivity work while enforcing
`nlist <= sample_count`; they do not prove sparse-sample adequacy. Version-2
reports record the minimum, ratios, exclusions and below-historical-32 warnings.
Historical version-1 reports retain their implicit minimum 32. Existing reports
must not be edited to admit excluded candidates or refit predictions.

After freezing a report, match its source/PQ and all trainer/router/resource
settings through the build environment. Model-associated full builds require
`IVF_SAMPLE_RATE=1`. The shell-wrapper workaround is:

```bash
export IVF_COST_MODEL_REPORT=/new/path/calibration-r10/report.json
export IVF_NLIST="$(python3 -c 'import json,os; print(json.load(open(os.environ["IVF_COST_MODEL_REPORT"]))["selected_nlist"])')"
# Match the report's remaining build settings before preparation.
bash benchmark/IVF/sift/query_prepare.sh
```

Freeze separate reports for different recall/top-k targets before validation.
Compare predicted and measured routing/scan costs and report absolute-cost misses
separately from selected-candidate ratios and end-to-end latency. Keep reserved
final queries out of calibration/tuning. PQ4/SIMD scanning remains a deferred
follow-up; current scalar-PQ8 coefficients do not reproduce SIMD small-list penalties.

`ivf_query_harness` covers actual build/run, canonical records/results, cache
restore/config/source identity changes, no-store behavior, no overwrites, failed
runs, corrupted caches, symlink refusal, changed GT and the unchanged incremental
GT helper path. It now also tests insert/delete pairs for all dtypes, canonical
IDs/count overrides, post-checkpoint scoring, underflow, fresh repeats and immutable
source/base/cache preservation. `ivf_trained_helper_integration` is opt-in; it uses unmodified
standalone helper sources to train/encode 4096 rows in 13 dimensions with three
unequal chunks, then checks native encoder/ADC parity and exact script-run results
for every dtype. Instrumenting only the IVF side with ASan/UBSan and leak detection
does **not** instrument the standalone helpers or system libraries.

Run the relevant CTest suites in your configured build; verification commands
and fixture scopes are in [QUALIFICATION.md](QUALIFICATION.md#functional-verification).
Production training, repeated recall/QPS/latency/memory/concurrency budgets,
sampling adequacy and prediction accuracy require their own measurements.
No production-performance result follows from fixtures.

Bounded exact/HNSW training comparisons and repeated query-sweep tooling now pass
fixtures, including declared budgets, preservation and failure gates:
[QUALIFICATION.md](QUALIFICATION.md). These collect evidence, not acceptance.

## Training selection

New native CLI and prepare/harness builds default to
`ann-reuse-seeded`, `random_partition`, training breadth 64. Set
`IVF_TRAINER_MODE=exact|ann-rebuild|ann-reuse|ann-reuse-seeded|ann-reuse-count|ann-reuse-seeded-count|lv|lv-plus`,
`IVF_INIT=sample|random_partition`, `IVF_TRAINING_BREADTH=N`, or the matching
`ivf_build --trainer_mode --init --training_breadth` flags. These settings enter
build/cache identity; query router overrides do not change them. Programmatic
`BuildConfig` retains legacy `ann-rebuild`/`sample` defaults for existing callers.
Training breadth is independent of frozen base/insert routing breadth. Exact
training remains scalar and work-bounded; GEMM is not required. A cost-model report
is optional for a manually selected build, but manual sweeps do not validate
predictive selection.

Legacy ANN modes use full-coordinate assignment with breadth-sized double
rescoring. The new full-coordinate `ann-reuse-count`/`ann-reuse-seeded-count`
controls return nearest-one under the graph's float metric and compute the double
assigned objective separately. They isolate candidate-count changes from PCA.

**LV and LV+ are designed to largely match the paper's training mechanisms.**
Both require `random_partition`, reuse a full-coordinate centroid graph with
old-neighbor reinsertion, apply one initial bottom refinement/isolated-node
reinsertion pass, and traverse 95%-variance PCA centroids during assignment.
LV+ adds only previous-assignment entry seeding, with first/invalid-seed fallback.
PCA is refit/reprojected at each joined assignment boundary, including the final
one. The moving-PCA and initial-only refinement schedules are disclosed
interpretations; hnswlib substitution and a fixed pass cap differ from an exact
reproduction of the authors' implementation/convergence experiment.

`IVF_TRAINING_RERANK_FACTOR` (3) bounds returned full-rerank candidates independently
of traversal breadth; `IVF_TRAINING_REFINE_BREADTH` (128) controls initial training
refinement. Native equivalents are `--training_rerank_factor` and
`--training_refine_breadth`. Nondefault overrides require LV/LV+. These settings
are independent of final query/insert router settings; LV training does not
require `IVF_OPTIMIZED_MODES=1` for the final router. Do not change other settings
when comparing LV to LV+ if the intended ablation is seeding alone.

Legacy trainer builds write `IVFINDEX3`, persisting mode, initialization and
training breadth with an explicit optimized-mode flag. New count-based/LV builds
write `IVFINDEX4`, additionally binding training-search policy and independent
rerank/refinement settings. Existing v1/v2 bundles reopen with their actual
legacy rebuilt/sample/insert-breadth identity; mutable checkpoints preserve their
origin version/settings. New builds expose per-pass graph/assignment/mean timing,
assigned SSE, occupancy summaries, refresh and seeding counts. Graph-distance
counts cover assignment traversal only, not graph construction/reinsertion, and
are unavailable for the vanilla rebuilt control. Assigned SSE is not exact
nearest-centroid SSE for approximate training. New per-pass diagnostics include
PCA/refinement setup, retained dimensions/variance, requested/effective breadth,
returned/reranked counts and connectivity measurements. Worker phase times are
summed worker seconds, not wall time; setup subsets must not be added to total
graph time again. Objective-distance work is counted separately from reranking.

For bounded ablations, `qualify.py train` accepts
`--modes exact,ann-rebuild,ann-reuse,ann-reuse-seeded --init random_partition
--training-breadth 64`. All modes use the same sample/init/seed/pass cap. At large
scale omit `exact` to avoid a full exact training run; bounded exact evaluation
still runs on the declared evaluation rows. The summary explicitly reports an
absent exact-trained reference, rather than inventing an SSE ratio. Diagnostic
artifacts include `.sample_ids.bin`, `.centroids.bin`, `.assignments.bin` (one
centroid per sample slot) and `.occupancy.bin`, and exact evaluation reports
tie-aware nearest-centroid assignment agreement on the sampled evaluation rows.
This is a training-sample check, not holdout or final-query scoring.
