# Training comparisons and repeated query sweeps

These Linux/WSL tools collect reproducible evidence on **caller-supplied** inputs.
They do not declare production acceptance or launch a paired SPFresh comparison.
Configure your dataset/PQ/GT paths and installed dependencies as described in
[README.md](README.md). The tools do not download datasets or infer a production
profile. Keep reserved final queries out of calibration, tuning and diagnostic
selection; use a declared tuning set or synthetic fixtures instead.

## Bounded exact-vs-HNSW training comparison

`benchmark/IVF/build.sh` also builds `tests/ivf_training_diagnostics`. The tool is
available in Linux routing-enabled isolated builds without BufANN/MKL/libaio;
it is separate from native index building and canonical ANN query metrics.

Example using an existing configured profile (values are **not** production tuning):

```bash
source benchmark/datasets/sift1m.sh
python3 benchmark/IVF/qualify.py train \
  --data "$DATA_BIN" --dtype "$DATA_TYPE" --dim "$DIM" \
  --full-count "$NPTS_FULL" --base-count "$NPTS_BASE" \
  --trainer "$PWD/build/ivf-benchmark/tests/ivf_training_diagnostics" \
  --output "$PWD/eval_training/sift1m-comparison-01" \
  --nlist 256 --sample-rate 0.01 --iterations 3 --seed 0 --threads 4 \
  --memory-budget-gb 1 --exact-work-limit 20000000 \
  --assignment-work-limit 100000 --evaluation-rows 1000 \
  --evaluation-work-limit 300000 --timeout 1800
```

The deterministic base-only reservoir, selected initialization, ordered means
and empty recovery are shared by both modes. Default `hnsw` is an alias for
`ann-rebuild`, which builds a full-coordinate index each pass. Training breadth
uses `--training-breadth`, falling back to legacy `--insert-breadth` when omitted;
it is independent of final index routing settings. Each mode runs in
a **fresh process**, so its lifetime peak RSS is independent of the other mode.
The output preserves `plan.json`, exact/HNSW logs, each mode's float centroid
matrix, sampled IDs, assignments and occupancy (ordinary two-uint32-header `.bin`
files; centroid values are float and the other matrices are uint32).
Cryptographic sample-ID hashes must match before a comparison summary is written.
Source and diagnostic-executable identities are checked before/after each mode.

Quality is compared using **exact double nearest-centroid SSE** on identical
evenly spaced training-reservoir slots. Evaluation is bounded by
`min(evaluation_rows, sample_count) * nlist` vector-distance computations.
This is a training-sample diagnostic, **not** holdout/base-list occupancy or
production assignment quality. The HNSW trainer's approximate assigned SSE is
reported separately, never compared as if it were an exact reference objective.
Training objective histories, sample occupancy extrema/empties/recovery counts,
assignment-search/index-build counts and conservative estimated workspace are
recorded. Assignment graph-distance computations are available for reused patched
graphs, but unavailable for the vanilla rebuilt control; known/unknown flags must
be respected. Construction/reinsertion work is not included in that counter.

### Trainer ablations, including LV/LV+

| Mode | Assignment policy |
|---|---|
| `exact` | Scalar double-L2 enumeration over all centroids |
| `hnsw` / `ann-rebuild` | Rebuilt full-coordinate HNSW, breadth-sized candidate rescoring |
| `ann-reuse` / `ann-reuse-seeded` | Reused full-coordinate graph, legacy breadth-sized rescoring, optional previous-assignment seed |
| `ann-reuse-count` / `ann-reuse-seeded-count` | Reused full-coordinate graph returning nearest-one under its float metric; assigned double objective counted separately |
| `lv` / `lv-plus` | Paper-aligned reused/refined graph with 95%-variance PCA assignment and bounded full-coordinate rerank; LV+ adds only previous-assignment seeding |

For LV/LV+, require `--init random_partition`. Both options share one initial
bottom refinement/isolated-node reinsertion, full-coordinate old-neighbor refresh,
and PCA refit/reprojection before every joined assignment phase. The default
`--training-rerank-factor 3` returns at most three candidates for nearest-one
assignment, independently of search breadth; `--training-refine-breadth 128`
controls initial training refinement. These settings are applied only to LV/LV+
in a multi-mode comparison and are recorded in its plan/native records.

To extend the example above, choose a **new** output path and add:

```bash
--modes exact,ann-rebuild,ann-reuse,ann-reuse-seeded,ann-reuse-count,ann-reuse-seeded-count,lv,lv-plus \
--init random_partition --training-breadth 64 \
--training-rerank-factor 3 --training-refine-breadth 128
```

This block is additional arguments, not a standalone shell command. At larger
scale omit `exact` if its declared work budget is infeasible. Bounded exact
evaluation of ANN centroids still runs, but is **not an exact-trained reference**;
the summary leaves corresponding ratios null rather than inventing one.
Compare the candidate-count correction on full search separately from optimized
search, and keep every setting except seeding identical between LV and LV+.
Use separate output paths/processes for repeated training measurements.

All graph construction/refresh/refinement and PCA fitting/projection costs are
included in native training time. Per-pass setup fields are subsets of graph
time; assignment worker fields sum worker seconds, not wall time. Returned
candidates, full-rerank distances and objective distances are separate counters.
Inspect occupancy, exact-evaluated SSE/agreement and unrepaired connectivity
alongside timings. Fixed-pass termination, hnswlib substitution and unspecified
moving-PCA/refinement schedules remain disclosed limits to paper reproduction;
neither the mode name nor a small fixture establishes the paper's speedup.

Exact training requires `sample_count * (iterations + 1) * nlist` distances.
Oversized exact/reference work is refused before launching either mode. These
limits count vector distances/searches, **not** individual coordinate operations
or elapsed CPU time; dimensions still affect cost. A smaller `nlist`/sample is an
explicit bounded comparison, **not** evidence about the 50K production setting.
`--routing-work-limit` separately bounds each refinement/PCA setup, including
coordinate/solver work; declare a larger allowance for larger/high-dimensional
studies rather than bypassing failures. Native memory-estimate guards reserve
evaluation scratch conservatively; this does not enforce a hard process RSS limit.

`--max-sse-ratio` optionally gates every selected mode's evaluation SSE relative
to the selected `exact` trainer, which must be present. Freeze a justified
threshold before measuring; none is implied by the design or this example.
If both objectives are zero, ratio is 1; if only exact is zero, ratio is null
and any supplied ratio gate fails. No bound is claimed to hold for all datasets.
`--max-peak-rss-mb` is an optional **post-run** lifetime-RSS gate (Linux KiB converted
to MiB), including evaluation/export, not a trainer-only interval or allocator cap.
`training_time_s` is the native trainer interval; `evaluation_time_s` is separate;
`process_wall_s` includes the entire diagnostic child, excluding Python hashing.

## Fixed-router diagnostics

`ivf_router_diagnostics` measures nearest-centroid recall against bounded exact
centroid enumeration on an existing router family; it does not build or refit
centroids/PCA. `router_sweep.py` compares `exact`, `hnswlib`, `patched-full`,
`refined`, `pca`, and `pca-refined`, with supplied breadth/probe grids and repeats.
Prepare a bundle with `IVF_OPTIMIZED_MODES=1` to include all required artifacts.

Example from the repository root (ensure each probe count fits the index):

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

Exact evaluation needs `query_rows * nlist` distances within the allowance.
Records include centroid recall, query/descriptor identities, projection/graph/
rerank time, graph-distance availability, projected dimension/variance,
connectivity and lifetime RSS. Timed worker search includes centroid-match
scoring; exact GT, load and identity checks are outside it. One mode-query warmup
and prior validation/evaluation warm caches; no cold-cache claim is made.

The summary averages repeats before choosing the fastest observed breadth per
mode that reaches each declared **minimum** centroid-recall target. This is not
identical-recall interpolation, exhaustive tuning or end-to-end ANN recall/QPS.
Bottom reachability and zero in-degree do not prove strong connectivity.
Save raw records and failures and report settings that miss a target.

## Repeated query sweeps on a prepared bundle

First prepare an immutable native bundle with the [query harness](README.md).
The sweep invokes the unchanged single-run path for every setting and repeat:

```bash
DATASET=sift1m IVF_BUILD_DIR="$PWD/build/ivf-benchmark" \
  bash benchmark/IVF/sift/query_sweep.sh \
  --tag repeated-01 --nprobes 8,16,32 --rerank-factors 2,4,8 --threads 1,4 \
  --repeats 3 --warmups 1 --memory-budget-gb 4 --buffer-frames 64 \
  --timeout 1800
```

`RECALL_AT` and profile/work-directory resolution are shared with `query.sh`.
Lists/centroids/codebook/GT and persisted query breadth are not changed. Sweep
memory budget, pool frames, nprobe, rerank factor and query threads are explicit
and checked against the actual native record; unsupported settings fail rather
than being silently retuned. Maximum is 4096 total trials.

Evidence lives under `query/sweeps/<tag>/`: explicit plan, orchestration logs,
per-trial completion records and a final `summary.json`. Native canonical records,
run logs and SPFresh binary neighbors remain under uniquely named `query/runs/`
directories. Every trial has its own process/output files. Existing sweep/run
destinations, including failures, are refused, never deleted or resumed. Query,
GT, manifest, prepared metadata, driver and relevant script/profile identities
must remain unchanged across the sweep. Native open checks bundle components on
every trial. Identities describe local evidence, not authentication or exhaustive
runtime-library/ABI provenance; record external machine/storage/compiler details
separately for any production report.

Warmups are **separate discarded processes**: they may warm OS caches, not the
next process's buffer pool. Each trial still opens/hashes the entire bundle.
No cold-cache/device-I/O claim, cache dropping or adaptive admission improvement
is made. Configurations run in supplied order, warmups then repeats; ordering can
bias results. CPU frequency, other processes and storage contention are not
controlled. To pin all descendants, add `--cpus` with comma-separated CPU IDs
from the current allowed affinity (requires installed `taskset`); the selected
and available sets are recorded. Pinning is optional and does not imply isolated
CPUs. `query_threads`/`active_query_threads` remain distinct (small query sets can
use fewer workers). Heap pin/copy admission remains serialized.

Measured repeat aggregates are **min/median/max across runs** for recall, QPS,
query/open times, average/p50/p95/p99 query latency, lifetime RSS, backend heap
pages read, workspace estimate and active workers. Medians of per-run percentile
latencies are **not pooled-query percentiles**; per-trial native records remain
available. `--min-recall` is an optional percent-Recall@K gate across measured
trials (default 0, no meaningful quality target). Optional `--max-peak-rss-mb`
gates the maximum measured process RSS. These are declared gates, not matched-
recall baseline selection or automatic production acceptance.

Native timing/read scopes remain unchanged; `process_wall_s` additionally records
the whole query script including hashes/open/GT/scoring/export, not just queries.
Orchestrator rehashes and result hashes incur additional untimed I/O. Cache policy
is uncontrolled. Pairing with SPFresh/baselines requires a separately declared
comparable protocol; these sweeps do not perform that comparison. Mixed workloads
use the separate prepare/run pairs described in
[README.md](README.md#concurrent-mixed-rounds).

## Completion and timeout

Output plans precede work; `summary.json` is written **last**, only after every
required child/identity check. Process failures/timeouts leave partial evidence
without a final summary. Timeouts terminate the entire child process group,
including shell/native descendants. Completed measurements that fail a declared
gate retain a summary with `gates_passed: false` and exit nonzero. No rollback,
atomic/fsync publication or resume guarantee is added. All summaries explicitly
set `production_qualified: false`; external reviewer acceptance is still required.

## Functional verification

Dependency-isolated checks do not require the native heap's MKL/libaio/BufANN
dependencies. From the repository root:

```bash
cmake -S tests/ivf -B build/ivf-contracts -DCMAKE_BUILD_TYPE=Release \
  -DIVF_BUILD_ROUTING=ON
cmake --build build/ivf-contracts -j4
ctest --test-dir build/ivf-contracts --output-on-failure
```

Routing-enabled configuration fetches the checksum-pinned hnswlib archive into
the build tree. Isolated `ivf_build`/`ivf_driver` still validate only; native heap
build/query/calibration CLI checks require a full Linux build. Enable its test
targets by adding `-DIVF_BUILD_CONTRACT_TESTS=ON` to the build command in
[README.md](README.md#build-with-installed-system-dependencies), then build/run
the focused trainer/calibration/CLI tests:

```bash
cmake --build "$IVF_BUILD_DIR" --target ivf_ann_training_tests ivf_build ivf_driver \
  ivf_cost_model ivf_training_diagnostics -j4
ctest --test-dir "$IVF_BUILD_DIR" --output-on-failure \
  -R '^(ivf_ann_training|ivf_training_diagnostics|ivf_ann_training_cli|ivf_cost_model_cli|ivf_cli_execution)$'
```

Ensure the installed MKL/Intel OpenMP libraries are discoverable at runtime; use
your installation's loader configuration or `LD_LIBRARY_PATH`. Declare
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` when retaining performance evidence.

`ivf_training_diagnostics` CTest covers actual trainer comparisons for all
dtypes, identical base-only sampling (including a nonfinite reserved float tail),
exact SSE/counters/artifacts, LV/LV+ candidate/PCA settings, CPU pinning, work/memory
refusal, collisions, timeout and post-run RSS failure. `ivf_query_harness` additionally covers repeated
native sweeps, warmup exclusion, canonical extraction, preserved output/results,
native setting validation, collisions, changed GT, failed trials and RSS gates.
`ivf_ann_training` checks joined-worker determinism, returned-candidate oracles,
moving PCA dimensions, tiny/degenerate/empty recovery and bounded work failures.
`ivf_ann_training_cli` checks typed build/reopen and immutable/mutable producer
identity. `ivf_cost_model_cli` checks base-only calibration, independent GT/radius/
count oracles, declared sample minima 32/20/1, historical-report compatibility,
frozen-report/setting provenance and destination/source preservation.

Optional `ivf_trained_helper_integration` requires both existing standalone PQ
helpers, supplied at configuration through `IVF_PQ_PIVOT_HELPER` and
`IVF_PQ_CODES_HELPER`. Other full-build component/harness tests require their
corresponding executables to be built before running CTest. Consult
`ctest --test-dir "$IVF_BUILD_DIR" -N` for the configured roster.

These are small fixtures, not meaningful QPS/latency/RSS measurements. Retain
your own configure/build/test logs and compiler/runtime/source identities.
Scoped sanitizer results do not cover uninstrumented external libraries/helpers
or imply full-system race qualification. Production quality/performance,
sampling-sensitivity and prediction-validation studies remain separate gates.
