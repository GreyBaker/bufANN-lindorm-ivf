# Training comparisons and repeated query sweeps

These Linux/WSL tools collect reproducible evidence on **caller-supplied** inputs.
They do not declare production acceptance. No production input is currently
configured on this machine: SIFT1M/SIFT10M profiles resolve to missing placeholder
roots and WSL has no dataset/PQ overrides. No dataset download, resampling, profile
edit or production benchmark was performed for this segment.

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

The unchanged deterministic base-only reservoir, random initialization, ordered
means and empty recovery are shared by both modes. HNSW uses the current rebuilt
index each pass and `insert_breadth` for training assignments. Each mode runs in
a **fresh process**, so its lifetime peak RSS is independent of the other mode.
The output preserves `plan.json`, exact/HNSW logs, each mode's float centroid
matrix and uint32 sampled-ID matrix (both ordinary two-uint32-header `.bin` files).
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
recorded. Graph distance computations are unavailable, not fabricated.

Exact training requires `sample_count * (iterations + 1) * nlist` distances.
Oversized exact/reference work is refused before launching either mode. These
limits count vector distances/searches, **not** individual coordinate operations
or elapsed CPU time; dimensions still affect cost. A smaller `nlist`/sample is an
explicit bounded comparison, **not** evidence about the 50K production setting.
Native memory-estimate guards reserve evaluation scratch conservatively; this
does not enforce a hard process RSS limit.

`--max-sse-ratio` optionally gates HNSW/exact evaluation SSE. Freeze a justified
threshold before measuring; none is implied by the design or this example.
If both objectives are zero, ratio is 1; if only exact is zero, ratio is null
and any supplied ratio gate fails. No bound is claimed to hold for all datasets.
`--max-peak-rss-mb` is an optional **post-run** lifetime-RSS gate (Linux KiB converted
to MiB), including evaluation/export, not a trainer-only interval or allocator cap.
`training_time_s` is the native trainer interval; `evaluation_time_s` is separate;
`process_wall_s` includes the entire diagnostic child, excluding Python hashing.

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
is uncontrolled. Pairing with SPFresh/baselines and mixed-update tests is pending.

## Completion, timeout and fixture verification

Output plans precede work; `summary.json` is written **last**, only after every
required child/identity check. Process failures/timeouts leave partial evidence
without a final summary. Timeouts terminate the entire child process group,
including shell/native descendants. Completed measurements that fail a declared
gate retain a summary with `gates_passed: false` and exit nonzero. No rollback,
atomic/fsync publication or resume guarantee is added. All summaries explicitly
set `production_qualified: false`; external reviewer acceptance is still required.

`ivf_training_diagnostics` CTest covers actual exact/HNSW comparisons for all
dtypes, identical base-only sampling (including a nonfinite reserved float tail),
exact SSE/counters/artifacts, CPU pinning, work/memory refusal, collisions,
timeout and post-run RSS failure. `ivf_query_harness` additionally covers repeated
native sweeps, warmup exclusion, canonical extraction, preserved output/results,
native setting validation, collisions, changed GT, failed trials and RSS gates.
These are small fixtures, not meaningful QPS/latency/RSS measurements.

Latest verification: full Release **12/12** with optional trained helpers (11
without); Linux routing-enabled isolated Release/ASan+UBSan **6/6** each;
routing-disabled isolation **3/3**. Full instrumented diagnostic and expanded
harness suites each pass **1/1** with leak detection. External system/helper
binaries are not instrumented; no complete full sanitizer-roster claim is made.
Production S03/S04 quality/time/RSS/recall/performance/admission remain pending.
