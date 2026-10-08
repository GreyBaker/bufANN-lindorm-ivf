# Native IVF query and mutation harness (S04/S05)

Linux/WSL query-only scripts are implemented and tested with synthetic and
standalone-helper-trained fixtures for `float`, `int8`, and `uint8`. This is not
production SIFT training, recall/QPS or memory qualification. No vector-level
graph, Vamana conversion, distributed service or SPFresh source change is required.
Sequential insert/delete prepare/run pairs and native mutable checkpoint/reopen
are implemented (S05); concurrent mixed prepare/run pairs remain pending (S06).

## Build with installed system dependencies

Use an already-configured full routing-enabled build via `IVF_BUILD_DIR`, or
configure a separate tree. For the verified oneAPI installation on this machine:

```bash
IVF_BUILD_DIR="$PWD/build/ivf-benchmark" bash benchmark/IVF/build.sh \
  -DMKL_ROOT=/opt/intel/oneapi/mkl/2026.1 \
  -DMKL_LIBRARY_DIR=/opt/intel/oneapi/mkl/2026.1/lib \
  -DINTEL_OMP_LIBRARY_DIR=/opt/intel/oneapi/compiler/2026.1/lib
```

The script enables `IVF_BUILD_ROUTING=ON` and builds `ivf_build`/`ivf_driver` plus
`ivf_training_diagnostics` without
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
Neither path has run production measurements. Run checks
profile, query, GT and native-manifest identities; native open validates all
component contents. Working index directories/components may not symlink to cache.

### Settings

Build settings use `IVF_` plus the uppercase CLI field: `IVF_NLIST` (required),
`IVF_SAMPLE_RATE` (1), `IVF_MAX_REPS` (15), `IVF_SEED` (0), `IVF_BUILD_THREADS` (1),
`IVF_MEMORY_BUDGET_GB` (1), `IVF_PAGE_SIZE` (4096), `IVF_BUFFER_FRAMES` (64),
`IVF_ROUTER_DEGREE` (16), `IVF_CONSTRUCTION_BREADTH` (200), `IVF_QUERY_BREADTH` (128),
`IVF_INSERT_BREADTH` (64) and `IVF_ROUTER_SEED` (100). CLI validates all settings.

Run uses `IVF_NPROBE` (1), `IVF_RERANK_FACTOR` (2), `QUERY_THREADS` (1), profile
`RECALL_AT` (normally 10), `IVF_BUFFER_FRAMES` (64), and
`IVF_QUERY_MEMORY_BUDGET_GB` (falls back to `IVF_MEMORY_BUDGET_GB`, then 1).
Persisted query breadth is a build setting; varying it needs separately prepared
bundles. No mutable scheduling/8+8 split is implied by query thread settings.

### Sequential insert/delete (S05)

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
explicitly sequential, not S06 mixed scheduling.

CLI examples, file ownership, metrics/budget scopes and verification:
[s05-cli-and-mutation-harness.md](../../docs/s05-cli-and-mutation-harness.md).
Checkpoint format/limits: [MUTABLE_INDEX.md](../../docs/MUTABLE_INDEX.md).

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
not cold-device throughput. Pin/copy admission remains serialized. Detailed
timing/read/RSS scopes: [s04-cli-execution.md](../../docs/s04-cli-execution.md).

## Tested gates and remaining work

`ivf_query_harness` covers actual build/run, canonical records/results, cache
restore/config/source identity changes, no-store behavior, no overwrites, failed
runs, corrupted caches, symlink refusal, changed GT and the unchanged incremental
GT helper path. It now also tests insert/delete pairs for all dtypes, canonical
IDs/count overrides, post-checkpoint scoring, underflow, fresh repeats and immutable
source/base/cache preservation. `ivf_trained_helper_integration` is opt-in; it uses unmodified
standalone helper sources to train/encode 4096 rows in 13 dimensions with three
unequal chunks, then checks native encoder/ADC parity and exact script-run results
for every dtype. Both suites pass with the IVF side instrumented by ASan/UBSan
and leak detection; standalone helpers/system libraries are **not** instrumented.

Reproduction/observed code parity and numerical tolerances:
[s04-query-harness-and-trained-pq.md](../../docs/s04-query-harness-and-trained-pq.md).
S03/S04 remain partial until production training and repeated recall/QPS/latency/
memory/concurrency budgets are qualified. S05 minimum updates/checkpoints/CLI/script
integration is implemented; S06 mixed and S07 paired SPFresh benchmark segments
remain pending. No production-performance result follows from fixtures.

Bounded exact/HNSW training comparisons and repeated query-sweep tooling now pass
fixtures, including declared budgets, preservation and failure gates:
[QUALIFICATION.md](QUALIFICATION.md). These collect evidence, not acceptance.
