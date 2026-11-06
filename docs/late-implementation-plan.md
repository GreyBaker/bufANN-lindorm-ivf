# Late implementation plan

Status: phases 1–3 implemented and validated. Phase 4 remains unstarted.

Beware that this document may require modification. 

Execute these bodies of work in order, validating each before starting the next:

1. Uniform 15-pass training and correctly scoped assignment guards.
2. Persistent mixed execution without inter-round stops or checkpoints.
3. Remove mutable checkpoint/recovery machinery; retain BufANN-like writeback.
4. Replace the IVF hnswlib graph implementations with FAISS HNSW.

Here, FAISS means **FAISS HNSW**, including the vanilla comparison and the
optimized centroid/training paths, not a change to a single-layer NSW algorithm.
Paths in this document are relative to `ivf-only/`.

## Progress and evidence

Mark a task complete only after its implementation and relevant checks pass.
Record commands, results, limitations, and evidence paths in the spaces below.
Do not mark full-scale experiments complete on the strength of synthetic tests.

| Phase | Implementation | Validation | Evidence / revision |
|---|---|---|---|
| 1. Training and guards | Complete | Passed | 3B allowance; fixed 15-pass recipes; evidence below |
| 2. Persistent mixed execution | Complete | Passed | One process/session/pool; version-3 accounting; evidence below |
| 3. Writeback-only mutable sessions | Complete | Passed | Bounded writeback; same-session scoring; 59 passed / 1 skipped; evidence below |
| 4. FAISS HNSW migration | Pending | Pending | |

## End-state contract and boundaries

- Mutable operation completion means online visibility, not durable commit.
- The prepared base is immutable and reusable. Each experiment owns a private
  writable session derived from that base.
- Mixed rounds share one native process, mutable index, heap, buffer pool, and
  worker infrastructure. Round transitions do not drain all workers, serialize
  the index, reopen it, or run standalone recall scoring.
- After the last round, join foreground workers, separately measure writeback,
  then score the stable final live set on the same index. Writeback is not a snapshot.
- Dirty eviction and explicit page writeback remain real storage operations.
  No WAL, per-update persistence barrier, crash recovery, or complete updated-index
  restart guarantee is introduced.
- BufANN-like describes these capabilities, not identical throughput, metadata
  formats, maintenance algorithms, or physical layouts.
- Keep SIFT100M-only Bash orchestration, `dataset/runner/workload` nesting,
  reusable GT/PQ under `ann-data`, and README experiment instructions.
- Preserve existing bases, caches, results, logs, neighbors, and historical
  checkpoints. Incompatible artifacts are rejected, not overwritten or relabeled.
- Keep the 73-pass wide and 80-pass refinement query grids, exclude nlist 1M,
  and retain the main PCA variance 0.95 and vector/router rerank factors 3.
- Do not change baseline implementations or add generic orchestration frameworks
  as part of this work. This file tracks implementation; the benchmark README
  remains the user-facing experiment guide.

## 1. Uniform 15-pass training and assignment admission

### Intended behavior

Main recipes use exactly 15 centroid-update iterations followed by the existing
final assignment. This is a fixed experimental policy, not convergence-based
training and not 16 mean-update iterations.

The training assignment count is `training_count * (15 + 1)`:

| Training population | Assignment searches |
|---|---|
| 90M base vectors | 1,440,000,000 |
| 100M vectors, if explicitly trained | 1,600,000,000 |

The actual current SIFT100M base remains 90M; this plan does not change that
population. Subsequent full-base routing is separate work and must remain
separately accounted for.

### Implementation tasks

- [x] Trace assignment allowances through `TrainerLimits`, `ListBuildOptions`,
  native build/training entry points, cost-model callers, and recipe settings.
- [x] Set the relevant assignment-search defaults to **3,000,000,000**, using
  checked 64-bit arithmetic. Keep actual work estimates in diagnostics.
- [x] Preserve rejection above the declared allowance; do not disable admission
  or automatically raise limits until every requested job passes.
- [x] Keep exact-distance, PCA, graph-refinement, and routing-work allowances
  separate. Do not replace all occurrences of `1000000000` indiscriminately.
- [x] Make main recipes select 15 explicitly and reject conflicting pass overrides
  rather than silently running a differently trained main experiment.
- [x] Remove README exports of `IVF_MAX_REPS=10` and explanations recommending
  the old workaround. Keep individual native/test interfaces configurable.
- [x] Record the fixed pass policy and assignment allowance in existing settings
  and native diagnostics; avoid a second experiment-plan format.
- [x] Ensure reuse checks distinguish 10-pass and 15-pass preparations. Check the
  required provenance fields directly; do not restore a generic validation layer.
- [x] Require matching 15-pass wide-grid bases for refinement reuse. Do not
  silently retrain or mutate a previous experiment's prepared bases.

Primary locations: `include/bufann/ivf_centroids.h`,
`include/bufann/ivf_lists.h`, `src/bufann/ivf_centroids.cpp`, native build and
training callers, `benchmark/IVF/settings.py`, `benchmark/IVF/experiment.sh`,
and `benchmark/IVF/README.md`.

### Validation and completion gate

- [x] Test admission arithmetic for 1.44B and 1.60B searches without actually
  executing billions of searches; test the limit boundary and overflow rejection.
- [x] Verify unrelated exact-distance and routing limits still reject excess work.
- [x] Verify a small 15-pass run emits 15 mean updates plus final assignment.
- [x] Verify every main recipe selects 15; keep tiny configurable test fixtures.
- [x] Verify incompatible prepared training provenance is rejected before reuse.
- [x] Run affected native training, preparation, recipe, and orchestration tests.

Evidence / commands:

2026-11-05, Linux/WSL. Build directory:
`/mnt/c/Users/GBaker/AppData/Local/Temp/opencode/ivf-layout-build-01`.

- `cmake --build <build-dir> -j4`: passed (existing warnings include WSL clock skew).
- `ctest --test-dir <build-dir> -R '^ivf_(centroid_training|mixed_scheduler)$' --output-on-failure`: 2/2 passed.
- The broader 12-test run passed centroid/ANN training, posting lists, cost-model
  CLI, preparation guards, recipe-adjacent orchestration, query harness, sequential
  CLI/parallel insertion, and mixed execution. CTest logs are under
  `<build-dir>/Testing/Temporary/`.
- `python3 -B tests/ivf/test_experiment.py --scratch <approved-temp>`:
  7 tests passed, including conflicting-pass and old-preparation refusal.
- `python3 -B tests/ivf/test_orchestration.py --scratch <approved-temp>`:
  22 tests passed.
- Bash syntax: 73 benchmark scripts and 26 README Bash blocks passed.

Notes / remaining issues:

The 3B limit follows the user's edited plan. Admission boundaries are checked
without billion-operation runs. No full-scale SIFT100M training or experiments
were launched. Exact-distance and PCA/routing guards remain independent.

## 2. Persistent mixed execution

### Intended behavior

`benchmark/IVF/mixed.py` launches one native process for the entire workload.
The driver forks the immutable base once and keeps its mutable state throughout
all rounds. The current checkpoint API remains temporarily available but is not
called by this path; its removal belongs to phase 3.

Round ordering remains meaningful: complete a round's required mutations before
admitting the next round's mutations. Query workers continue across transitions.
Persistent mutation workers may wait for the next round's work; that is not a
global workload stop or a thread-pool teardown.

### Implementation tasks

- [x] Replace Python checkpoint chaining with one driver invocation and a simple
  ordered schedule of existing per-round insert/delete input paths and counts.
  Use a narrowly scoped schedule representation, not a generic job framework.
- [x] Prevalidate schedule structure, ID ranges/counts, paths, and required final
  GT before mutating. Keep trusted input identity checks outside foreground time.
- [x] Refactor the native scheduler so index ownership and worker lifetimes span
  rounds. Preserve the existing scheduler, allocation, and operation mix policies.
- [x] Preserve operation dependencies at transitions. Do not admit a later-round
  mutation early merely to keep threads busy.
- [x] Keep the query stream/cursor and cache state continuous; do not perform
  repeated per-round warmup, complete-file hashing, or base copying.
- [x] Remove mixed-path checkpoint prefixes, per-round private heaps, and reopen
  setup. Retain round output directories only for metrics/evidence that exist.
- [x] Emit completed-round records without joining/restarting every worker. Define
  query attribution by an explicit event, such as query completion; allow queries
  to cross round boundaries without double counting or making snapshot claims.
- [x] Score final recall only after all updates finish and queries are drained,
  against final-live-set GT, using the same mutable index. Never use one static GT
  to claim recall for arbitrary concurrent queries over changing live sets.
- [x] Stop producing mandatory per-round recall/neighbor files. Update consumers
  to distinguish unavailable intermediate recall from zero recall.
- [x] Preserve failures and completed-round evidence; do not publish a successful
  aggregate result when a later round fails. No checkpoint-resume behavior.

### Timing and resource accounting

- [x] Report one active-workload elapsed interval covering continuous execution
  and round transitions, alongside per-round mutation counts and intervals.
- [x] Use summed foreground worker busy time only under an explicitly different
  metric name; it is not elapsed workload time or a throughput denominator.
- [x] Report setup, warmup, final stable scoring, and total workflow time
  separately. Do not reconstruct total elapsed time by summing overlapping spans.
- [x] Report total queries and distinguish update-overlap queries from any query
  tail. Do not inflate concurrent throughput using post-update scoring queries.
- [x] Keep bounded ID/input staging, result buffering, and thread budgets. Do not
  load or duplicate every round's coordinates solely to implement persistence.
- [x] Version changed result/protocol semantics and update tests/CSV consumers.
  Historical files remain untouched; no misleading old metric labels.

Primary locations: `tests/ivf_cli_execution.cpp`, native mixed scheduler/worker
helpers, `benchmark/IVF/mixed.py`, and mixed execution/orchestration tests.

### Validation and completion gate

- [x] Assert one native process, one mutable fork, no inter-round checkpoint or
  reopen, and persistent worker infrastructure.
- [x] Compare cumulative state with a deterministic reference for uneven rounds,
  unequal insert/delete counts, and supported zero-count cases.
- [x] Verify next-round mutation ordering while queries remain active across a
  transition. Check final liveness and no duplicate/missing accounting.
- [x] Validate final recall and returned distances on small exact-GT fixtures.
- [x] Exercise a later-round failure, output collision, and malformed schedule;
  confirm fail-fast behavior and retained earlier evidence.
- [x] Run native mixed, Python mixed, shared orchestration, and recipe tests.

Evidence / commands:

2026-11-05, same Linux/WSL build directory as phase 1.

- Updated `ivf_mixed_execution` passed real float/int8/uint8 prepare/run paths,
  weighted/fixed/update-driven schedules, same-session final ID/distance/GT
  checks, unequal/one-sided native rounds, paths containing spaces, malformed
  schedules, changed IDs, low-budget and no-overwrite refusals.
- The scheduler's cross-boundary fixture deliberately holds a query from round 0
  until a round-1 mutation executes. It verifies persistent mutation-thread
  identity and completion-round query attribution without sleeps.
- Later-round preparation and mutation exceptions are tested in the native
  scheduler; Python harness failure tests verify retained completed-round and
  private-heap evidence with no overall success result/summary.
- Standalone scheduler AddressSanitizer/UndefinedBehaviorSanitizer passed with
  leak detection and halt-on-error enabled. Binary:
  `<approved-temp>/ivf-persistent-scheduler-asan`.
- Standalone scheduler ThreadSanitizer passed 10/10 repetitions using
  `setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1
  <approved-temp>/ivf-persistent-scheduler-tsan`. An initial invocation without
  disabling ASLR failed at sanitizer startup with an unexpected-memory-mapping
  error, before test execution; it is not counted as a test pass.
- Full `ctest --test-dir <build-dir> -R '^ivf_' -j2 --output-on-failure`
  completed in 491.95 seconds: **58 passed, 1 skipped, 0 failed**. This includes
  the final native mixed/scheduler, training, script/orchestration, sequential
  updates, warmup, query composition, mutable-session, heap/I/O, cost-model,
  routing and dependency/build-contract suites.
- The skipped CTest entry is `ivf_gt_depth`: the configured `/usr/bin/python3`
  lacks NumPy. No exact-scan depth qualification is claimed. GT coverage and
  management tests, plus the small final-live-set ID/distance/recall oracles in
  mixed execution, passed independently.
- Final CTest evidence: `<build-dir>/Testing/Temporary/LastTest.log` and
  `<build-dir>/Testing/Temporary/CTestCostData.txt`. The standalone sanitizer
  commands/results above are separate, scoped scheduler checks, not full native
  sanitizer qualification.
- `git diff --check` passed. No full-scale SIFT100M run, commit, or push was made.

Notes / remaining issues:

Main shell profiles retain equal-cap rounds; unequal and one-sided schedules
are covered at the native interface. Fully empty mutation rounds remain invalid.
One largest-round coordinate buffer is reused. Later-round staging and accounting
are included in continuous elapsed time, while initial staging is excluded.
Query counts are attributed by completion; finite-quota workers may wait when a
round's quota is exhausted, but are never restarted. Full-scale throughput,
hard-RSS bounds and production qualification are not claimed. The shared
checkpoint API and sequential mutation checkpoints remain for phase 3; no new
final writeback or FAISS migration has been implemented.

An extra failure test initially assumed the tiny list would grow. It instead
correctly hit preflight admission because that list already had spare capacity.
The test's invalid assumption was removed; later-failure coverage uses the
deterministic scheduler/harness cases above, without weakening admission guards.

## 3. Remove mutable checkpoints; retain writeback-only capabilities

### Intended behavior

The prepared base can be saved and loaded. An updated mutable session cannot be
exported and reopened as a supported updated index. It is disposable, even after
successful final writeback. Written coordinate pages alone do not preserve the
posting lists, logical liveness ledger, or complete allocation metadata.

### Implementation tasks

- [x] Add a narrow explicit writeback operation at the IVF heap/index boundary,
  independent of copying, hashing, serialization, or manifest publication.
- [x] Define writeback as a quiescent lifecycle operation: drain foreground work
  before the final explicit flush. Keep normal dirty eviction during execution.
- [x] Measure final writeback duration, bytes/pages written, and remaining dirty
  pages where available. Do not call it checkpoint, commit, or durable sync.
- [x] Order final processing explicitly: join foreground workers, perform and
  measure writeback, then score on the same index. Snapshot phase counters so
  writeback and scoring I/O do not contaminate foreground metrics. Document the
  ordering and any cache effects.
- [x] Make writeback errors prevent an overall success record. Existing completed
  foreground records may remain as partial evidence, clearly marked incomplete.
- [x] Apply the same single-session mutation-then-scoring policy to sequential
  insert/delete workloads in `benchmark/IVF/harness.py`; remove their separate
  query process dependence on exported updated checkpoints.
- [x] Remove `MutableIndex::checkpoint`, mutable-checkpoint `reopen`, and their
  exclusive mutable-only formats, helpers, CLI flags, metrics, and path checks.
- [x] Remove checkpoint-specific workspace admission from normal mutation and
  mixed execution; keep genuine mutation/query/writeback resource estimates.
- [x] Remove now-unused mutable posting-list and heap-ledger serialization code
  after checking all callers. Do not remove structures needed for live operation.
- [x] Keep immutable preparation save/load, private-session conversion, and shared
  `InPlaceGraphStore` snapshot facilities needed by BufANN or immutable IVF builds.
  A symbol containing `snapshot` is not automatically obsolete.
- [x] Retain I/O error propagation, poisoned-session handling, publication ordering,
  pin admission, liveness checks, and concurrency protections.
- [x] Reject removed mutable formats/options clearly. Do not add legacy fallback
  loading, silently reinterpret old checkpoints, or delete their files.
- [x] Update README, implementation contracts, and result fields to state online
  visibility plus writeback, with no WAL/durable acknowledgment/crash recovery.

Primary locations: `include/bufann/ivf_index.h`, `include/bufann/ivf_heap.h`,
`src/bufann/ivf_index.cpp`, `src/bufann/ivf_heap.cpp`, mutable posting-list
serialization, `tests/ivf_cli_execution.cpp`, `benchmark/IVF/harness.py`,
`benchmark/IVF/mixed.py`, and their tests/documentation.

### Validation and completion gate

- [x] Replace mutable checkpoint round-trip tests with session visibility,
  writeback, storage-failure, and immutable-base protection tests.
- [x] Verify explicit writeback writes dirty data without creating an exported
  index family. Verify it does not claim complete updated-index restartability.
- [x] Verify foreground metrics exclude the separately reported final writeback,
  while total workflow time includes it and in-workload eviction is not excluded.
- [x] Verify failure propagation and no false successful completion on short/failed
  writes. Fault tests validate error handling, not a new crash-recovery guarantee.
- [x] Run sequential update, persistent mixed, query-only, storage, and relevant
  BufANN regression suites to catch accidental removal of shared functionality.
- [x] Confirm no active evaluator requires mutable checkpoint prefixes, mutable
  checkpoint manifests, or checkpoint workspace admission.

Evidence / commands:

Full Linux/WSL build and evidence directory:
`C:\Users\GBaker\AppData\Local\Temp\opencode\ivf-layout-build-01`
(Linux `/mnt/c/Users/GBaker/AppData/Local/Temp/opencode/ivf-layout-build-01`).
The full build includes `BufANN`, `BufANN_s`, `bufann_driver` and its preparation
tools as well as the native IVF drivers/tests. Mounted-workspace clock-skew warnings
remain; they are not compiler/sanitizer findings.

```bash
BUILD=/mnt/c/Users/GBaker/AppData/Local/Temp/opencode/ivf-layout-build-01
cmake --build "$BUILD" -j4
ctest --test-dir "$BUILD" -R '^ivf_' -j2 --output-on-failure \
  --output-log "$BUILD/phase3-ctest.log"
```

Full regression result: **59 passed, 1 skipped, 0 failed** (60 entries),
**333.77 seconds**. `ivf_gt_depth` skips because the configured `/usr/bin/python3`
lacks optional NumPy. The passing storage tests include the shared BufANN
`InPlaceGraphStore::save_snapshot` round trip (`ivf_heap_storage`), immutable
heap/base save/load, direct-I/O faults and concurrent fetch/pin admission. All
four optimized mutable routing modes pass alongside the default native session.
`bufann_driver --help` also succeeds after the full build; no baseline performance
or full-system sanitizer qualification is inferred from these checks.

After final test-only cleanup, rebuilding the mutable test targets and rerunning
`ivf_(mutable_index_session|mutable_posting_lists|writeback_cli|cli_contracts)`
passes **4/4** in **17.85 seconds**; log: `phase3-final-checks.log`. A final
all-dtype native CLI oracle run also passes, including preservation/refusal of
both historical mutable manifest versions. The updated writeback fault fixture
explicitly checks foreground completion before writeback and writeback completion
before final queries; it passes independently and in the final four-test run.

Targeted validation passes the native mutable list/session, parallel insertion,
input-policy budget, sequential scoring/warmup, optimized-router/training, real
shell-harness and persistent mixed suites. `ivf_writeback_cli` passes all nine
mutation-only/sequential/mixed success/error/short-completion combinations. Its
fixture-local `LD_PRELOAD` shim targets only the owned private heap's `pwrite`,
with enough frames to avoid foreground dirty eviction. Failed drains return
nonzero, publish neither root success nor neighbors, and preserve heap/PQ/logs
and completed mixed round accounting. These are error-handling checks, not
crash-recovery tests. Evidence remains under `tests/ivf/writeback_fixtures/run-*/`.

Native session tests cover real dirty-page drain/accounting, clean idempotence,
continued mutation, exclusive writeback admission overlapping query/delete,
workspace refusal, tombstone/aborted-slot visibility and real `RLIMIT_FSIZE`
write failure poisoning. The multi-batch heap fixture has more than 32 dirty
pages, exercising repeated bounded drains. Immutable shared-storage snapshot,
reopen, direct-I/O and read-failure regression tests are retained.

Shared orchestration tests pass **22/22**, including both sides of the minimum
mutation/query budget rule, a single combined command and unchanged driver-owned
native JSON. Python compileall, **73 Bash scripts / 26 README Bash blocks** syntax
checks and `git diff --check` pass; syntax evidence is `phase3-syntax-checks.json`
in the build directory.

Notes / remaining issues:

Writeback uses `flush_dirty_budget(32)` with scratch admission
`65536 + 32 * (page_size + 64)`, not full-pool snapshot allocation. Update results
use protocol version 4; sequential harness provenance uses version 2. Persistent
mixed round/summary version 3 and completion-round attribution stay unchanged.
The same index/pool survives mutation, writeback and scoring. Sequential warmup
follows writeback; mixed warmup remains before all rounds. Foreground counters
and timers exclude final writeback/scoring, while normal dirty eviction remains
included. Workflow time includes writeback.

No WAL, `fsync`, updated metadata export, resume, durable acknowledgment or updated
index restartability is provided. No existing artifact migration/deletion, full-scale
experiment, commit or push was performed. Phase 4 remains unstarted. Historical
phase-1/2 checkpoint notes and older S05/S06 sanitizer results describe their
milestones, not current mutable persistence or phase-3 sanitizer qualification.

## 4. Migrate to FAISS HNSW last

### Intended behavior

FAISS replaces both stock hnswlib routing and the optimized hnswlib-derived
graph used by the main query/training recipes. Preserve the separation between
an unmodified vanilla FAISS comparison and our optimized graph algorithms.

This is the largest technical risk: current code accesses hnswlib internals for
PCA traversal, graph refresh/reinsertion, refinement, and seeded searches.
Substituting a constructor or renaming a mode does not implement the migration.

### Dependency and capability preparation

- [ ] Pin a specific CPU FAISS version/source identity and record its license and
  provenance. Keep optional routing dependencies out of dependency-light targets.
- [ ] Integrate target-scoped CMake linkage and support the documented native and
  isolated-test builds. Disable unneeded GPU/Python components.
- [ ] Resolve BLAS LP64/ILP64 compatibility and OpenMP runtime selection with the
  existing MKL build. Avoid linking conflicting runtimes or oversubscribing workers.
- [ ] Map required operations against the pinned FAISS interfaces. Prefer public
  APIs; identify and document any necessary narrow extension before porting.
- [ ] Freeze pre-migration small-fixture quality/performance evidence after phase 3,
  so graph-backend effects are not confused with lifecycle/persistence changes.

### Vanilla and optimized graph migration

- [ ] Implement vanilla centroid search with FAISS HNSW, preserving centroid IDs,
  metric conventions, configured breadths, and deterministic final tie handling.
- [ ] Separate raw vanilla-search measurements from wrapper oversampling and
  exact reranking costs. Do not label an augmented path as raw FAISS timing.
- [ ] Port optimized full-coordinate construction and projected-distance traversal.
  The PCA path must search the full-coordinate graph using projected distances,
  then rerank full centroids; building a different graph in PCA space is not an
  equivalent substitution.
- [ ] Port bottom-layer refinement and connectivity diagnostics/isolated-node repair.
- [ ] Port training centroid refresh, graph reuse, neighbor-seeded reinsertion,
  and previous-assignment entry-point seeding. Retain LV/LV+ distinctions.
- [ ] Keep all graph mutation in quiescent construction/training phases. Frozen
  runtime graphs must support concurrent queries and insertion routing safely.
- [ ] Use per-call search settings/scratch rather than racing shared `efSearch`
  setters. Test different concurrent query and assignment breadths.
- [ ] Update memory/work estimates and diagnostics for FAISS structures. Avoid
  unsafe global statistics as per-query counters; explicitly mark unavailable
  measurements rather than inventing values.

### Artifacts and integration

- [ ] Implement immutable prepared-router save/load with explicit FAISS and
  optimized-variant identities, including projection/configuration provenance.
- [ ] Update mode names, defaults, native CLI help, recipes, cost-model callers,
  cache/prepared identities, tests, and documentation consistently.
- [ ] Reject hnswlib-produced bundles as incompatible with new FAISS runs. Preserve
  old artifacts and require fresh matching 15-pass preparations for new results.
- [ ] Remove IVF's stock and patched hnswlib dependencies only after all required
  paths migrate. Leave unrelated baseline dependencies and historical evidence alone.
- [ ] Update notices and clearly distinguish upstream FAISS from local extensions.

Primary locations: `cmake/IVF.cmake`, dependency CMake integration,
`include/bufann/ivf_router.h`, graph adapter interfaces,
`src/bufann/ivf_router.cpp`, `src/bufann/ivf_patched_graph.cpp`, ANN training
factories, router artifacts, settings, tests, and dependency documentation.

### Validation and completion gate

- [ ] Test centroid ID mapping, distance calculations, ties, serialization, and
  malformed/incompatible artifacts using small fixtures.
- [ ] Test overlapping query/insertion routing with different per-call parameters;
  run applicable race/sanitizer checks on build, training, and frozen search.
- [ ] Verify PCA traversal/rerank behavior, graph reuse, seeded assignment,
  refinement, and connectivity diagnostics individually.
- [ ] Compare clustering objective, recall, build time, query time, and memory
  against recorded pre-migration evidence. Do not require identical ANN neighbors
  or promise speedups; investigate material quality regressions.
- [ ] Run all affected native and Python suites and verify the documented build
  from a clean build directory with the pinned dependency.
- [ ] Confirm main recipes no longer execute any hnswlib path and do not silently
  downgrade LV/LV+ or PCA-refined modes to vanilla HNSW.

Evidence / commands:

<!-- Fill in executed commands, results, and retained evidence paths. -->

Notes / remaining issues:

<!-- Fill in limitations or deviations before marking the phase complete. -->

## Final integration and experiment release gate

- [ ] Review diffs for unrelated changes and preservation of existing user work.
- [ ] Run the full relevant regression set and record completed commands/results;
  disclose timeouts and rerun incomplete suites rather than treating them as passes.
- [ ] Check Bash scripts and README shell blocks, recipe schedules, output nesting,
  GT preflight, no-overwrite behavior, and successful query-private-heap cleanup.
- [ ] Verify no new SIFT10M support, nlist 1M entries, WAL subsystem, generalized
  orchestrator, or silent artifact compatibility path was introduced.
- [ ] Run small end-to-end query, insert, delete, and multi-round mixed experiments
  with final recall and independent operation/live-set checks.
- [ ] Record final timing boundaries, warmup policy, writeback policy, FAISS version,
  15-pass training policy, and remaining departures from Lindorm.
- [ ] Use fresh output roots and matching FAISS/15-pass bases for SIFT100M results.
  Do not automatically launch expensive full-scale runs merely to close this plan.
- [ ] Keep README as the execution guide; update this checklist with implementation
  evidence and explicitly list any unfinished full-scale qualification.

Final evidence / commands:

<!-- Fill in completed integration checks and separately authorized experiments. -->

Remaining limitations:

<!-- Fill in final scope exclusions and any unexecuted full-scale validation. -->
