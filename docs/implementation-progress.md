# IVF Implementation Progress

**Active addendum — first functional increment (2026-10-09):**
[Lindorm LV/LV+ training fidelity](lindorm-training-fidelity-addendum.md) addresses
optimized training search and breadth/candidate-rescoring separation. Initial
`lv`/`lv-plus` paths are now implemented in the mounted Windows checkout and
functionally checked. The >=32 calibration-policy review is **active**, no longer
deferred. Only PQ/SIMD work remains **deferred and last**. This does not qualify
large-scale performance, reproduce the authors' implementation or update the
separate native source checkout. Existing `ann-reuse-seeded` remains a distinct
full-coordinate, breadth-rerank control. Historical evidence/reports are unchanged.

## LV/LV+ addendum — initial implementation and active sampling review

* **Paper alignment:** both new paths use the same balanced random-partition
  initialization, stable-ID reused full-coordinate HNSW graph, old-neighbor
  reinsertion, one initial frozen bottom refinement/zero-in-degree reinsertion
  pass, and PCA-compressed assignment retaining 95% variance. LV+ adds only the
  previous assigned centroid seed, with first/invalid-seed fallback. PCA is
  refit/reprojected at every joined assignment boundary, including the final one;
  construction/pruning/reinsertion remain full-coordinate. The moving PCA and
  initial-only refinement schedule are disclosed interpretations of unspecified
  paper details, not a claim of authors' implementation identity.
* **Candidate contract:** separate training breadth from returned count. LV/LV+
  default to `min(nlist,3)` candidates for nearest-one full-coordinate double-L2
  rerank. Full-search `ann-reuse-count` and `ann-reuse-seeded-count` return one
  under the graph's float metric; the assigned double-L2 objective is separately
  counted. Legacy modes retain breadth-sized rescoring. Internal frontier heaps
  may remain breadth-sized; only returned/full-rescored candidates are bounded
  by the new count. New paths reject excessive coordinates for float traversal.
* **Integration/accounting:** native builder/diagnostics, qualification and
  calibration tools expose the modes plus independent training rerank/refinement
  settings. New modes use IVFINDEX4 with explicit training-search identity;
  legacy modes still write v3, and old v1/v2/v3 producers retain their semantics
  when copied into mutable origins. Final-router selection is independent.
  Preflight includes projection/solver/refinement/worker memory. Per-pass records
  disclose PCA/refinement setup subsets, summed worker projection/traversal/rerank
  times, requested/effective breadth, actual candidate/objective work, retained
  dimensions/variance and unrepaired connectivity counts.
* **>=32 review:** `Lindorm.pdf` §4.1 uses 32 as a SIMD batch example, not a
  calibration adequacy mandate; §5.2 trains at 20 rows/centroid. New calibration
  reports use v2 and declare `min_sample_per_centroid`, default **32**. Positive
  minima down to **1** support predeclared sensitivity work while enforcing
  `nlist <= m`. Sparse eligible cases warn; ratio/policy/exclusions are saved and
  flow via the frozen report/hash into sidecars/cache identity. Historical v1
  reports retain implicit 32. This does **not** admit 50K into the frozen pilot,
  prove sparse adequacy or promote a lower production default.
* **Executed functional checks:** isolated Release **12/12**, targeted full
  mounted Release **5/5**, scoped scalar ASan/UBSan **3/3**. Fixtures exercise
  all dtypes, deterministic joined workers, independent returned-candidate
  double-L2 oracles over multiple breadth/factor settings, changing PCA rank,
  degenerate/tiny/empty recovery, work/budget failures, typed build/reopen/mutable
  origins, trainer-setting rejection and calibration minima 32/20/1. The initial
  full-build test run was **4/5**: an expanded fixture reused its own output path;
  mode-specific paths corrected the fixture. That failed log is retained, not
  suppressed. The first full build also reached its 120-second shell limit and
  was resumed successfully. Windows-mount clock-skew and existing compiler
  warnings remain disclosed; test times are not performance measurements.
* **Evidence:** mounted `build/lv-fidelity-release-20261009/`,
  `build/lv-fidelity-native-20261009/`, `build/lv-fidelity-asan-20261009/` and
  `build/lv-fidelity-evidence-20261009/`. Final logs and source/tool/configuration
  identities are retained separately from prior S09/S10 artifacts. The separate
  native repository remains clean at `9b1d6243`; no assistant transfer, edit,
  commit or push occurred. No reserved official query was read/scored.
* **Still open:** new-path scoped race-detector qualification, repeated retained
  10K/50K and higher-dimensional training/downstream controls, empirical
  sample/seed adequacy sensitivity, selected-trainer/default decision and fresh
  W8 calibration/validation. Fixed-pass policy and hnswlib substitution remain
  disclosed deviations. No new large-scale calibration or comparison is running;
  M8/SIFT10M and M5/SPFresh remain their existing gates.

The functional core (S00–S06) and **S08's revised W10 hnswlib implementation are functionally complete and fixture-tested**. Native SIFT1M 1024/10K/50K studies and a frozen **SIFT1M W8 prediction pilot** completed on 2026-10-09; these are not complete production qualification. **Required W9/S09 ANN reuse/seeding is implemented**, including native trainer selection, random-partition initialization, v3 manifest/cache identity and bounded diagnostic ablations. User S09/S10 native handoffs match tested identities, and the fresh S10 native check passed 6/6. Exact GEMM is optional, not a prerequisite. **W8/S10 estimator, calibration tool and selection provenance are functionally implemented; the SIFT1M pilot selected the best candidate under its declared probe policy, but underpredicted absolute cost. Required SIFT10M/M8 validation remains pending.** Broader W9 and M1–M4 qualification, required W8 validation, then S07/M5 remain pending. Manual grids/tuning sweeps remain validation controls, not a replacement for the predictor. **SPFresh comparison is deferred, not underway, and requires M8.** Current scope follows [implementation-plan.md](implementation-plan.md) and the reconciled [executive-summary.md](executive-summary.md).

## S10 functional cost model (2026-10-09)

Implemented in Windows `ivf-only`, exercised through the WSL mount, and subsequently
transferred by the user to `/home/omega/bufANN-lindorm-ivf`. Functional coding proceeded ahead of larger-grid
qualification using existing components; production calibration/study prerequisites
remain open. Details and examples: [s10-cost-model.md](s10-cost-model.md).

**Historical S10 native handoff verified before the addendum:** clean commit
`9b1d6243f7f6a8ead7d8f481e52a5a9fedb6a31b`; all 14 S10 handoff hashes match,
along with the retained S09 identities (S10 versions supersede overlapping files).
A fresh full Release build in `build/wsl-s10-native` passed **6/6** targeted tests,
30.02 s: cost-model core/CLI, ANN training/CLI, query/mutation harness and mixed
execution. Evidence: native `build/wsl-s10-handoff-20261009-153459/` (`plan.json`,
`summary.json`, preserved commands/logs). Sources remained unchanged and clean;
the assistant did not transfer or edit native sources.

**10K qualification completed and audited:** native
`build/wsl-w9-10k-20261009-153848/`, with its declared plan, environment, scripts,
commands/resource logs, training artifacts, native bundle, every query/router
record and `verification.json`. Results are in the next section. The declared
50K manual feasibility control also completed and passed its evidence audit under
native `build/wsl-w9-50k-20261009-155413/`, with the same CPU/thread/training
envelope and retained inputs. Its 900K/50K sample averages **18 rows/list**, so it
is **ineligible under the historical/default-32 W8 policy**. The active addendum
now supports separately declared lower-minimum sensitivity reports; this does
not change the prior study or frozen report's exclusion.
A base-only **SIFT1M W8 pilot completed and passed its independent audit** under
native `build/wsl-w8-sift1m-pilot-20261009-164023/`: predictions froze before
single-query-worker tuning-set validation. It includes eligible 1024/10K
predictions and a recorded 50K exclusion/manual validation control, with no
post-validation refit. This pilot does not close the required SIFT10M study.
Calibration completed in 324.24 s and froze **nlist=10000**. Initial preparation
then failed at the query shell wrapper's explicit-`IVF_NLIST` guard, before work
tree creation. The retained `resume-plan.json` and `resume.py` resumed with
`IVF_NLIST=10000` matching the frozen report, with no recalibration or native
source edits. Python preparation still verifies settings/report identity and
records cost-model provenance. Report-only shell preparation remains a known
integration gap (also present in mutation/mixed wrappers); fixture coverage of
Python selection did not catch it. The pilot's prediction and measurement results
are recorded below; no benchmark or audit remains running.
No SIFT10M or SPFresh comparison is underway; the 9000 reserved final queries
are not read/scored.

* Seeded disjoint base sample/query IDs, exact double sample GT, neighbour-cluster
  multiplicity and inclusive radius ties; candidates require `m/nlist >= 32`.
* Production W9 training/frozen W10 assignment/router; measured per-query `T1` and
  radius code count scaled by `N/m` times a scalar W4 scan/selection coefficient.
  Reuses the real scanner, not another ADC kernel. Fetch/exact rerank stay excluded.
* Full-build `ivf_cost_model` target plus `benchmark/IVF/cost_model.py` orchestration;
  frozen commands/settings/source/PQ/tool identities and actual sample/GT/router
  artifacts. Every candidate/exclusion and the mean-`T1+T2` choice are reported.
  All-excluded runs retain an incomplete native report and fail visibly.
* `IVF_COST_MODEL_REPORT` binds the chosen `nlist` and report path/hash to actual
  preparation/cache metadata and query/mutation/mixed experiment sidecars. Source,
  settings and conflicting manual choices are checked; native index/result/scorer
  contracts remain unchanged. Manual builds retain explicit manual provenance.

Executed checks (not a full-project or new TSan qualification):

| Check | Result / evidence under mounted `ivf-only/build/` |
|---|---|
| Clean isolated Release suite | **12/12**, 70.99 s; `s10-core-20261009-144543/`; new core plus existing training/list/routing/contracts |
| Native typed calibration and regressions | **5/5**, 116.80 s; `s10-checks-20261009-145924/`; cost core/CLI, S09 CLI, actual query/mutation harness and mixed execution |
| Final-source native verification | **5/5**, 117.95 s; `s10-final-20261009-150445/`; same selection, including final all-excluded/unique-field fixtures; all 14 implementation-file hashes unchanged during checks |
| ASan/UBSan with leak detection | **2/2**, 1.74 s; `s10-asan-20261009-145924/`; new core/scanner and existing immutable lists, not the full PQ-loading tool/system libraries |

Typed tool checks use independent Python GT/radius/occupancy/scaling oracles,
PCA-refined routing, preserved insertion-tail NaNs, source/settings rejection and
real prepare/cache-restore/query provenance. OpenMP/MKL limits were one for native
fixtures; explicit internal worker tests retain their existing settings. Earlier
S10 build/check logs remain in separate timestamped directories.

Final `summary.json` hashes the changed implementation files and three native
executables; `handoff-files.txt` lists those 14 files for a later user transfer.
`git diff --check` passed and the separate WSL-native execution repo remained clean.
Read-only mounted Git discovery emitted line-ending warnings; it changed no
baseline files. Planning docs remain ignored by `docs/.gitignore`, unstaged.

**Still open:** completed higher-`nlist` W9/W10/W4 production inputs, SIFT10M provision,
qualified SIFT10M calibration, the required recall@10=.90 / recall@100=.95 validation
study and Figure 6 analogue. The user performed the native transfer; the assistant
performed no source transfer, commit or push. The completed 10K/50K qualification
and SIFT1M prediction pilot are separate from the small S10 fixtures, and do not
close the required production study.
M8/M5 and the overall project remain incomplete; fixtures establish mechanics,
not prediction accuracy or an end-to-end optimum.

## Frozen SIFT1M W8 prediction pilot (2026-10-09)

Native evidence: `build/wsl-w8-sift1m-pilot-20261009-164023/` (`plan.json`,
`prediction-freeze.json`, `resume-plan.json`, scripts, calibration artifacts/report,
fresh model-selected bundle, all 84 query/scorer trials, `summary.json`,
`summary.md`, `verification.json`). Calibration used **899900 sample rows and
100 disjoint base-query rows**, seed 0, topk10/target .90, the qualified component
settings used in the manual grid, and the real scalar scan profile. Calibration
accepted no official query/GT argument; their environment variables were removed.
50K is recorded as excluded (`m/nlist <32`), not silently restored as a prediction.

**Freeze:** eligible predictions chose **nlist=10000** before any validation.
Mean predicted T1+T2: **215.51 us at 1024**, **125.81 us at 10K**; mean estimated
probes 10.00 and 23.46. Frozen outer report SHA-256:
`c9d874ea551b8ec47b35b89f0bd6d61dac76f001ad0ecfa06776017f352dcae6`.
The initial shell-guard failure and matching-value resume are preserved above.
Successful resumed preparation took 221.36 s, made a fresh 900K-base build,
and recorded `source:cost_model` with the frozen report identity. Its centroid
bytes match the existing manual 10K build. The report remains unmodified with
`validation_completed:false`; validation evidence is separate, not a report refit.

**Validation:** 1000 tuning queries, one query worker on the declared CPU set,
vanilla router/breadth128/rerank4; one separate-process warmup plus three measured
trials at each of 21 declared probe settings. Per-candidate comparison uses the
**smallest declared probe reaching >=90% on every measured repeat**, not identical
recall, interpolation, or an exhaustive measured optimum over every setting.
Run medians:

| nlist | Predicted T1+T2 us | Selected validation probe | Recall@10 (%) | Measured T1+T2 us | End-to-end mean us |
|---:|---:|---:|---:|---:|---:|
| 1024 | 215.51 | 16 | 92.42 | 448.32 | 1140.69 |
| 10000 | 125.81 | 40 | 91.58 | 354.27 | 914.50 |
| 50000 | excluded/manual | 64 | 91.67 | 376.20 | 1045.69 |

Frozen selected/best ratio is **1.0000** for measured T1+T2 and separately
end-to-end, both among eligible candidates and when including the excluded
manual 50K control under this policy. The 1.10x reference is met **for this pilot**.
This is a finite-grid selection result, not absolute prediction accuracy or M8
completion: measured/predicted T1+T2 is **2.08x at 1024, 2.82x at 10K**. Estimated
mean probes are below the target-reaching fixed validation probes; sample-based
thresholds/calibration timing differ from full-build/tuning validation. No hidden
refit, per-list coefficient or extra prediction candidate was introduced.

At the selected 10K/40 point, separate median routing/scan/ADC/fetch/rerank stages
are **146.30/207.17/4.19/548.39/5.88 us** per query. Median combined T1+T2 is
354.27 us (not necessarily the sum of separately aggregated stage medians).
Fetch/admission/wait and exact rerank remain outside the predictor. Serial-query
times are not comparable to prior four-worker QPS; no CPU-frequency/host-contention
or cold-cache control is claimed. Only the selected candidate was rebuilt with
model provenance; other candidates remain preserved manual controls.

**Independent audit:** all source/tool/input/model/manual-bundle identities and
unique fields passed; base sample/query IDs partition exactly rows 0..899999;
all **200** eligible candidate-query radius/probe/occupancy/scaling calculations
match an independent Python oracle, plus **one** complete 899900-row exact-GT
query. Every repeated neighbor payload agrees byte-for-byte per configuration,
native and unchanged SPFresh scoring agree, and preparation/run sidecars bind
the same frozen report. No official final queries were read/scored, and no native
source changed. Required SIFT10M targets/study, real update qualification, wrapper
report-only fix/coverage, M8 and paired SPFresh remain open.

## Pinned SIFT1M 10K-list qualification (2026-10-09)

The declared next-grid study completed and its independent evidence audit passed.
Native evidence: `build/wsl-w9-10k-20261009-153848/` (`summary.json`, `summary.md`,
`verification.json`, `training/`, `routing/`, native bundle and all query results).
Clean native commit remains `9b1d6243`; sources, executables, input/PQ/tuning-GT
identities and bundle bytes were verified unchanged. Every query record has
unique fields, unchanged SPFresh scoring agrees, and neighbors are byte-identical
across all four repetitions of each probe configuration.

**Envelope:** CPUs `0,2,4,6` (distinct reported cores), four training/query workers,
OpenMP limit four, MKL limit one, 4-GiB estimate budget, 4096 buffer frames
(16 MiB). Identical 900K base rows, random partitions, seed 0, breadth 64 and
15 mean updates plus final assignment. One diagnostic run per training control;
exact nearest-centroid evaluation on 2,048 shared training rows, not holdout or
full exact-trained centroids. Input hashing warms sources; no cache drop,
frequency isolation or host-contention control. No paper-scale speedup claim.

| Trainer | Training s | Evaluated exact-nearest SSE / rebuilt ANN | Evaluated assignment agreement | Empty sample clusters | Bottom-unreachable |
|---|---:|---:|---:|---:|---:|
| ann-rebuild | 174.52 | 1.000000 | 100% | 0 | unmeasured |
| ann-reuse | 163.52 | 1.001057 | 99.951% | 36 | 24 |
| ann-reuse-seeded | 159.37 | 1.000603 | 100% | 0 | 49 |

Diagnostic lifetime RSS is 491–495 MiB. The seeded graph-free native build took
**237.53 s**, including **160.06 s** training, at **501.15 MiB** peak process RSS.
Its centroid bytes exactly match the seeded diagnostic export. All 900K base IDs
are present once; final frozen-router list occupancy: min 0, median 56, p95 299,
max 1320, **two empty lists**. Training-graph occupancy/reachability is separate
from final list assignment and rebuilt query-router state.

Seeded downstream vanilla router, 1000 tuning queries, Recall@10, vector rerank
factor 4, breadth 128; one separate-process warmup plus three measured trials
per probe setting. Medians across runs (percentiles are not pooled):

| nprobe | Recall@10 (%) | QPS | Mean / p99 ms | Scanned codes/query | Fetch share of summed stage wall |
|---:|---:|---:|---:|---:|---:|
| 8 | 67.65 | 1370.53 | 2.909 / 6.027 | 1869.1 | 92.6% |
| 16 | 80.48 | 1379.23 | 2.892 / 6.066 | 3658.8 | 91.2% |
| 32 | 89.70 | 1399.18 | 2.850 / 6.340 | 7179.2 | 87.4% |
| 64 | 94.61 | 1344.07 | 2.966 / 8.354 | 13908.3 | 81.5% |
| 128 | 96.97 | 1295.47 | 3.075 / 10.089 | 26749.8 | 71.6% |

The predeclared manual rule selects **64 probes**, the smallest supplied setting
meeting 90% on every measured repeat. This is not a cost-model selection or
exactly matched vector recall; the grid overshoots the target. At that setting,
median per-query routing/scan-select/fetch stage wall is 0.167/0.369/2.413 ms.
Fetch includes admission/waiting, not pure device latency. The earlier unpinned
1024-list points are **not** controlled performance baselines for this result.

**Fixed-router measurements:** 108 trials on the first 256 tuning queries,
probes 16/64, breadths 64/128/256, six modes, three repeats. PCA retains **51/128**
dimensions (95.0417% variance). Patched-full has zero bottom-unreachable nodes;
refinement reports 181 before one-pass reinsertion and **17 afterward**, despite
zero remaining zero-in-degree nodes. This is a reported connectivity result,
not a reason to add multi-pass repair. At 64 probes and minimum centroid recall
0.99, fastest observed mean router QPS is 67.19K for vanilla/breadth64, 41.80K
for refined/breadth128 and 30.77K for PCA-refined/breadth128. PCA/refinement do not
show a universal gain; selection is over this finite, short-query sweep, not
an end-to-end throughput claim. Diagnostic QPS includes worker joining and match
scoring, uses warm exact-GT preparation and short 256-query intervals; do not
interpret these finite-sweep numbers as precise production mode speedups.
Per-mode recall/timing/distance evidence and
the 0.90/0.95/0.99 selections remain in `routing/summary.json`.

**Closed/advanced:** first adequately sampled real 10K W9 training-control point,
seeded native build/occupancy, repeated manual probe curve, bounded fixed-router
minimum-recall measurements and retained identity/scorer evidence.
**Still open:** repeated/more extensive training quality, downstream rebuilt/reuse
controls, exact-trained reference at scale, 50K/higher-grid results, real update/
mixed/checkpoint workloads, SIFT10M, real W8 calibration/validation and M5/E1–E3.
No new source mechanism, final-query scoring or paired SPFresh comparison occurred.

## Pinned SIFT1M 50K-list manual feasibility control (2026-10-09)

Native `build/wsl-w9-50k-20261009-155413/` completed and passed the same independent
identity, unique-field, occupancy, byte-identical repetition and unchanged
SPFresh-scorer audit as 10K. Training/resources/exact-evaluation scope are the
same. At 18 average sample rows/list, this is an **under-sampled manual control**,
not an eligible W8 candidate. No larger SIFT1M grid is launched by this result.

| Trainer | Training s | Evaluated exact-nearest SSE / rebuilt ANN | Evaluated assignment agreement | Empty sample clusters | Bottom-unreachable |
|---|---:|---:|---:|---:|---:|
| ann-rebuild | 505.20 | 1.000000 | 100% | 16 | unmeasured |
| ann-reuse | 469.31 | 0.995741 | 99.854% | 327 | 153 |
| ann-reuse-seeded | 447.69 | 0.996135 | 100% | 9 | 187 |

Lower bounded SSE is not a full exact-training superiority claim or out-of-sample
quality guarantee. One timing per diagnostic control, no downstream rebuilt/reuse
indexes. Lifetime diagnostic RSS is 604–622 MiB. Seeded native build:
**609.17 s**, including **466.77 s** training, **613.85 MiB** peak RSS. Native and
diagnostic centroid bytes agree. Base occupancy: **48 empty lists**, min 0,
median 3, p95 84, max 760, all 900K source IDs present exactly once. Large skew
and empty lists are retained evidence, not repaired away.

Same vanilla-router downstream scope as 10K, on tuning queries only:

| nprobe | Recall@10 (%) | Median QPS | Mean / p99 ms | Scanned codes/query | Fetch share of summed stage wall |
|---:|---:|---:|---:|---:|---:|
| 32 | 84.65 | 1567.72 | 2.542 / 5.793 | 2829.8 | 86.5% |
| 64 | 91.67 | 1623.48 | 2.452 / 6.605 | 5494.5 | 82.7% |
| 128 | 95.66 | 1603.02 | 2.487 / 6.532 | 10558.2 | 75.7% |
| 256 | 97.34 | 1503.71 | 2.647 / 8.928 | 20113.7 | 59.6% |
| 512 | 97.75 | 1307.04 | 3.043 / 12.846 | 37803.3 | 38.6% |

Manual minimum-90% rule again selects **64 probes**; recall differs from the
10K selection, so their QPS values are not exactly matched-recall speedups.
At this setting routing/scan-select/fetch stage wall is 0.226/0.185/2.028 ms
per query (run medians). No fetch-wait/device-service decomposition is claimed.

108 fixed-router trials on 256 tuning rows: PCA retains **67/128** dimensions
(95.0669% variance); patched-full has **seven** unreachable/zero-in-degree nodes
before and after, while refinement reduces bottom-unreachable **860→49** with
zero remaining zero-in-degree nodes. At probes64/minimum centroid recall .99,
fastest observed mean router QPS is 19.49K vanilla/breadth128, 22.05K refined/
breadth128 and 17.85K PCA-refined/breadth64. At probes256/.99: 10.46K vanilla,
12.52K refined, 5.00K PCA-refined. These short, exact-GT-warmed diagnostic QPS
include joining/match scoring and selection over a finite breadth grid; neither
universal PCA improvement nor end-to-end optimized-mode benefit is established.
All losses/exclusions/raw trials remain in the evidence tree. W9/W10 broader
qualification, W4 telemetry, real updates, SIFT10M and M8/M5 remain open.

## Historical cost-model scope restoration (2026-10-09, before S10 coding)

The user's latest decision supersedes the temporary W8/S10/M8 deferral.
**Status at that documentation-only boundary: required, not started/implemented;
no prediction or calibration evidence.** The functional work above supersedes
that implementation status, not the production-study requirements.
Existing measured sweeps and SIFT1M smoke runs are not cost-model validation.

Remaining deliverables, after qualified W9 training/W10 routing/W4 scan inputs:

1. Seeded base-only calibration sample plus disjoint synthetic query rows and
   exact sample GT; no official tuning/final queries used for calibration.
2. Paper-style per-query probe estimation, measured production-router `T1`, and
   scaled code count times the profiled scalar per-code scan coefficient for `T2`.
   Enforce `m/nlist >= 32` for calibration candidates and record exclusions.
3. `ivf_cost_model` tool and JSON report of identities/settings, candidates,
   predictions and selected `nlist`; carry report path/hash and selection source
   through build/preparation/cache metadata and experiment records.
4. Deterministic small fixtures, then a frozen prediction-vs-measurement study
   on SIFT10M at recall@10 = 0.90 and recall@100 = 0.95. Report the 1.10×
   selection reference against measured `T1+T2`, misses and end-to-end latency
   separately, without refitting after validation.
5. Required Figure 6 analogue and reproducible selection provenance before M5.

Fetch and exact rerank remain **outside** the paper's `T1+T2` model. Their observed
dominance (about 85–87% fetch share of summed query-stage wall) and serialized
heap-fetch admission must be reported alongside predictions, not folded into an
unrequested new model or used to claim that the predicted optimum maximizes QPS.
Manual feasibility builds may exceed calibration sample eligibility; they remain
labeled controls. GEMM is still optional. This is a documentation/scope update,
not an implementation, benchmark launch or change to completed run evidence.

## First W9 native SIFT1M run (2026-10-09)

User-updated execution repo `/home/omega/bufANN-lindorm-ivf` is clean at
`e6d765c6662c15356c1ee7a009d5d6063148d714`. All 25 source hashes matched the
mounted handoff, including the final combined-record JSON fix. Fresh native
Release tools in `build/wsl-s09-native` passed the two targeted ANN training/CLI
fixtures (**2/2**, 5.57 s). No native source edits or dependency installs.

Evidence: `build/wsl-s09-sift1m-20261009-141638/` in that repo (`plan.json`,
`environment.sh`, `summary.json`, `summary.md`, `verification.json`, commands,
GNU-time resource logs and audited native records). Reused prior frozen inputs,
full-source PQ32 and matching base GT without modification; 900K base/100K
reserved insertion tail, 1000 tuning queries, **9000 final queries unscored**.

| Measurement | Result |
|---|---|
| Training configuration | reuse-seeded, random_partition, breadth 64, 1024 lists, 900K training rows, 15 mean updates + final assignment, seed 0, four workers, 4 GiB estimate budget |
| Native build / training time | **373.99 / 242.45 s** |
| Peak build process RSS | **476.04 MiB** |
| Training graph counters | one build, 15 refreshes, 15,360 serial reinsertions; 13.5M prior-seeded / 900K fallback assignments |
| Final approximate assigned SSE | 46,194,839,036.4; not exact-nearest or exact-trained evaluation |
| Training occupancy/reachability | zero final empty clusters, largest 4211; 508 empty recoveries over training; six bottom-unreachable nodes |
| Final base list occupancy | min 1, median 789.5, p95 1952, max 4211, zero empty lists; all 900K rows accounted for |
| HNSW query medians, three trials | **486.02 QPS**, mean latency **8.212 ms**, p99 **12.443 ms**, query process RSS 119.82 MiB |
| Recall@10 / neighbors | **92.42%**; all eight trials across six modes returned byte-identical neighbors, no short results; unchanged SPFresh scorer agrees |

The first vanilla query scanned 19,723.843 PQ codes/query; fetch accounts for
**84.66%** of summed query-stage wall time (including contention). Both native
trees have matching compiler flags. The earlier rebuilt/sample run is **not a
matched training control**: initialization, centroids, recall and scanned work
differ, and CPU placement/frequency/cache were uncontrolled. Neither a training
speedup nor a query regression can be attributed to reuse from these two runs.
Optimized modes each have one trial; no mode speedup claim is made. Training
reachability is distinct from the separately rebuilt final query routers (both
patched/refined final routers report zero unreachable after their frozen repair).

**Closed:** native handoff/rebuild, v3 trainer identity, real-data S09 build/query
integration at 1024 lists, phase counters and bounded tuning-query scoring.
**Pending:** declared higher-`nlist` grid (10K next, then 50K if feasible), matched
rebuilt/reuse/seeded training controls with exact bounded evaluation, matched-recall
query sweeps, SIFT10M and real insert/delete/mixed/checkpoint qualification.
No larger-grid run has been launched; final queries remain reserved.

## S09 implementation — mounted small-scale verification (2026-10-09)

**Ready for larger-`nlist` experimental testing; user handoff/rebuild is complete.**
Functional implementation and scoped mounted verification are complete; the
large-scale quality/performance gate is not. The initial native W9 SIFT1M
build/query passed as described above; the larger-`nlist` grid remains pending.

Implementation edits and mounted tests were in `ivf-only` / its `/mnt/c/.../ivf-only`
mount. The user subsequently transferred the code to the native repo; the assistant
did not transfer or edit native sources. Mechanisms, settings, artifact
changes and user handoff commands: [s09-ann-kmeans.md](s09-ann-kmeans.md).

* Shared sampling/reduction now supports seeded balanced random partitions;
  exact/rebuilt/reuse/reuse-seeded controls share initialization, sample and pass cap.
* Patched graph refresh snapshots old neighbors before any edit, replaces every
  centroid coordinate, then reinserts stable IDs once in serial order using nearest
  eligible old-neighbor entries. Prior-assignment searches bypass global upper
  descent; first/invalid seeds fall back. Workers join before every refresh. No
  two-hop neighbor refresh, multi-pass repair or per-iteration PCA was added.
* Native CLI/harness now default to reuse-seeded/random-partition/breadth 64;
  programmatic defaults retain rebuilt/sample. `IVFINDEX3` and cache settings
  persist trainer identity/breadth separately from final query/insert routing.
  Per-pass time/SSE, build/refresh/reinsertion/seed counters, occupancy summaries
  and measured reachability are exported. Exact training remains scalar.
* Diagnostic ablations can omit exact training at scale while retaining bounded
  exact nearest-centroid evaluation; final sample assignments/occupancy are exported.
  No reference ratio is invented when the exact-trained control is absent.

Final-source evidence: `ivf-only/build/s09-evidence-final/`, with separate mounted
Release/native, ASan/UBSan and TSan build trees. Earlier checks and superseded
timings remain in `build/s09-evidence/`. Current checks:

| Check | Result / scope |
|---|---|
| Isolated Release, clean rebuild | **11/11**, 72.77 s; includes typed reuse/seeding, final SSE/occupancy, worker determinism, joined refresh, fallback/empty recovery and existing routing/list/contract tests |
| ASan/UBSan targeted | **3/3**, 77.85 s: scalar training, new ANN training and patched routing |
| TSan race checks | Final ANN training and patched routing passed scoped runs with `detect_deadlocks=0`; scalar training passed the earlier unchanged check; see caveats below |
| Full native Release, final targeted | **5/5**, 376.39 s: ANN training, typed S09 trainer/manifest/cache/legacy-checkpoint CLI, S08 modes, mixed execution, and unique-field query/mutation/combined CLI contracts; OpenMP/MKL limited to one worker for these fixtures |

**TSan caveats retained:** default deadlock detection reports upstream hnswlib
node/global lock-order inversion during **serial** construction (also vanilla).
Concurrent construction/refresh is not supported. Disabling only that detector
allowed data-race instrumentation; no race report was suppressed. WSL runtime
startup intermittently failed with unexpected memory mappings/silent early
segfaults. Final bounded retry records retain all attempts: ANN passed first;
patched routing passed after one silent early segfault (earlier run: two mapping
failures before success). These are scoped
race checks, not unconditional TSan/deadlock qualification. An initial mounted
FetchContent rename failed with permission denied; retry reused the already
checksum-fetched upstream source, with failure logs preserved. Mounted make
also emitted clock-skew warnings; the Release tree was clean-rebuilt.

### Bounded 128D training observations

4096 synthetic float128 rows, 32 planted Gaussian centers; `nlist=256/1024`,
three mean updates plus final assignment, two workers pinned to CPUs 0–1,
random-partition initialization, breadth 64, same seed/sample across four controls,
two repeats. Exact nearest-centroid evaluation covers 512 training-reservoir rows,
**not holdout**, and does not measure downstream vector-search recall.
`small-summary.json` and per-mode artifacts/logs retain raw results.

| Lists | Trainer | Median training s | Evaluated SSE / exact-trained | Nearest-assignment agreement | Empty sample clusters | Bottom-unreachable |
|---:|---|---:|---:|---:|---:|---:|
| 256 | exact | 0.887 | 1.0000 | 100% | 0 | unmeasured |
| 256 | ann-rebuild | 1.389 | 1.0052 | 100% | 2 | unmeasured |
| 256 | ann-reuse | 1.076 | 1.0124 | 99.80% | 1 | 3 |
| 256 | ann-reuse-seeded | 1.035 | 1.0132 | 100% | 0 | 6 |
| 1024 | exact | 3.022 | 1.0000 | 100% | 0 | unmeasured |
| 1024 | ann-rebuild | 2.282 | 0.9582 | 99.02% | 30 | unmeasured |
| 1024 | ann-reuse | 2.112 | 1.1686 | 93.95% | 137 | 104 |
| 1024 | ann-reuse-seeded | 2.103 | 1.1831 | 96.88% | 61 | 101 |

Diagnostic process RSS, including evaluation/export, was about 29 MiB. Reuse-seeded
was faster than rebuilding at these settings, **but** the short under-sampled
1024-list case (four training rows/list on average) had about 18% higher evaluated
SSE and 101 bottom-unreachable nodes. This is a real quality/reachability caveat,
not grounds to invent connectivity repair or claim production benefit. Different
Lloyd/ANN trajectories can produce a lower-than-one evaluated SSE ratio; exact
training is a local-reference trajectory, not a global optimum. Final query
routers are rebuilt separately on returned centroids. Large-scale convergence,
adequately sampled quality, reachability and downstream recall remain to measure.

**Corrections before handoff:** initial reuse assignment selected the graph's
float top candidate whereas the rebuilt control double-rescored its oversampled
candidate set. Reuse now uses the same breadth/oversampling/double rescoring;
final checks and the table above were rerun. Initial float-only timings in the
earlier evidence root are superseded. The first broader native selection passed
14/15 but caught duplicate trainer/assignment-router fields in combined
update/query JSON. Mutation metadata already supplies these fields; emission is
now conditional, restoring unique canonical fields. Failure evidence is retained;
corrected final CLI/mixed verification passed as part of the final 5/5 selection.
The earlier broader native run's remaining 14 tests passed; this is not a claim
that all 15 were rerun on final sources or that the entire project suite passed.

The user committed the initial S09 implementation as `1beb28b` during work;
the later duplicate-field fix was a separate working-tree edit at mounted handoff.
It is now verified present in the clean native `e6d765c` copy.
`build/s09-evidence-final/handoff-files.{txt,json}` lists/hashes all 25 changed
files relative to S08 (`cb2d760`), including new test/header files. Assistant did
not stage, commit, push, transfer native-copy files or change baseline sources.

## Earlier 2026-10-09 WSL verification and initial SIFT1M run

Execution repo: `/home/omega/bufANN-lindorm-ivf`, commit
`cb2d76017dd8b24ceb8141f0e21e8a50d89ec0c7` (S08), clean sources during the run.
It is the IVF checkout at the repo root, without `docs/`; these planning docs
remain in the Windows `ivf-only/docs/` tree. Helpers and native executables were
built from the WSL repo's own sources, against installed oneAPI MKL/iomp5,
libaio and tcmalloc. No dependency install, native-source edit, commit or push
was performed for these checks.

### Local functional and resource checks

* Routing-enabled isolated Release (`build/wsl-light-isolated`): **10/10** CTests,
  6.10 s, SIMD enabled with `-march=native`.
* Full native Release (`build/wsl-light-native`): **13/13** targeted tests,
  202.08 s: original query/mutable controls, all four optimized-mode query/session
  fixtures, typed S08 CLI/cache smoke, legacy CLI execution and mixed execution.
  OpenMP/MKL were limited to one worker for this selection; explicit query/session
  threads remain exercised. This is not a full-project or new sanitizer claim.
* The repo's narrow PQ helpers passed the trained all-dtype parity/integration
  fixture before real SIFT training. Narrowly compiled unchanged GT/filter
  sources worked; canonical DiskANN top-level MKL discovery remains unqualified.
* `build/wsl-light-evidence/` retains machine/flags, scripts, **72** synthetic
  router trials (4096 centroids, 128D, 1024 queries; full-rank and rank-16,
  six modes × two breadths × three repeats), bounded scalar-exact/HNSW training
  comparisons and the 50K-list cardinality smoke. PCA retained 120/128 and
  16/128 dimensions respectively: it was slower on the nearly full-rank fixture
  and faster on rank-16. These are not production speedup measurements.
* The current 100K-row/13D, 50K-list, two-iteration cardinality control took
  46.86 s for rebuilt-HNSW training and 14.31 s for the final vanilla router.
  This does not qualify 128D SIFT at large `nlist`, or W9 reuse (not implemented at that earlier boundary).

### SIFT1M inputs, scope and measurements

Evidence root:
`/home/omega/bufANN-lindorm-ivf/build/wsl-sift1m-20261009-114223/`.
Retained files include `plan.json`, `input-identities.json`, `environment.sh`,
`run.py`/`run_initial.py`, `finish.py`, `verify.py`, `phases.jsonl`,
`summary.json`, `summary.md`, `verification.json`, inputs, command/time logs,
native index components and every result/neighbor file.

* **Dataset:** standard float32 SIFT1M from a checksum-verified pinned mirror,
  copied into WSL; **not** a SIFT1B-derived subset despite the profile's old
  description. Preserve 1M original row IDs: base `[0,900000)`, reserved insert
  tail `[900000,1000000)`, dimension 128. The profile and SPFresh sources/scripts
  were unchanged; paths were supplied through saved environment overrides.
* **Query reservation:** official rows `[0,1000)` are the initial/tuning set;
  `[1000,10000)` are reserved final queries, copied/hashed but not scored.
  Neither official set was used for centroid training or cost-model calibration.
  Final-set GT still needs preparation for the eventual paired experiment.
* **PQ:** global non-residual PQ8, 32 chunks. The unchanged helper trained on
  the **full 1M source, including the insertion tail**, then encoded all 1M rows
  (32,000,008-byte code file). Its legacy random sampling seed is uncontrolled;
  actual codebook/code identities are retained and frozen, not claimed reproducible
  from a training seed. Measured process phases: 98.47 s training, 26.48 s encoding.
* **GT:** unchanged `compute_groundtruth.cpp` compiled narrowly and run over
  four contiguous 250K-row shards, each returning top 100 for the 1000 tuning
  queries. Merge by `(distance, original source-row ID)` into full-source top 100,
  offsetting shard-local IDs; use unchanged `incremental_groundtruth` to filter
  the 900K base top 10. This bounds the helper's fixed 2048-query distance buffer
  on the 8-GiB WSL instance without changing GT semantics; no padding or GT fitting
  to IVF. All native recall values agree with the unchanged SPFresh tie-aware scorer.
* **Build:** `nlist=1024`, full base sample, 15 iterations, seed 0, four workers,
  4-GiB allowance, 4096 buffer frames, optimized modes built, frozen assignment
  default `hnswlib`. Native build (including bundle work/router reload) **412.30 s**,
  peak process RSS **476.23 MiB**; preparation process phase 415.09 s. This uses
  `hnsw_rebuilt_each_pass`, **not** exact GEMM or W9 graph reuse/seeding. No per-phase
  training breakdown, training SSE or occupancy report was collected here.
* **Queries:** `nprobe=16`, vector rerank factor 4, Recall@10, four workers,
  router breadth 128, router rerank factor 3. Six modes once each, plus two
  extra vanilla repeats. Native open validates/reopens the immutable bundle in
  each fresh process. All eight neighbor payloads were byte-identical; no short
  rows, duplicate result IDs, nonfinite distances or IDs outside the base prefix.

| Measurement | Result / interpretation |
|---|---|
| All six modes, Recall@10 | **91.07%** on the 1000 tuning queries |
| Vanilla QPS, min / median / max (3 runs) | **1234.61 / 1309.96 / 1383.86** |
| Vanilla query interval, median | **0.763 s** for 1000 queries; excludes open/GT/scoring/export |
| Vanilla mean / p99 latency, median across runs | **3.047 / 3.930 ms**; p99 is not pooled across runs |
| Query peak process RSS, maximum vanilla run | **119.88 MiB** |
| PCA retained dimension / variance | **28/128**, **0.951372** |
| Fetch share, median vanilla summed query-stage wall | **87.3%**, including admission/contention; not device I/O or CPU time |
| Per 1000-query vanilla run | 15,238,823 scanned codes; 40,000 reranked vectors; 1000 logical fetch stages/backend batches; 38,097 backend page reads (not device requests) |

The one initial exact-router trial was slower (435.85 QPS); other non-vanilla
modes have only one ordered trial. Cache state, frequency and CPU placement
were uncontrolled. This is **not** matched-recall router-speedup evidence.
`exact` means exact centroid routing, not exact vector search or exact training.
Identical final neighbors do not establish exact centroid recall or training
assignment agreement. The observed fetch bottleneck also means W8's paper-defined
`T1+T2` must be reported separately from end-to-end fetch/rerank latency.

The initial measurement orchestrator stopped after the successful exact-mode
query because it expected mutable-only `live_count`; the immutable record uses
`base_count`. The check was corrected and only missing trials resumed. Native
sources, inputs, index and the completed trial were unchanged; initial script
and correction are retained. No native failure or rebuilding was hidden.

### Which plan items does this resolve?

| Item / prerequisite | What is now closed or advanced | Still open |
|---|---|---|
| Local SIFT1M inputs | Initial-set vectors, full-source PQ and matching GT are provisioned and hash-verified | SIFT10M inputs; final-set GT; standard profile/helper flow portability |
| W1 / M1 | Initial real SIFT1M graph-free build and repeated reopen succeed at 1024 lists | W9 production trainer; high-`nlist` builds, objective/occupancy, phase timing and assignment-quality study |
| W2 / W10 / M6 | Six persisted modes run on one real bundle; PCA truly reduces dimension; final outputs agree | Centroid-only exact recall/distance study, matched-recall repeated routing timings and large-`nlist` benefit |
| W3 | Real global PQ32 trained/encoded by unchanged helpers; full-source/tail policy disclosed | Large-input insert encoding/update evidence; no canonical helper-build qualification implied |
| W4 / M2 | Initial real QPS/recall/latency/RSS/stage counters; bounded candidates and batch fetch observed | `nlist`/probe/rerank sweeps, separate scan/selection, cache hit/miss telemetry and final evaluation |
| W7 / protocol preparation | Native scripts/JSON/results/scorer work on real data; initial settings and tuning/final split saved | Trainer-selection integration, model/manual `nlist` selection provenance and final comparable CPU/cache/tuning protocol; required predictor report plumbing |
| W5/W6 / M3/M4 | Additional tiny fixture evidence only | Real insert/delete/mixed runs, mutable checkpoint/reopen, skew and list-growth measurements |
| W9, SIFT10M, M5, E1–E3, M9 | **Not resolved by this run** | Required implementation/experiments and paired SPFresh report |
| W8/S10/M8 | **Not resolved by this run; required** | Predictor/calibration/report implementation and frozen validation study before M5 |

This closes the **initial SIFT1M build/query smoke and its local input blocker**,
not a complete production milestone. `nlist=1024` is near the usual `sqrt(N)`
scale, not evidence for the paper's high-`nlist` regime. The native query timing
is internally consistent with four workers and the latency records; it does not
justify tuning the cost model to official queries or consuming the final set.

To run another query **from the WSL repo**, without rebuilding:

```bash
cd /home/omega/bufANN-lindorm-ivf
source build/wsl-sift1m-20261009-114223/environment.sh
IVF_RUN_TAG=NEW_TAG IVF_ROUTER_MODE=hnswlib bash benchmark/IVF/sift/query.sh
```

Use a fresh tag; existing results and prepared trees are deliberately refused.
The saved input/helper artifacts support subsequent workload preparation, but
this run did not execute real insert/delete/mixed workloads.

### New training decision (2026-10-09)

At that decision point W9/S09 was **required, not implemented yet** (now implemented
and mounted-tested as above): paper-style random-partition
initialization, ANN assignment, quiescent coordinate replacement, one reinsertion
per centroid seeded by old neighbors, and assignment seeded by the vector's
previous centroid. Retain rebuilt-ANN and scalar exact controls, and integrate
trainer identity into native build/cache/manifest/records. The training graph is
mutable only between joined passes; the final query router/PCA stays frozen.

**Exact GEMM is not the next step or a completion prerequisite.** Existing scalar
exact training and bounded exact evaluation suffice for initial correctness and
quality experiments. Optional later GEMM can extend reference coverage; a sampled
exact SSE on ANN centroids is not a full exact-training comparison. Large-scale
quality losses and reference limits must be disclosed, not assumed away.

**Cost-model decision history (same date):** the user temporarily deferred
W8/S10/M8, then explicitly restored it. The latest required scope is described
at the top of this document and in plan §3; the earlier deferral is superseded.
Manual `nlist` grids, large-`nlist` feasibility and quality measurements remain
necessary for validation. Declare the grid, recall targets, resource limits,
calibration and tuning budgets and selection rule; save the frozen predictor
report and selection rationale before scoring reserved final queries. Give
SPFresh an equivalent tuning budget. The predictor's `m/nlist >= 32` rule applies
to calibration candidates, not every manual feasibility build; resource limits,
`nlist <= training_count` and sample/occupancy disclosure still apply. Both the
predictive Figure 6 analogue and the measured sweep are now required.

## Historical S08 revised-plan verification — functional completion

The from-scratch graph/refresh/multi-pass repair draft was superseded on explicit
user direction. The replacement is a minimal hnswlib extension with frozen
bottom out-edge refinement, one zero-in-degree reinsertion pass, projected
traversal and full-centroid rerank. Release SIMD is restored with bounded
prefetches. See [s08-native-routing.md](s08-native-routing.md).

Finalized-source results, after the empty-neighbor-copy fix and quiescent
PCA/SIMD kernel initialization:

| Check | Result / scope |
|---|---|
| Full-build Release, targeted S08 + W4/W6 | **13/13**; all optimized modes, original query/session controls, router/diagnostic and focused CLI/cache smoke |
| Full-build ASan/UBSan, leak detection | **13/13**, same targeted selection |
| Scoped full-build TSan mutation sessions | **4/4**, all optimized modes; default deadlock detector enabled; one OpenMP worker |
| Scalar TSan router concurrency | **1/1**, race checks enabled; deadlock detector disabled for serial upstream construction |
| SIMD/`-march=native` TSan router concurrency | **1/1**, same deadlock-detector scope |
| SIMD-enabled ASan/UBSan routing | **2/2**, vanilla and patched; leak detection |
| Fixed-input diagnostic sweep | **48/48** trials; six modes × four probes × two breadths, one seed/repeat |
| Python syntax and `git diff --check` | Pass |

The selection deliberately reruns W4 query composition and W6 concurrent-reader
sessions, **not the full S04–S06 workload roster**. Fixtures cover all existing
dtypes; query modes leave assignment/router/PCA/source identities frozen.
Patch-off graph bytes and results match vanilla. Producer identity is preserved
when saving legacy router artifacts so frozen checkpoint digests do not change.

TSan initially reports a lock-order cycle in hnswlib's **serial construction**
on the larger router fixture. Only that fixture disables deadlock detection;
race detection stays enabled. Native sessions retain the known parallel legacy
OpenMP-loader exclusion and do not qualify uninstrumented system libraries.
ASan/UBSan caught upstream `memcpy(nullptr, ..., 0)` during isolated reinsertion;
the common safety patch now guards it. Projected distance spaces are initialized
before readers because AVX-enabled hnswlib constructors write dispatch globals.

Final build/test logs are in each relevant `build/<tree>/s08-final-evidence/`;
SIMD proof logs also live in each isolated tree's `Testing/Temporary/LastTest.log`.
The Release evidence directory includes `machine-and-sources.json` and
`router-sweep/{settings,summary}.json` plus all 48 native records. These are
local, ignored build artifacts, not published production results.

### Synthetic measurements (not SIFT/production)

Fixed seed 0, 768 centroids, dimension 13, 40 float queries, probes 1/4/10/32,
breadths 32/128 and router rerank factor 3, on WSL/GCC 13.3 and a Ryzen 9 5900HX.
PCA retained **13/13 dimensions** at .95 variance: no dimensionality reduction
or PCA speedup is claimed. Rank-one/zero-variance fixtures separately exercise
reduced/degenerate projections. At probe 32, requested breadth 32:

| Mode | Centroid recall | Mean counted distances/query |
|---|---:|---:|
| exact | 1.000000 | 768.000 |
| hnswlib | .989063 | unavailable |
| patched-full | .989063 | 368.625 |
| refined | .977344 | 314.750 |
| pca | 1.000000 | 662.775 |
| pca-refined | .999219 | 606.850 |

Both graphs have measured zero unreachable/zero-in-degree nodes in this random
fixture; the separate duplicate fixture exercises actual one-pass reinsertion.
Distances include patched traversal plus full-centroid rerank (or exact
enumeration). PCA's candidate count raises effective breadth to at least 96 in
this row. The summary selects observed configurations at declared minimum
recall targets (.90/.95/.99), not identical-recall interpolation. Timings are
tiny warmed, uncontrolled trials, possibly overlapping sanitizer activity;
all raw scopes/QPS/stage times are retained, **not interpreted as speedups**.

### Reproduction

After configuring an existing full routing-enabled build with installed
dependencies, build the six targets `ivf_build`, `ivf_driver`, `ivf_search_tests`,
`ivf_mutable_index_tests`, `ivf_patched_router_tests`, `ivf_router_diagnostics`.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/ivf-system-2026 \
  -R 'ivf_patched|ivf_refined|ivf_pca|ivf_query_composition|ivf_mutable_index_session' \
  --output-on-failure -j2
# Repeat with build/ivf-system-s02-sanitized for ASan/UBSan.

setarch x86_64 -R env OMP_THREAD_LIMIT=1 TSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/ivf-s05-native-tsan \
  -R 'ivf_(patched_full|refined|pca|pca_refined)_mutable_session' --output-on-failure
setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1:detect_deadlocks=0 \
  ctest --test-dir build/ivf-s08-simd-tsan \
  -R ivf_patched_centroid_routing --output-on-failure

python3 benchmark/IVF/router_sweep.py \
  --diagnostic build/ivf-system-2026/tests/ivf_router_diagnostics \
  --index-prefix build/ivf-system-2026/tests/ivf/patched_router_fixtures/router_0 \
  --query-file build/ivf-system-2026/tests/ivf/patched_router_fixtures/queries.bin \
  --output NEW_EVIDENCE_DIRECTORY --scope synthetic_fixture_not_production \
  --nprobes 1,4,10,32 --breadths 32,128 --query-rows 40 --query-threads 1 \
  --evaluation-work-limit 30720 --repeats 1 --timeout-s 60
```

Isolated SIMD sanitizer builds use `-S tests/ivf -DIVF_BUILD_ROUTING=ON
-DIVF_ROUTER_SANITIZER_SCALAR=OFF`, Debug, and CXX flags
`-fsanitize=address,undefined -fno-omit-frame-pointer` or
`-fsanitize=thread -fno-omit-frame-pointer -march=native`. Run the vanilla/patched
routing selection `-R centroid_routing` for ASan/UBSan; run the patched fixture
with the documented deadlock-detector setting for TSan.

Earlier revised-source runs and the superseded native implementation's
37-suite run (35 passes, two mid-transition artifact-name failures) are historical,
not substituted for the final evidence above. No production gate or SPFresh
comparison is closed by these results.

## Historical S00–S06 verification (before S08)

| Check | Result |
|---|---|
| Full-project Release (routing and optional trained helpers enabled) | **16/16** suites |
| Selected full ASan/UBSan, leak detection | **8/8** |
| Isolated routing-enabled Release; ASan/UBSan | **8/8** each |
| Isolated routing-disabled | **4/4** |
| Isolated mixed-scheduler TSan | **30/30** repetitions |
| Scoped native TSan (scheduler, mutable session, mixed scripts) | **3/3** suites × 3 repetitions |
| Original `bufann_driver` | Builds |

Native TSan excludes the known parallel legacy OpenMP loader race (run with `OMP_THREAD_LIMIT=1`); system libraries and helper binaries are not instrumented. These are functional results, not production acceptance. Build evidence: [system-build-and-s02-verification.md](system-build-and-s02-verification.md).

## Segment status

Segments are review boundaries, not commits.

| Segment | Title | Status | Scope / next gate |
|---|---|---|---|
| S00 | Logger fix | Done | Logger overflow fix; exact-output regression and ASan |
| S01 | Contracts and PQ adapter | Done (isolated) | Config/input contracts, existing PQ adapter, validation-only CLI scaffolds |
| D01 | Build dependencies | Done for BufANN / narrow helpers | WSL native targets and narrow PQ/GT helpers build/resolve/run; canonical DiskANN MKL discovery still blocked/unqualified |
| S02 | Graph-free heap storage | Done | Graph-free heap adapter, streaming/base-prefix loader, clean reopen, failed-build refusal |
| S03 | Training and vanilla router | Functional; production partial | Scalar exact, rebuilt/reuse/seeded controls and native selection implemented; initial rebuilt-control SIFT1M measured. Large-scale W9 qualification pending; GEMM optional |
| S04 | Immutable lists and query path | Partial | Persisted bundles, batched rerank, scripts/sweep tooling; initial SIFT1M query point measured. Pending: tuned sweeps/full qualification |
| S05 | Mutable index | Done (functional) | Mutable lists/locator, prepared-heap sessions, checkpoint/reopen, sequential mutation CLI and insert/delete scripts |
| S06 | Mixed rounds | Done (functional) | Weighted and fixed 8+8 rounds, round barriers, failure-path hardening, update scripts |
| S07 | SPFresh comparison | Deferred; not underway | Final paired comparison (M5) only after production M1–M4, S08, required S09 and S10/M8 |
| S08 | Optimized centroid routing (W10) | Done (functional); initial real-data check | hnswlib/PCA/refinement verified; six SIFT1M modes run; matched-recall production M6 measurements still pending |
| S09 | ANN K-Means (W9) | Done (functional); initial native real-data check passed | Graph reuse/seeding, initialization, native builder/v3 manifest/cache and diagnostics; mounted scoped verification and native 1024-list SIFT1M run passed; higher-nlist measurement gate open |
| S10 | `nlist` cost model (W8) | Done (functional); SIFT1M pilot audited | Frozen pilot selects 10K under its policy, absolute costs underpredicted; shell report-only gap; required SIFT10M study/M8 before M5 |


## Work-item coverage against the current plan

| Item | Implemented | Remaining |
|---|---|---|
| W1 — build | Base-only sampling, all W9 controls/native selection and graph-free bundles; native 1024/10K/50K SIFT1M builds (50K under-sampled manual control) | Broader W9/SIFT10M builds; downstream trainer controls; GEMM optional |
| W2 — vanilla routing | Pinned hnswlib, safety patch/SIMD, exact/all-list modes, persistence/readers; initial SIFT1M queries | Centroid-only and large-scale routing measurements |
| W3 — PQ | Existing global PQ8, all-dtype helper parity, frozen insert encoder; real full-source SIFT1M PQ32 provisioned and training-tail policy disclosed | Real update evidence; other datasets; canonical helper flow portability |
| W4 — query | Bounded ADC/batch fetch/exact rerank, fixtures, stage counters; SIFT1M 1024-list point and audited pinned 10K repeated probe curve | Separate scan/selection, fetch-service/wait and cache hit/miss telemetry; broader real sweeps/full qualification |
| W5 — mutations | Immediate publication, +16 growth, swap deletion, retained tombstones/aborted slots, checkpoint/reopen, sequential CLI/scripts | Production measurements |
| W6 — concurrency | Weighted and fixed 8+8 scheduling, overlap fixtures, per-round barriers, scoped sanitizers | Production measurements; no full-system TSan claim |
| W7 — harness | Native tools/all four prepare/run pairs, source IDs/JSON/scoring, router families; earlier SIFT1M query path/split; W9 trainer identity/settings/cache; S10 model/manual report provenance in prepare/cache/run sidecars | Final comparable protocol, production qualification and paired report |
| W8 — cost model | Base-only seeded/disjoint estimator/tool/provenance; audited frozen SIFT1M pilot including independent oracles, candidate ranking, absolute-cost misses and 50K exclusion | Report-only shell guard fix/coverage; required qualified SIFT10M validation and Figure 6 analogue before M5 |
| W9 — ANN K-Means | Rebuilt/reuse/seeded, initialization, native integration and bounded diagnostics; audited 10K/50K matched training-control points with 2048-row exact-nearest evaluation | Repeated/broader and downstream control studies; 50K under-sampling disclosed; no GEMM prerequisite |
| W10 — optimized routing | Functional hnswlib/PCA/refinement; audited 10K/50K fixed-router 108-trial minimum-centroid-recall studies each, losses/reachability reported | Broader/repeated and end-to-end mode measurements; no universal real-data speedup claim |

ANN graph reuse/seeding is implemented and small-scale tested. GEMM reference
training remains optional and unimplemented. `ivf_cost_model` is functionally
implemented and small-scale tested; real calibration and validation remain required
before M5. Manual sweeps do not replace W8.
S08 adds `ivf_pca`,
hnswlib patches, mode dispatch and centroid rerank. W4/W6 checks cover every
optimized mode; the full S04–S06 script suite is not multiplied across modes.

### Superseded planning records

Records P01–P08 resolved defects in the previous, expanded plan revision. The current plan drops most of what they specified: the multi-version compatibility contract (P03), p95 probe aggregation and the two-model `T2` (P04–P05), frozen graph refresh and multi-pass connectivity repair (P06), PCA refit and degeneracy handling (P07), and the W9 quality thresholds and mandatory machine-limit protocol (P08). Two rules carry over: status claims stay tied to tests (P01), and functional gates are kept separate from production gates (P02). Patching hnswlib is now permitted, superseding the earlier requirement of a separate decision before changing the pinned dependency.

## Next steps

0. Continue the active LV/LV+ addendum: qualify races and repeated/downstream
   training controls, declare and measure sample/seed sensitivity around 18/20/32
   and a denser reference, then decide defaults before fresh W8 calibration.
   Verify a user handoff before running changed sources in the separate native
   checkout. PQ/scanning remains deferred and last.

1. S09/S10 user handoffs, native rebuild/fixtures, audited pinned 10K/50K studies and the frozen SIFT1M W8 pilot are complete. Preserve separate evidence/input/index trees. No benchmark/audit is running; large-scale qualification/M8 is not complete. Fix/test report-only shell guards before claiming that integration works without explicit matching IVF_NLIST.
2. Retain 50K as an under-sampled manual control, W8-ineligible at 18 rows/list under the historical/default-32 policy. Any lower-minimum sensitivity report must be new and declared, not a retroactive change. Collect missing downstream training controls, repeated quality evidence and real updates rather than assuming broader production qualification. No further SIFT1M grid is active; do not block on GEMM or repair away measured losses.
3. Use the now-configured SIFT1M inputs for real insert/delete/mixed/checkpoint and QPS–recall experiments; measure feasible higher `nlist` with W9 and required telemetry. Configure uint8 SIFT10M with adequate training samples. Check SPFresh build and SIFT1M smoke readiness without comparison numbers.
4. Use the implemented S10/W8 tool with qualified W9 trainer, W10 router and W4 scalar scan inputs after verified native handoff/rebuild. Freeze real calibration reports before declared `nlist`/probe/rerank validation sweeps and save predictions, measured optima, exclusions and misses. The required SIFT10M targets/study are still open. No official queries used for calibration; no post-validation refitting.
5. Complete M8 and save the final plan §4.1 comparable CPU/cache/tuning protocol, model selection and labeled manual controls; prepare final-set GT; only then score the reserved queries in paired M5 runs including E1–E3 and write the report (M9), including the cost-model figure and separately measured fetch/rerank effects.

## Blockers

* **Resolved locally for the initial SIFT1M query set:** vectors, full-source PQ and matching GT have saved WSL paths/identities. Profiles still default to placeholders; use the retained environment. SIFT10M and final-set GT remain unconfigured.
* W9 mechanisms/trainer identity, native handoffs and 1024/10K/50K SIFT1M integration are complete; broader qualification, missing query telemetry and real update/mixed measurements remain open. Quality/reachability losses, 50K under-sampling and TSan caveats are disclosed above. GEMM is optional. W8/S10 native fixtures and the frozen SIFT1M pilot are audited; absolute-cost misses and report-only shell integration gap are disclosed. Required M8/SIFT10M work remains open.
* Canonical DiskANN CMake rejects the installed oneAPI MKL layout. The repo's narrow PQ wrapper and narrowly compiled unchanged GT/filter sources successfully prepared this SIFT1M run; this removes that run's input-generation blocker, **not** the generic README/baseline helper-build portability issue.
* Unrestricted native TSan retains the legacy parallel OpenMP loader exclusion.
* `docs/.gitignore` currently hides all these documents from ordinary `git status`. No ignore rule or staging state was changed by this work. No commit or push is part of this work.

## Segment records

### S00 / S01 — contracts and PQ adapter

* Config and size-checked metadata helpers under `include/bufann/ivf_*` and `src/bufann/ivf_*`.
* Opt-in ADC-only load of `FixedChunkPQTable` (default graph behavior unchanged); IVF uses existing `deflate_vec` and `populate_chunk_distances`. Pivot metadata is validated first; only the current five-offset format is accepted.
* `ivf_build`/`ivf_driver` targets with strict validation modes; no simulated metrics. Dependency-light CTest in `tests/ivf/README.md`.
* S00: ASan found an overflow in `ANNStreamBuf::overflow`, reached by PQ loading; fixed with flush-before-append and regression-tested.
* Validation is compiled without fast-math, since the parent's `-Ofast` breaks NaN/Inf rejection.
* Results (Ubuntu WSL, GCC 13.3, CMake, Python 3.12): isolated Debug, ASan+UBSan and Release **2/2** each.

### D01 — MKL / iomp5 build dependencies

Verified in Ubuntu WSL; supersedes earlier D01 notes, whose local-prefix workaround was rolled back.

* MKL 2026.1 (`/opt/intel/oneapi/mkl/2026.1`), Intel OpenMP (`/opt/intel/oneapi/compiler/2026.1/lib/libiomp5.so`), `libaio-dev` 0.3.113-6build1.1, `libgoogle-perftools-dev` 2.15-3build1. The installation method is not inferred; no system changes were made during verification.
* CMake uses explicit roots in `build/ivf-system-2026`, preserving ILP64 / `mkl_intel_thread` / `iomp5` linkage.
* BufANN library, `bufann_driver` and IVF targets build; `ldd` resolves all dependencies with iomp5 as the OpenMP runtime; an SGEMM/OpenMP/libaio probe passes; full CTest passed **4/4** at this boundary.
* Unqualified: canonical DiskANN MKL discovery, and shared scripts' `BENCH_DEPS_PREFIX` interaction with this installation.

### S02 — graph-free heap storage

`HeapStorage<T>` (float/int8/uint8, `ivf_heap` target) over zero-degree, no-PQ `InPlaceGraphStore`, with stable external/node IDs, guarded copies into aligned scratch, and a bounded streaming loader with base-prefix support. Existing artifact paths are refused, not truncated. Clean checkpoints pair backend v4 metadata with a typed `.ivf` descriptor; failed builds can't checkpoint; no rollback or WAL. Results: Release **4/4**, adapter ASan/UBSan **1/1**, four concurrent readers. Details: [s02-heap-adapter.md](s02-heap-adapter.md).

### S03 — training and vanilla router

* Seeded base reservoir sampling, blocked **scalar** exact Lloyd assignment, deterministic reduction, empty-cluster recovery, guarded work/memory estimates. The scalar trainer remains the tractable correctness reference; required W9 is the intended large-`nlist` production path. GEMM is optional later reference work.
* HNSW-assignment trainer that rebuilds every pass, without reuse/seeding. It was the original builder and the initial SIFT1M run's trainer; it remains a control. The S09 builder now dispatches selected modes instead; earlier SIFT1M numbers are not W9 reuse/seeding evidence.
* Router: pinned, unmodified hnswlib v0.8.0 with list-ID labels, typed query conversion, exact/all-list modes, serial build reproducibility, per-operation breadth via oversampling, and validated persistence; no shared `ef` setter is raced.
* ASan exposed an upstream SSE-prefetch out-of-bounds read, so all router builds use `NO_MANUAL_VECTORIZATION`, with no patch. The plan now fixes the read as a patch and restricts the flag to sanitizer builds (S08, pending).
* Results at this boundary: Release **6/6**, isolated ASan/UBSan **4/4**, routing-disabled **3/3**. Centroid recall@10 = 1.0 (40 queries, 1024 13-D centroids); 1024-row/64-centroid HNSW-trainer SSE ratio 0.998775 vs. exact; 100K-row/13-D, 50K-centroid, two-iteration smoke: training 23.0 s, total 33.6 s, peak RSS 41,840 KiB. Synthetic only.

Details: [s03-centroid-training-routing.md](s03-centroid-training-routing.md).

### S03/S04 — qualification tooling

`ivf_training_diagnostics` and `qualify.py` run exact and HNSW training in separate processes on identical reservoir IDs and compare exact SSE on the same rows, with work/memory guards, timeouts and RSS measurements. Repeated query sweeps retain every record, exclude separate-process warmups, optionally pin CPUs, and summarize min/median/max across runs. Cache dropping and paired SPFresh evaluation are not implemented. Usage: [benchmark/IVF/QUALIFICATION.md](../benchmark/IVF/QUALIFICATION.md).

### S04 — immutable lists and query path

* Base routing and byte-preserving repacking of existing PQ codes into bounded immutable CSR lists; bounded ADC selection with deterministic ID ties.
* Real backend/libaio batched coordinate copies and exact double-L2 rerank. A tiny-pool test showed total frame capacity doesn't bound pins, so batches respect the smallest shard with serialized pin/copy admission. Counters separate scanned/reranked rows, batch calls and heap-page reads.
* All-list/full-budget queries match exact oracles for every dtype; limited-probe queries match independent ADC + rerank oracles.
* Checksummed CSR persistence and a `NativeIndex<T>` build/open/query API with strict manifests, relocation support and corruption/mismatch refusal. Integrity checks are not authentication or atomic publication.
* Native CLI builds and search-only queries; query prepare/run scripts with SHA-256-keyed cache identity. GT tags are accepted only when they equal source IDs.
* Trained-helper parity via a narrow wrapper: 12,288 identical code bytes per dtype on 4096-row/13-D fixtures. This doesn't qualify the canonical DiskANN helper build.
* Results at this boundary: Release **13/13** with trained helpers; isolated routing-enabled Release and ASan/UBSan **7/7** each; routing-disabled **3/3**.

Details: [s04-lists-query-components.md](s04-lists-query-components.md), [s04-index-bundles.md](s04-index-bundles.md), [s04-cli-execution.md](s04-cli-execution.md), [s04-query-harness-and-trained-pq.md](s04-query-harness-and-trained-pq.md).

### S05 — mutable index

* **Lists:** stable per-list reader/writer locks, contiguous ID/PQ8 arrays with 16 spare slots and +16 growth, preallocated locators, CAS insertion reservations, swap deletion with moved-locator repair. Missing deletes are idempotent and counted; deleted IDs are never reused.
* **Sessions:** `MutableIndex<T>::fork` onto a verified private heap copy. Inserts reserve, route, encode and prepare the heap record while hidden, then publish list membership. Erase removes list visibility before heap retirement. Search scans under list locks, releases them before heap I/O, and revalidates liveness; queries are weakly consistent. Backend failures poison the session; no physical rollback.
* **Checkpoints:** exclusive capture against shared operations. `IVFMUTABLE1` manifests (ten components), `IVFMPL1` lists, `IVFMHS1` heap states and an `IVFHEAPM1` descriptor preserve membership, order, capacities, tombstones and aborted slots. Reopen goes onto new private files after strict validation. Outputs are claimed exclusively; not fsync, atomic or WAL.
* **CLI and scripts:** `insert_only`, `delete_only` and `search_after_update`; `benchmark/IVF/sift/{insert,delete}_{prepare,}.sh`. Update metrics go to `mutation_result.json`, final query metrics to `native_result.json`/`result.json`. One mutation worker, no overlapping queries.
* **TSan:** GCC 13.3 TSan needs per-process `setarch x86_64 -R`. List fixtures pass **30/30**; native session **10/10** and mutation CLI **1/1** with `OMP_THREAD_LIMIT=1`.
* Results at this boundary: Release **14/14**; selected ASan/UBSan **6/6**; isolated **7/7** each; routing-disabled **3/3**.

Details: [MUTABLE_LISTS.md](MUTABLE_LISTS.md), [MUTABLE_INDEX.md](MUTABLE_INDEX.md), [s05-cli-and-mutation-harness.md](s05-cli-and-mutation-harness.md).

### S06 — mixed rounds

* `tests/ivf_mixed_workload.h`: weighted mode reuses `WeightedUpdatePool`'s policy; fixed mode launches eight mutation and eight query workers before admission. Failures stop admission, then join and rethrow. `WeightedUpdatePool` received only a join-on-thread-creation-failure fix.
* `--workload mixed` caps OpenMP/MKL at one, keeps page-data locks on, and starts no background flush worker. Concurrent queries measure latency, QPS and overlap, not recall; post-round barriers check counts and membership, checkpoint, then score against round GT.
* Scripts `update_prepare.sh`, `update.sh`, `_mixed_profile.sh` and `benchmark/IVF/mixed.py` keep the canonical layout and SPFresh's equal round caps, run one native process per round with checkpoint/reopen chaining (≤4096 rounds, 8 MiB metadata), and write `summary.json` with `production_qualified: false`.
* **Hardening:** `MixedStartGate` now publishes release/abort under its wait mutex, fixing a lost wakeup on fixed-mode launch failure. Canonical metrics are extracted and checked per round before the next round or final publication; previously a mismatch could leave a successful final summary. Summary publication is not atomic or crash-safe.
* Final results are those in "Current verification." Native TSan command:

```bash
setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1 OMP_THREAD_LIMIT=1 \
  ctest --test-dir build/ivf-s05-native-tsan \
  -R 'ivf_mixed|ivf_mutable_index_session' --repeat until-fail:3 --output-on-failure
```

Details: [s06-mixed-workloads.md](s06-mixed-workloads.md).
