# IVF FAISS HNSW dependency

* Upstream: https://github.com/facebookresearch/faiss, release **v1.9.0**.
* Archive: https://codeload.github.com/facebookresearch/faiss/tar.gz/refs/tags/v1.9.0
* SHA-256: `f6721f1a479b0bdd8547e5bb0dbfa872643e7ea37686c6bf6fc99fb933b3fe4f`.
* License: **MIT**, full upstream text in [LICENSE](LICENSE).

`cmake/Faiss.cmake` fetches unmodified upstream sources only when routing is ON.
The target-scoped CPU library consumes the pinned upstream CPU source manifest;
no GPU, Python, demos, upstream tests, header generation or patch application.
Offline builds can set `FETCHCONTENT_SOURCE_DIR_IVF_FAISS` to that exact extracted
release. `ivf_faiss_dependency` verifies central upstream source/header hashes.
Routing OFF retains dependency-light contracts/training without FAISS or BLAS.

## API/capability boundary

| Capability | Implementation |
|---|---|
| Vanilla construction | Unmodified `IndexHNSWFlat::add`, serial batch construction with configured RNG seed/M/efConstruction |
| Vanilla traversal | Unmodified `HNSW::search`, `IndexFlat` distance computer, local visited/result context and `SearchParametersHNSW` |
| Projected traversal | Same full-coordinate FAISS graph; per-call projected-array `DistanceComputer`, full-coordinate double rerank afterward |
| Previous-assignment seeding | Public `HNSW::search_level_0` from the supplied stable ID, bypassing global descent; empty seed falls back to ordinary search |
| Refresh/reinsertion | Local quiescent adapter snapshots old per-layer neighbors, replaces all coordinates, then reinserts each ID from its nearest old eligible neighbor using FAISS traversal and pruning |
| Bottom refinement | Local frozen-pass candidate collection and FAISS `shrink_neighbor_list`; replace bottom out-edges, then one zero-indegree reinsertion pass |
| Diagnostics | Local directed bottom-layer reachability/indegree, not a promise of connectedness |
| Immutable persistence | Checked local `IVFFAIS1` framing for FAISS levels/offsets/edges/full coordinates; no unchecked upstream `read_index` on input |

No upstream source extension is required. The local optimization is in
`src/bufann/ivf_patched_graph.cpp`; its historical adapter name does not denote a
patched FAISS source tree. Reinsertion is an explicit local algorithm, not an
upstream FAISS update API or a claim of identical hnswlib update trajectories.
When an old neighbor is unavailable, the adapter uses a non-self per-layer anchor;
stable IDs, levels and full-coordinate construction are preserved.

The high-level `IndexHNSW::search` updates global `hnsw_stats` without protecting
independent callers in this release. Frozen routing instead uses the unmodified
low-level search interface; it never reads/resets those globals. Optimized counters
are local distance-computer calls. Vanilla graph counters are labeled unavailable
and do not pay counting instrumentation. The raw kernel time excludes projection,
output double rescoring, sorting, and wrapper validation; end-to-end router time
must not be labeled raw FAISS timing. `faiss` returns nprobe candidates without
breadth-sized oversampling; `faiss-full`/`refined` retain explicit oversampling.

Visited storage is one exact-size epoch table per calling thread, not a fresh
O(nlist) allocation/clear for every assignment or refreshed centroid. Each call
leases it exclusively; reentrant calls get independent temporary storage. The
lease handles upstream normal-return advancement and seeded/exception paths;
upstream clears once per 249 epochs. Switching graph sizes releases the previous
cache before allocating, and thread exit frees it. Worker estimates include this
nlist-byte cache, which remains on idle caller threads after routing. They are
per-workload allowances, not a process-wide cap across arbitrary external threads
or nested callers. Optimized `graph_us` includes adapter context/results work;
vanilla `graph_us` times only the upstream search call, excluding context setup,
cache allocation, lease cleanup and output materialization (upstream's own epoch
advance/periodic clear stays inside its kernel). Router wall time includes all.

## BLAS/OpenMP and artifacts

Native IVF uses **sequential MKL ILP64**, with FAISS `FINTEGER=int64_t`, and only
the compiler's GNU OpenMP runtime. It reuses the unchanged BufANN backend archive
without inheriting the baseline target's Intel-threaded MKL/iomp5 link interface.
Baseline targets keep their original policy. Isolated routing builds use generic
system LP64 BLAS/LAPACK (`FINTEGER=int`) plus compiler OpenMP; install
`libblas-dev liblapack-dev`. Construction temporarily sets the calling OpenMP
task's thread count to one and restores it. Search uses per-call efSearch and
never changes shared search breadth or process-wide BLAS settings.

On Debian/Ubuntu, the build prefers the reference `blas/` and `lapack/` library
directories, with CMake runtime search paths. Merely requesting `BLA_VENDOR=Generic`
is insufficient when system alternatives redirect `libblas.so` to `mkl_rt`:
that dispatcher can dynamically load Intel OpenMP alongside GNU OpenMP. Resolved
MKL/iomp5 libraries are rejected in isolated mode rather than accepted silently.

Producer IDs are `faiss-1.9.0-cpu1` and `faiss-1.9.0-ivf1` (optimized adapter).
Router descriptors use versions 3/4; the HNSW component suffixes remain `.hnsw`,
`.patched.hnsw`, `.refined.hnsw` but their checked contents are FAISS, not hnswlib.
New prepared/cache identities bind the FAISS backend. All former hnswlib-produced
bundles are rejected; preserve them and rebuild into fresh FAISS/15-pass prefixes,
never relabel them. License/notice and dependency hashes must accompany upgrades.
