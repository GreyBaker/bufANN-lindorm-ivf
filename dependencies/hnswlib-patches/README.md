# hnswlib v0.8.0 IVF patches

`apply.py` creates private header copies; exact anchors refuse another upstream
version. The five labeled sections are the patch series:
Configuration writes individually reviewable unified diffs to
`<router-binary-dir>/ivf-hnswlib/patches/01-*.patch` through `05-*.patch`.

1. **Memory safety, both builds:** guard empty/full adjacency lookahead,
   out-of-array visited pointers and empty neighbor copies during reinsertion;
   leave SIMD distance kernels enabled.
2. **Traversal context:** per-call second vector array and distance counter
   through the existing upper/base traversal, with no mutable query settings.
3. **Caller entries:** prefer a closer supplied bottom-layer entry; retain the
   global descent as fallback.
4. **Quiescent edits:** validated bottom out-edge replacement and one ordinary
    existing-ID reinsertion through upstream `updatePoint`.
5. **ANN training reuse/seeding:** explicit training-only seeded searches skip
   global upper descent; empty seeds retain global fallback. After replacing all
   coordinates, reinsert each stable ID once through upstream per-layer update
   search/pruning/reciprocal connection. At each layer select the nearest
   eligible pre-refresh neighbor as the entry (ordinary global descent fallback
   if none). This does not run upstream `updatePoint`'s extra two-hop neighbor
    refresh loop or post-build frozen refinement. Node levels/labels stay fixed.

Namespace separation is build plumbing. Default search retains upstream queues,
stopping rules, pruning and visited pools. Frozen refinement and zero-in-degree
measurement live in `src/bufann/ivf_patched_graph.cpp`. No native graph,
multi-pass repair or alternative graph algorithm is included. The training-only
refresh adapter snapshots all old neighbor lists before any coordinate/edge edit;
assignment workers must be joined before refresh. The query router's default
traversal and frozen artifacts remain unchanged.

Training LV/LV+ uses the same patched traversal over a transient projected
centroid array. Coordinate refresh/reinsertion always uses the graph's original
full-coordinate distance space; PCA is refit and its distance kernel reconfigured
only after assignment workers join. Initial refinement is separate from later
old-neighbor refreshes. There is no extra refinement/repair loop per iteration.

For build, router flags and artifacts see the
[IVF harness guide](../../benchmark/IVF/README.md#optimized-centroid-routing).
For centroid-only diagnostics and fixture commands see
[qualification tools](../../benchmark/IVF/QUALIFICATION.md#fixed-router-diagnostics).
`cmake/IVFCentroids.cmake` runs `apply.py` during routing-enabled configuration;
the fetched archive and generated headers/diffs stay in the build tree, not the
source checkout. The upstream pin and license are recorded in
[hnswlib.md](../hnswlib.md).
