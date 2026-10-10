# IVF centroid HNSW dependency

* Upstream: https://github.com/nmslib/hnswlib
* Based on release **v0.8.0**.
* Archive: https://codeload.github.com/nmslib/hnswlib/tar.gz/refs/tags/v0.8.0
* SHA-256: `cf61d7dd8dc9bfba7f4abe0ed26698e90ac9f49a0badb2e1b0f3ba89b72cf3bb`
* License: **Apache-2.0**, not MIT; complete upstream [LICENSE](LICENSE).

## Local IVF implementation and external comparison

`include/ivf_patched/` contains the seven modified, checked-in hnswlib headers
used by IVF training and optimized routing. These are authoritative source files:
edit them directly and review changes through Git. There is no patch application,
header generation or Python step during dependency configuration. Each header
identifies its modifications; namespace `ivf_hnswlib` separates this version from
the external comparison's `hnswlib` namespace.

The `hnswlib` router mode uses **unmodified upstream v0.8.0**, not a second local
copy. `cmake/Hnswlib.cmake` fetches its checksum-pinned archive into the build tree
when `IVF_BUILD_ROUTING=ON`; it never edits upstream or configures upstream
executables/Python bindings. Routing OFF needs no download. Existing routing
builds still include the stock mode, so they need the fetch even when an optimized
mode is selected at runtime. For offline builds, supply an already extracted,
unmodified archive using `FETCHCONTENT_SOURCE_DIR_IVF_HNSWLIB`.

The `ivf_hnswlib_headers` INTERFACE target exposes the local and external include
roots to explicit consumers. No upstream comparison headers are vendored in this
directory. `ivf_hnswlib_dependency` checks the local file/namespace ownership and,
with routing ON, the exact SHA-256 of all seven upstream headers. Functional
training/router fixtures cover the local extensions.

## Local modifications

The initial vendored version matches the previously generated optimized headers,
apart from prominent modification notices. It retains these changes:

1. **Memory safety:** guard empty/full adjacency lookahead, out-of-array visited
   pointers and empty neighbor copies during reinsertion; retain SIMD kernels.
2. **Traversal context:** per-call second vector array and distance counter
   through existing upper/base traversal, with no mutable query settings.
3. **Caller entries:** prefer a closer supplied bottom-layer entry; retain global
   descent as fallback.
4. **Quiescent edits:** validated bottom out-edge replacement and one ordinary
   existing-ID reinsertion through upstream `updatePoint`.
5. **Training reuse:** seeded searches skip global upper descent; empty seeds
   retain fallback. After replacing all coordinates, reinsert each stable ID once
   through upstream per-layer update search, pruning and reciprocal connection.
   Select the nearest eligible pre-refresh neighbor at each layer, with ordinary
   global descent fallback if none. No extra two-hop refresh or post-build frozen
   refinement; levels and labels remain fixed.

Default local search retains upstream queues, stopping rules, pruning and visited
pools. Frozen refinement and zero-in-degree measurement remain in
`src/bufann/ivf_patched_graph.cpp`. The refresh adapter snapshots all old neighbor
lists before any coordinate/edge edit, after assignment workers join. LV/LV+
traverses transient projected coordinates; refresh uses full coordinates. PCA is
refit only after assignment workers join. No alternative graph algorithm is added.

## Comparison and sanitizer caveats

Before vendoring, the comparison also received the memory-safety fixes. It is now
truly stock upstream; historical measurements remain safety-corrected comparison
results and must not be relabeled. New comparisons require fresh build identities.
CLI defaults and graph artifact formats are unchanged, but the stock dependency
implementation has changed from the previous safety-corrected build.

New router descriptors and index manifests identify this producer as
`hnswlib-0.8.0-stock1`. Readers still accept historical
`hnswlib-0.8.0-prefetch1` and the older scalar `hnswlib-0.8.0-nomv` family.
Copies and mutable checkpoints preserve the loaded producer identity, not the
current build default. This label describes the comparison dependency; it does
not imply that the local IVF graph is unmodified upstream or SIMD-qualified.

Unmodified v0.8.0 retains known prefetch lookahead issues, including the neighbor-ID
read previously found by ASan in `hnswalg.h:277-278`. Sanitizer builds disable manual
HNSW SIMD by default via `IVF_ROUTER_SANITIZER_SCALAR` for both variants, avoiding
that SIMD prefetch path without modifying upstream. Stock SIMD runs are not
memory-safety-qualified; scalar sanitizer checks do not qualify the SIMD path.
No production equivalence or speedup is claimed without measurements.

Upstream upgrades are deliberate source updates: obtain a pristine new release,
review its differences and port the local modifications. Update provenance and
comparison pin/checks together, then rerun functional and sanitizer qualification.
Do not regenerate or silently overwrite the maintained local headers.

For router flags and artifacts see the
[IVF harness guide](../../benchmark/IVF/README.md#settings-and-frozen-routing)
and [fixed-router diagnostics](../../benchmark/IVF/README.md#fixed-router-diagnostics).
Distributions containing the compiled router must include this license and
retain applicable upstream attribution. Existing BufANN/DiskANN sources retain
their own licenses.
