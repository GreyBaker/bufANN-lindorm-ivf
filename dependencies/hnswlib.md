# IVF centroid HNSW dependency

* Upstream: https://github.com/nmslib/hnswlib
* Selected release: **v0.8.0**, pinned upstream with local build-tree patches.
* Archive: https://codeload.github.com/nmslib/hnswlib/tar.gz/refs/tags/v0.8.0
* SHA-256: `cf61d7dd8dc9bfba7f4abe0ed26698e90ac9f49a0badb2e1b0f3ba89b72cf3bb`
* License: **Apache-2.0**, not MIT. The complete upstream license is
  [hnswlib.LICENSE](hnswlib.LICENSE) and remains in the fetched archive too.

`cmake/IVFCentroids.cmake` fetches/extracts the checksum-pinned archive into the
build tree only when `IVF_BUILD_ROUTING=ON`. Default OFF requires no download.
No Python bindings, upstream executable targets or library installation are
needed. No Faiss dependency is added. Headers are private to `ivf_router`.

Sanitizer builds use upstream's `NO_MANUAL_VECTORIZATION` definition by default.
An ASan test found a four-byte out-of-bounds neighbor-ID lookahead in
`hnswalg.h:277-278`, when an upper-layer neighbor array is full; the stock SSE
path allocates only one spare byte. A shared prefetch-bounds patch now guards
these lookaheads, including existing-ID reinsertion. Release retains hnswlib's
SIMD distance kernels. The optimized copy adds traversal/refresh patches; see
[hnswlib-patches/README.md](hnswlib-patches/README.md). No production performance
equivalence or speedup is claimed without measurements.

Distributions containing the compiled router must include the dependency's
license and retain applicable upstream attribution. Existing BufANN/DiskANN
sources retain their own licenses.
