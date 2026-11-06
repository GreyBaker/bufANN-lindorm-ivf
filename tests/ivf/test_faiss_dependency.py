"""Pinned upstream identity, no patch application, and scope/ABI contracts."""
import argparse
import hashlib
from pathlib import Path

HASHES = {
    'LICENSE': '52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85',
    'faiss/IndexHNSW.cpp': 'fa4bc705f3b06c9b13e0a5309560e8533538ef293f98ca3ad2f7e570ea851e2f',
    'faiss/impl/HNSW.cpp': '9154291206e402d1239872e2591a23624ce191ea93959533a0755ec8d7c5e820',
    'faiss/impl/HNSW.h': '971f6cb33b0fe5e6b2d3c0f374ee1b0ae259a9b90f21c649cd58e9d2777f35a2',
    'faiss/impl/DistanceComputer.h': 'cd42ed87e664e8f8d80bbb6fda0cc29da0a2b901034a1b7c60dc4ae1def979fe',
    'faiss/utils/distances_simd.cpp': 'fc38fd159aa4dddbf6a036aba3d3602c662c672c6a67c68ca9cd18e38388b314',
    'faiss/utils/random.cpp': '9ad98838c1f355b5adbd474188964e9ff65735f4d26c38562f3ef26691fe443f',
}
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True, type=Path)
p.add_argument('--source', type=Path)
a = p.parse_args()
cmake = (a.repo / 'cmake/Faiss.cmake').read_text()
assert 'SHA256=f6721f1a479b0bdd8547e5bb0dbfa872643e7ea37686c6bf6fc99fb933b3fe4f' in cmake
assert 'execute_process' not in cmake and 'find_package(Python' not in cmake
assert 'FINTEGER=int64_t' in cmake and 'mkl_sequential' in cmake
assert 'REALPATH' in cmake and 'IVF_REFERENCE_BLAS' in cmake
assert 'Hnswlib.cmake' not in (a.repo / 'cmake/IVF.cmake').read_text()
assert not (a.repo / 'cmake/Hnswlib.cmake').exists()
assert not list((a.repo / 'dependencies/hnswlib').rglob('*.h'))
for source in (a.repo / 'src/bufann').glob('ivf_*.cpp'):
    assert '#include <hnswlib/' not in source.read_text()
if a.source:
    for path, expected in HASHES.items():
        assert hashlib.sha256((a.source / path).read_bytes()).hexdigest() == expected, path
print('FAISS 1.9.0 pin, unmodified HNSW kernels and dependency/ABI scope passed')
