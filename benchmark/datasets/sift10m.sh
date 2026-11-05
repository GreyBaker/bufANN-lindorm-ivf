#!/usr/bin/env bash
# Placeholder profile: 10M-vector subset of SIFT-1B.
# Students should create the dataset by sampling from SIFT-1B and configuring this script correctly.

# Identity and benchmark split.
DATASET="sift10m"
DATA_TYPE="uint8"
DIM=128
METRIC="l2"
NPTS_FULL=10000000
NPTS_BASE=9000000
UPDATE_POINTS=1000000
DISKANN_R="${DISKANN_R:-64}"

# Shared inputs default to ann-data; canonical DiskANN indexes remain explicit.
ANN_DATA_ROOT="${ANN_DATA_ROOT:-$HOME/project/ann-data}"
__dataset_root="${SIFT10M_ROOT:-${DATA_ROOT:-$ANN_DATA_ROOT/sift10m}}"
__index_root="${SIFT10M_INDEX_DIR:-/path/to/indexes/sift10m}"

# Override these paths for existing input names/locations.
DATA_BIN="${DATA_BIN:-$__dataset_root/base.bin}"
QUERY_BIN="${QUERY_BIN:-$__dataset_root/query.10K.bin}"

# Alternatively, uncomment these to convert bvecs inputs to the paths above.
# DATA_BVECS="${DATA_BVECS:-$__dataset_root/base.10M.bvecs}"
# QUERY_BVECS="${QUERY_BVECS:-$__dataset_root/query.10K.bvecs}"

# Generated artifact paths; no configuration needed.
GT_ARTIFACT_DIR="${GT_ARTIFACT_DIR:-$__dataset_root/groundtruth}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$GT_ARTIFACT_DIR/sift10m_10000000_gt100.bin}"
PQ_PREFIX="${PQ_PREFIX:-$__dataset_root/pq/pq${PQ_CHUNKS:-32}-base${NPTS_BASE}}"
DISKANN_INDEX_0P9B="${DISKANN_INDEX_0P9B:-$__index_root/sift10m_9M_R64_L100_B32_M8_noPQ}"
