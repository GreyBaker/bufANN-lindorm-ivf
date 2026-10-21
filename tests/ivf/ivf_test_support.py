"""Stdlib-only IVF fixture framing and strict native CLI contracts.

Importing this module does not prepare files or configure the process. Distance,
recall, and other production-independent oracles belong in the individual tests.
Writers deliberately do not reject malformed/nonfinite fixture contents.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Optional


def write_matrix(path, rows, fmt, *, width=None):
    """Write an II header followed by typed rows (vectors or code matrices)."""
    if width is None:
        width = len(rows[0])
    values = [value for row in rows for value in row]
    path.write_bytes(struct.pack("<II", len(rows), width)
                     + struct.pack(f"<{len(values)}{fmt}", *values))


def read_matrix(path, fmt):
    """Read a typed matrix, including its exact-length contract."""
    raw = path.read_bytes()
    rows, dim = struct.unpack_from("<II", raw)
    values = struct.unpack_from("<" + fmt * (rows * dim), raw, 8)
    assert len(raw) == 8 + struct.calcsize(fmt) * rows * dim
    return [values[r * dim : (r + 1) * dim] for r in range(rows)]


def write_ids(path, values):
    """Counted IDs are I count + u32 IDs, NOT an II matrix."""
    values = list(values)
    path.write_bytes(struct.pack("<I", len(values))
                     + b"".join(struct.pack("<I", value) for value in values))


def write_truth(path, rows, *, tags=None):
    """GT is II + ID plane + float distance plane + optional u32 tag plane."""
    width = len(rows[0])
    raw = (struct.pack("<II", len(rows), width)
           + b"".join(struct.pack("<I", vid) for row in rows for vid, _ in row)
           + b"".join(struct.pack("<f", distance) for row in rows for _, distance in row))
    if tags is not None:
        raw += b"".join(struct.pack("<I", tag) for row in tags for tag in row)
    path.write_bytes(raw)


def read_results(path):
    """Search results are II + interleaved (u32 ID, float distance) slots."""
    data = path.read_bytes()
    rows, k = struct.unpack_from("<II", data)
    assert len(data) == 8 + rows * k * 8
    entries = list(struct.iter_unpack("<If", data[8:]))
    return [entries[q * k : (q + 1) * k] for q in range(rows)]


def write_pivots(path):
    """The original 3D/two-chunk PQ fixture, with the first block at 4096."""
    dim = 3
    blocks = [
        struct.pack("<II", 256, dim) + struct.pack("<768f", *([0.0] * 768)),
        struct.pack("<II3f", dim, 1, 1.0, 2.0, 3.0),
        struct.pack("<II3I", dim, 1, 0, 1, 2),
        struct.pack("<II3I", 3, 1, 0, 1, 3),
    ]
    offsets = [4096]
    for block in blocks:
        offsets.append(offsets[-1] + len(block))
    with path.open("wb") as out:
        out.write(struct.pack("<II5Q", 5, 1, *offsets))
        out.seek(offsets[0])
        for block in blocks:
            out.write(block)


def run_native_cli(command, phase=None):
    """Native success/progress/timing validation, or unpublished ERROR failure.

    This is intentionally not the validation-only or diagnostics runner.
    """
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)

    def unique_fields(pairs):
        assert len(dict(pairs)) == len(pairs), ("duplicate JSON metric field", pairs)
        return dict(pairs)

    records = [
        json.loads(line, object_pairs_hook=unique_fields)
        for line in result.stdout.splitlines()
        if line.startswith('{"baseline":')
    ]
    if phase is None:
        assert result.returncode != 0 and "ERROR:" in result.stderr, (
            command,
            result.stdout,
            result.stderr,
        )
        assert not records, result.stdout
        return result
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    assert len(records) == 1 and records[0]["phase"] == phase, result.stdout
    assert records[0]["baseline"] == "IVF"
    if phase == "build":
        phases = [line.split("phase=", 1)[1].split()[0]
                  for line in result.stderr.splitlines() if line.startswith("[IVF build]")]
        assert phases == ["input_identity", "training", "final_router", "posting_lists",
                           "coordinate_heap", "heap_checkpoint", "artifact_verification", "complete"], phases
    else:
        progress = [dict(item.split("=", 1) for item in line.split()[2:])
                    for line in result.stderr.splitlines() if line.startswith("[IVF driver]")]
        assert progress and progress[0]["phase"] == "open_private_copy_verify", result.stderr
        assert not any(p["event"] in ("failed", "abandoned", "incomplete") for p in progress), result.stderr
        if phase == "query":
            record = records[0]
            assert record["fetch_adapter_mutex_wait_us_total"] >= 0
            assert record["fetch_backend_pin_us_total"] >= 0
            assert record["fetch_pin_admission_wait_us_total"] >= 0
            assert (record["fetch_adapter_mutex_wait_us_total"] + record["fetch_backend_pin_us_total"]
                    + record["fetch_pin_admission_wait_us_total"]
                    <= record["fetch_us_total"] * 1.01 + 1.0)
            assert "bounds_reads_and_writes" in record["fetch_pin_admission_policy"]
            assert "not_additional_stages" in record["fetch_substage_timing_scope"]
        if phase in ("insert", "delete"):
            for name, count in (("insert", records[0]["insert_count"]),
                                ("delete", records[0]["delete_count"] + records[0]["missing_delete_count"])):
                finished = [p for p in progress if p["phase"] == name and p["event"] == "complete"]
                assert len(finished) == 1 and int(finished[0]["completed"]) == count, progress
            assert records[0]["foreground_heap_flush_bytes"] == records[0]["foreground_heap_page_writes"] * 4096
            assert records[0]["checkpoint_heap_flush_bytes"] == records[0]["checkpoint_heap_page_writes"] * 4096
            assert "not_device_io" in records[0]["heap_io_counter_scope"]
    return records[0]


def reserve_attempt(parent):
    """Reserve a fresh run directory without overwriting earlier evidence."""
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 10001):
        path = parent / f"run-{n:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError("parallel insertion fixture attempts exhausted")


@dataclass
class TinyPopulation:
    work: Path
    vectors: list
    queries: list
    data: Path
    query: Path
    prefix: Path
    build: list
    source_hash: str
    originals: dict
    insert_ids: Path
    delete_ids: Path
    empty_ids: Optional[Path]


def build_tiny_population(build_executable, root, dtype, fmt, *,
                          reverse_insert_ids, include_empty_ids, run_build):
    """Build the shared 64-row/3D, eight-base/eight-query fixture.

    Insertion order and empty-ID creation are explicit: parallel/policy tests
    reverse the tail and need an empty file; warmup uses ascending IDs and none.
    Keep the build command for policy's deliberately nonfinite unindexed tail.
    The injected strict runner also lets unit tests check parity without native IO.
    """
    work = root / dtype
    work.mkdir()
    vectors = [tuple((i * 7 + d * 3) % 31 - (16 if dtype == "int8" else 0)
                     for d in range(3)) for i in range(64)]
    queries = vectors[::8]
    data, query, prefix = work / "source.bin", work / "queries.bin", work / "base"
    write_matrix(data, vectors, fmt)
    write_matrix(query, queries, fmt)
    pq = work / "pq"
    write_pivots(Path(str(pq) + "_pq_pivots.bin"))
    write_matrix(Path(str(pq) + "_pq_compressed.bin"), [bytes(2)] * 64, "B")
    build = [build_executable, "--validate_inputs", "0", "--data_type", dtype,
             "--data_file", str(data), "--pq_prefix", str(pq), "--index_prefix", str(prefix),
             "--dim", "3", "--base_count", "8", "--nlist", "2", "--max_reps", "2",
             "--build_threads", "2", "--query_breadth", "2", "--insert_breadth", "2"]
    run_build(build, "build")
    source_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in work.glob("base.*")}
    insert_ids, delete_ids = work / "insert.ids", work / "delete.ids"
    tail = range(8, 64)
    write_ids(insert_ids, reversed(tail) if reverse_insert_ids else tail)
    write_ids(delete_ids, [0])
    empty = work / "empty.ids" if include_empty_ids else None
    if empty is not None:
        write_ids(empty, [])
    return TinyPopulation(work, vectors, queries, data, query, prefix, build,
                          source_hash, originals, insert_ids, delete_ids, empty)
