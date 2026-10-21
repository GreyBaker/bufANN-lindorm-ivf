"""Shared local-file validation, evidence I/O and Linux process helpers.

No workload preparation, index policy or calibration selection lives here.
Evidence destinations are exclusive; partial/failed work is never removed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def require_absent(path, message):
    """Reject occupied destinations, including dangling symlinks, without resolving."""
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), message)


def digest(path):
    h = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        while chunk := source.read(8 << 20):
            h.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": h.hexdigest()}


def digests(named_paths):
    """Hash every named path in mapping order, retaining the caller's names."""
    return {name: digest(path) for name, path in named_paths.items()}


def counted_range_identity(
    path, first, count, *,
    framing_message="invalid prepared counted-ID framing",
    count_message="counted-ID count mismatch",
    range_message="prepared update IDs differ from IVF source-row range",
):
    """Validate a counted little-endian contiguous source-row range, then hash it.

    Workload-specific range arithmetic/admission belongs to the caller.
    """
    path = Path(path)
    require(path.stat().st_size == 4 + 4 * count, framing_message)
    with path.open("rb") as source:
        require(source.read(4) == struct.pack("<I", count), count_message)
        for offset in range(0, count, 4096):
            length = min(4096, count - offset)
            require(
                source.read(4 * length)
                == struct.pack("<" + "I" * length, *range(first + offset, first + offset + length)),
                range_message,
            )
    return digest(path)


def is_query_record(record, workload):
    """Recognize native query records; scoring/count/policy checks stay local."""
    return (
        record["baseline"] == "IVF"
        and record["phase"] == "query"
        and record["workload"] == workload
    )


def header(path, dtype, expected_rows, expected_dim):
    size = {"float": 4, "int8": 1, "uint8": 1}[dtype]
    with Path(path).open("rb") as source:
        raw = source.read(8)
    require(len(raw) == 8, f"invalid input header: {path}")
    rows, dim = struct.unpack("<II", raw)
    require((rows, dim) == (expected_rows, expected_dim), f"profile/input shape mismatch: {path}")
    require(Path(path).stat().st_size == 8 + rows * dim * size, f"input length mismatch: {path}")


def query_truth_shapes(query, truth, dtype, dim, recall_at):
    with Path(query).open("rb") as source:
        raw = source.read(8)
    require(len(raw) == 8, "invalid query header")
    rows, cols = struct.unpack("<II", raw)
    require(0 < rows <= 0x7FFFFFFF and cols == dim, "invalid query shape")
    header(query, dtype, rows, dim)
    with Path(truth).open("rb") as source:
        raw = source.read(8)
    require(len(raw) == 8, "invalid GT header")
    gt_rows, gt_cols = struct.unpack("<II", raw)
    require(
        gt_rows == rows and 0 < recall_at <= gt_cols <= 0x7FFFFFFF, "GT/query/recall shape mismatch"
    )
    require(
        Path(truth).stat().st_size in (8 + rows * gt_cols * 8, 8 + rows * gt_cols * 12),
        "invalid GT length",
    )


def read_json(path, max_bytes=65536):
    require(Path(path).stat().st_size <= max_bytes, f"oversized harness metadata: {path}")
    return json.loads(Path(path).read_text())


def write_json(path, record):
    with Path(path).open("x") as out:
        json.dump(record, out, sort_keys=True, indent=2)
        out.write("\n")


def flags(settings):
    return [part for key, value in settings.items() for part in ("--" + key, str(value))]


def command_log(command, path):
    """Stream an untimed prepare/run command to both the console and owned log."""
    with Path(path).open("x") as log:
        log.write("Command: " + repr(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        with process:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        require(process.returncode == 0, f"command failed ({process.returncode}); inspect {path}")


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def finite(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def values(text, convert, minimum):
    result = [convert(part) for part in text.split(",")]
    require(
        result and all(math.isfinite(v) and v >= minimum for v in result), "invalid sweep values"
    )
    require(len(set(result)) == len(result), "duplicate sweep values")
    return result


def affinity(cpus):
    available = sorted(os.sched_getaffinity(0))
    selected = values(cpus, int, 0) if cpus else available
    require(set(selected) <= set(available), "requested CPUs outside available affinity")
    return {"available": available, "selected": selected, "explicitly_pinned": bool(cpus)}


def execute(command, log, timeout, cpus, env=None):
    """Run a measured experiment child, preserving logs and killing its group on timeout."""
    # taskset applies before exec; all descendants inherit the selected affinity.
    if cpus:
        command = [
            "taskset",
            "--cpu-list",
            ",".join(str(v) for v in values(cpus, int, 0)),
        ] + command
    start = time.monotonic()
    with Path(log).open("x") as out:
        out.write("Command: " + repr(command) + "\n")
        out.flush()
        with subprocess.Popen(
            command, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise ValueError(f"command timed out/interrupted; inspect {log}")
    require(code == 0, f"command failed ({code}); inspect {log}")
    return time.monotonic() - start


def runtime_environment():
    return {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "LD_LIBRARY_PATH",
            "ASAN_OPTIONS",
            "UBSAN_OPTIONS",
        )
    }


def shared_module_paths():
    """Include extracted helpers in new experiment provenance, not legacy cache keys."""
    return {
        "python_common": Path(__file__).resolve(),
        "settings": Path(__file__).with_name("settings.py").resolve(),
    }
