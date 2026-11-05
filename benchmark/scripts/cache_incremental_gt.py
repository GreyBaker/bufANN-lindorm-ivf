#!/usr/bin/env python3
"""Publish immutable derived GT in ann-data and link it into an eval work tree.

Exact deep-GT provenance is the caller's responsibility. Source vectors must be
immutable: their resolved path/stat identify them without rehashing 100M rows on
each prepare. Queries, deep GT, ID lists, tool and outputs are SHA256-checked.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity(path):
    path = path.resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, device=stat.st_dev, inode=stat.st_ino)


def identity(args):
    filters = list(args.filters)
    if filters[:1] == ["--"]:
        filters.pop(0)
    # ID-file paths need not be identical across work trees; their bytes must be.
    normalized = list(filters)
    for index, flag in enumerate(filters):
        if flag in ("--inserts", "--deletes"):
            if index + 1 == len(filters):
                raise ValueError(f"missing file after {flag}")
            normalized[index + 1] = {"sha256": digest(Path(filters[index + 1]))}
    return dict(version=1, source=source_identity(args.source),
                query_sha256=digest(args.query), deep_gt_sha256=digest(args.deep_gt),
                tool_sha256=digest(args.tool), dtype=args.dtype, metric=args.metric,
                k=args.k, filters=normalized), filters


def install(args):
    output = args.output.absolute()
    if os.path.lexists(output):
        raise ValueError(f"preserve existing GT destination: {output}")
    expected, filters = identity(args)
    key = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    root = args.cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key
    if not entry.exists():
        # Only complete directories are published. A failed tool cannot poison
        # later discovery. Concurrent producers may duplicate work, never files.
        with tempfile.TemporaryDirectory(prefix=".partial-", dir=root) as temporary:
            staging = Path(temporary) / key
            staging.mkdir()
            truth = staging / "truth.bin"
            subprocess.run([str(args.tool.resolve()), str(args.deep_gt.resolve()),
                            str(args.k), str(truth), *filters], check=True)
            if not truth.is_file() or truth.is_symlink():
                raise ValueError("incremental GT producer did not create a regular file")
            if identity(args)[0] != expected:
                raise ValueError("GT inputs/tool changed during derivation")
            report = dict(identity=expected, truth_sha256=digest(truth))
            (staging / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
            truth.chmod(0o444)
            try:
                staging.rename(entry)
            except OSError:
                if not entry.is_dir():
                    raise
    truth = entry / "truth.bin"
    if entry.is_symlink() or truth.is_symlink():
        raise ValueError("shared GT cache entry must not be a symlink")
    report = json.loads((entry / "identity.json").read_text())
    if report["identity"] != expected or digest(truth) != report["truth_sha256"]:
        raise ValueError(f"shared GT cache identity/content mismatch: {entry}")
    if identity(args)[0] != expected:
        raise ValueError("GT inputs/tool changed during cache validation")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(truth)
    print(f"[GT] Reused shared truth: {output} -> {truth}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-root", "source", "query", "deep-gt", "tool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("filters", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.k <= 0:
            raise ValueError("GT width must be positive")
        install(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
