#!/usr/bin/env python3
"""Download an IVF SIFT prefix and public queries; never overwrite or resume."""

import argparse
from pathlib import Path
import struct
import urllib.request


SOURCE_ROWS = 1_000_000_000
DIM = 128
QUERY_ROWS = 10_000
URL_ROOT = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/"


def require_new(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output exists; preserve it and choose a fresh path: {path}")


def download_base(path, rows):
    require_new(path)
    size = 8 + rows * DIM
    request = urllib.request.Request(
        URL_ROOT + f"base.1B.u8bin?subset={rows}",
        headers={"Range": f"bytes=0-{size - 1}", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        expected = f"bytes 0-{size - 1}/{8 + SOURCE_ROWS * DIM}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not return the requested subset range")
        if response.read(8) != struct.pack("<II", SOURCE_ROWS, DIM):
            raise ValueError("Unexpected source header")
        with path.open("xb") as output:
            output.write(struct.pack("<II", rows, DIM))
            remaining = rows * DIM
            while remaining:
                block = response.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Incomplete download; partial file retained")
                output.write(block)
                remaining -= len(block)
    print("Downloaded:", path, flush=True)


def download_queries(path):
    require_new(path)
    size = 8 + QUERY_ROWS * DIM
    with urllib.request.urlopen(URL_ROOT + "query.public.10K.u8bin", timeout=120) as response:
        queries = response.read(size + 1)
    if len(queries) != size or queries[:8] != struct.pack("<II", QUERY_ROWS, DIM):
        raise ValueError("Unexpected query header/length")
    with path.open("xb") as output:
        output.write(queries)
    print("Downloaded:", path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=(10_000_000, 100_000_000), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.base.resolve() == args.query.resolve():
            raise ValueError("Base and query outputs must differ")
        # Refuse either existing output before starting a large download.
        for path in (args.base, args.query):
            require_new(path)
        for path in (args.base, args.query):
            path.parent.mkdir(parents=True, exist_ok=True)
        download_base(args.base, args.rows)
        download_queries(args.query)
    except (ValueError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
