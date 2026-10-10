"""Configure routing without Python discovery; build against cached upstream when supplied."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="routing_no_python_", dir=args.scratch) as temporary:
        root = Path(temporary)
        # OFF suites must not fetch upstream merely to exercise configure wiring.
        # ON suites pass the existing pinned source and also compile the router.
        source = args.source or root / "configure_only_upstream"
        if not args.source:
            (source / "hnswlib").mkdir(parents=True)
        tree = root / "build"
        run(
            [
                args.cmake,
                "-S",
                str(args.repo / "tests/ivf"),
                "-B",
                str(tree),
                "-DIVF_BUILD_ROUTING=ON",
                "-DIVF_BUILD_CONTRACT_TESTS=OFF",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE",
                "-DFETCHCONTENT_SOURCE_DIR_IVF_HNSWLIB=" + str(source),
            ]
        )
        if args.source:
            run([args.cmake, "--build", str(tree), "--target", "ivf_router", "-j2"])
        if any(tree.rglob("ivf-hnswlib")):
            raise RuntimeError("routing dependency generated headers")
        cache = (tree / "CMakeCache.txt").read_text()
        if "Python3_EXECUTABLE:" in cache:
            raise RuntimeError("routing dependency discovered Python")
    print(
        "Routing configure without Python passed"
        + ("; router build passed." if args.source else " (no fetch).")
    )


if __name__ == "__main__":
    main()
