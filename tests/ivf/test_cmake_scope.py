"""Configure sibling IVF consumers to check imported-target visibility without fetch."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cmake_scope_", dir=args.scratch) as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(IVFSiblingScope LANGUAGES CXX)\n"
            'set(IVF_BUILD_ROUTING OFF CACHE BOOL "No dependency fetch" FORCE)\n'
            "add_subdirectory(owner)\nadd_subdirectory(consumer)\n"
        )
        for name in ("owner", "consumer"):
            directory = source / name
            directory.mkdir()
            (directory / "probe.cpp").write_text("int probe() { return 0; }\n")
            (directory / "CMakeLists.txt").write_text(
                f'include("{args.module.resolve().as_posix()}")\n'
                "if(NOT TARGET Threads::Threads)\n"
                '  message(FATAL_ERROR "Threads target is not visible in this sibling")\n'
                "endif()\n"
                f"add_library({name}_probe STATIC probe.cpp)\n"
                f"target_link_libraries({name}_probe PRIVATE ivf_contracts Threads::Threads)\n"
            )
        build = root / "build"
        done = subprocess.run(
            [args.cmake, "-S", str(source), "-B", str(build)],
            text=True,
            capture_output=True,
            timeout=30,
        )
        if done.returncode:
            raise RuntimeError(done.stdout + done.stderr)
        if (build / "_deps").exists() or any(build.rglob("ivf-hnswlib")):
            raise RuntimeError("routing-OFF sibling configuration fetched/generated HNSW")
    print("Sibling library/tool scopes see Threads::Threads; routing OFF does not fetch HNSW.")


if __name__ == "__main__":
    main()
