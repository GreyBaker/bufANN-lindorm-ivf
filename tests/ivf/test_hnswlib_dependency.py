"""Check vendored IVF ownership and exact, unmodified fetched v0.8.0 headers."""

import argparse
import hashlib
from pathlib import Path


UPSTREAM_SHA256 = {
    "bruteforce.h": "da1f179e99d8e5a4c39b5db0c681bccf3c63e797aaa4982620519f8e001cb2c6",
    "hnswalg.h": "0fb2c1b1d3aae1ea959b536f96844f3f3ae6a2f32ce063f2c31f93d84a7e5c32",
    "hnswlib.h": "54403db81f55fd28246114c5c9f683e663a2a55ad64903b56578c73e6db5ad2c",
    "space_ip.h": "6a89b5275527fe3592a9b02ea4fc50293fdc68cbd9f7c1ec13683581e1ebbaf7",
    "space_l2.h": "c599d024657896412250b4fde1a95e39f8bc55cccd263f36c8948fa35779edf1",
    "stop_condition.h": "930da73eead77dff0c8c9f3281c5d12250051c90ef98fcfbab8f5f307bdae8d9",
    "visited_list_pool.h": "8e5bab8d12b5ef26e3b603b656b8a1acba6738498d1d98f44c9fdb1cb0e25110",
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    local = args.dependency / "include/ivf_patched"
    check({p.name for p in local.iterdir()} == set(UPSTREAM_SHA256), "unexpected local header set")
    check(not (args.dependency / "include/ivf_vanilla").exists(), "comparison must not be vendored")
    check(
        not (args.dependency / "apply.py").exists(), "build-time patch application must be removed"
    )
    for name in UPSTREAM_SHA256:
        text = (local / name).read_text(encoding="utf-8")
        check(
            text.startswith("// Modified from hnswlib v0.8.0:"),
            f"missing modification notice: {name}",
        )
        check(
            "namespace ivf_hnswlib" in text and "namespace hnswlib" not in text,
            f"local namespace separation lost: {name}",
        )
    repo = args.dependency.parents[1]
    module = (repo / "cmake/Hnswlib.cmake").read_text()
    check(
        "execute_process" not in module and "find_package(Python3" not in module,
        "routing dependency must not execute Python or generate headers",
    )
    check(
        "SHA256=cf61d7dd8dc9bfba7f4abe0ed26698e90ac9f49a0badb2e1b0f3ba89b72cf3bb" in module,
        "upstream archive pin changed",
    )
    check(
        "#include <hnswlib/hnswlib.h>" in (repo / "src/bufann/ivf_router.cpp").read_text(),
        "stock comparison must include the fetched upstream header",
    )
    check((args.dependency / "LICENSE").is_file(), "missing upstream license")
    if args.source:
        upstream = args.source / "hnswlib"
        check(
            {p.name for p in upstream.glob("*.h")} == set(UPSTREAM_SHA256),
            "unexpected upstream header set",
        )
        for name, expected in UPSTREAM_SHA256.items():
            check(
                hashlib.sha256((upstream / name).read_bytes()).hexdigest() == expected,
                f"comparison header is not unmodified v0.8.0: {name}",
            )
    print(
        "Local IVF headers checked; "
        + (
            "all seven comparison headers are exact upstream v0.8.0."
            if args.source
            else "routing OFF: no upstream fetch required."
        )
    )


if __name__ == "__main__":
    main()
