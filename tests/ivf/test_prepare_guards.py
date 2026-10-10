"""Shell admission allows manual/report-only selection without validating reports."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    env = {key: value for key, value in os.environ.items() if not key.startswith("IVF_")}
    env.update(
        REPO_DIR=str(args.repo),
        DATASET="fixture",
        EVAL_TEMPFILES="/unused",
        IVF_BUILD_BIN="/bin/true",
        IVF_DRIVER_BIN="/bin/true",
    )
    command = [
        "bash",
        "-c",
        'set -euo pipefail; source "$REPO_DIR/benchmark/IVF/common.sh"; ivf_require_build_selection',
    ]
    for extra, success in (
        ({}, False),
        ({"IVF_NLIST": "2"}, True),
        ({"IVF_COST_MODEL_REPORT": "/python-validates-this.json"}, True),
        ({"IVF_NLIST": "2", "IVF_COST_MODEL_REPORT": "/python-validates-this.json"}, True),
    ):
        result = subprocess.run(
            command, env=dict(env, **extra), text=True, capture_output=True, timeout=10
        )
        if (result.returncode == 0) != success:
            raise RuntimeError(result.stdout + result.stderr)
    for name in ("query_prepare.sh", "_mutation_prepare.sh", "update_prepare.sh"):
        text = (args.repo / "benchmark/IVF/sift" / name).read_text()
        if "ivf_require_build_selection" not in text or "${IVF_NLIST:?" in text:
            raise RuntimeError("Preparation bypasses shared selection guard: " + name)
    print("Shared query/mutation/mixed manual and report-only shell guards passed.")


if __name__ == "__main__":
    main()
