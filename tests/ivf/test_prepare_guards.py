"""IVF-only shell ownership, output selection and prefix-truth contracts."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    env = {key: value for key, value in os.environ.items() if not key.startswith("IVF_")}
    env.update(
        REPO_DIR=str(args.repo),
        DATASET="fixture",
        WORKLOAD="query",
        EVAL_TEMPFILES="/unused",
        IVF_BUILD_BIN="/bin/true",
        IVF_DRIVER_BIN="/bin/true",
    )
    def invoke(body, extra=None, success=True):
        command = ["bash", "-c",
                   'set -euo pipefail; source "$REPO_DIR/benchmark/IVF/common.sh"; ' + body]
        result = subprocess.run(command, env=dict(env, **(extra or {})), text=True,
                                capture_output=True, timeout=10)
        if (result.returncode == 0) != success:
            raise RuntimeError(result.stdout + result.stderr)
        return result

    for extra, success in (
        ({}, False),
        ({"IVF_NLIST": "2"}, True),
        ({"IVF_COST_MODEL_REPORT": "/python-validates-this.json"}, True),
        ({"IVF_NLIST": "2", "IVF_COST_MODEL_REPORT": "/python-validates-this.json"}, True),
    ):
        invoke("ivf_require_build_selection", extra, success)
    for name in ("query_prepare.sh", "_mutation_prepare.sh", "update_prepare.sh"):
        text = (args.repo / "benchmark/IVF/sift" / name).read_text()
        if "ivf_require_build_selection" not in text or "${IVF_NLIST:?" in text:
            raise RuntimeError("Preparation bypasses shared selection guard: " + name)
    for tag, suffix in (("", ""), ("run-01_A", "/runs/run-01_A")):
        result = invoke('ivf_select_output_dir; printf "%s" "$OUTPUT_DIR"', {"IVF_RUN_TAG": tag})
        assert result.stdout == "/unused/fixture/IVF/query" + suffix
    for tag in (".", "../escape", "a/b", "has space"):
        invoke("ivf_select_output_dir", {"IVF_RUN_TAG": tag}, success=False)

    with tempfile.TemporaryDirectory(prefix="ivf shell guards ") as temporary:
        root = Path(temporary)
        data, queries, truth = (root / name for name in ("full.bin", "queries.bin", "truth.bin"))
        for path in (data, queries, truth):
            path.write_bytes(b"unchanged source fixture")
        env.update(DATA_BIN=str(data), QUERY_BIN=str(queries), EVAL_CACHED=str(root / "cache"))
        # Shared input storage is not an allowed destructive eval destination.
        invoke('safe_mutable_path "$ANN_DATA_ROOT/truth.bin"',
               {"ANN_DATA_ROOT": str(root / "ann-data")}, success=False)
        fresh = root / "fresh"
        invoke("ivf_prepare_work_dir", {"EVAL_TEMPFILES": str(fresh)})
        work = fresh / "fixture/IVF/query"
        assert (work / "data/full.bin").resolve() == data
        assert (work / "data/query.bin").resolve() == queries
        retained = work / "failure-evidence"
        retained.write_bytes(b"retain partial work")
        invoke("ivf_prepare_work_dir", {"EVAL_TEMPFILES": str(fresh)}, success=False)
        assert retained.read_bytes() == b"retain partial work"
        for kind in ("file", "directory", "live-link", "dangling-link"):
            temp = root / kind
            collision = temp / "fixture/IVF/query"
            collision.parent.mkdir(parents=True)
            if kind == "file":
                collision.write_bytes(b"retain existing file")
            elif kind == "directory":
                collision.mkdir()
                (collision / "keep").write_bytes(b"retain existing directory")
            else:
                collision.symlink_to(data if kind == "live-link" else root / "missing")
            invoke("ivf_prepare_work_dir", {"EVAL_TEMPFILES": str(temp)}, success=False)
            if kind == "file":
                assert collision.read_bytes() == b"retain existing file"
            elif kind == "directory":
                assert (collision / "keep").read_bytes() == b"retain existing directory"
            else:
                assert collision.is_symlink()

        destination = work / "groundtruth/supplied.bin"
        invoke('ivf_stage_prefix_truth "$DESTINATION" 10 90',
               {"EVAL_TEMPFILES": str(fresh), "IVF_GT_FILE": str(truth), "DESTINATION": str(destination)})
        assert destination.is_symlink() and destination.resolve() == truth
        missing = work / "groundtruth/missing.bin"
        invoke('ivf_stage_prefix_truth "$DESTINATION" 10 90',
               {"IVF_GT_FILE": str(root / "absent-truth"), "DESTINATION": str(missing)}, success=False)
        assert not missing.exists() and not missing.is_symlink()
        # A stub checks forwarding only; do not run a generator or baseline scorer.
        marker = root / "derive-arguments"
        derived = work / "groundtruth/derived.bin"
        invoke('resolve_incremental_gt() { printf "%s\\n" "$@" > "$MARKER"; }; '
               'ivf_stage_prefix_truth "$DESTINATION" 10 80',
               {"IVF_GT_FILE": "", "DESTINATION": str(derived), "MARKER": str(marker)})
        assert marker.read_text().splitlines() == [str(derived), "10", "--base_trunc", "80"]
        assert all(path.read_bytes() == b"unchanged source fixture" for path in (data, queries, truth))

    # Producers and consumers for every runner use dataset/runner/workload.
    for runner in ("BufANN", "DiskANN", "Greator", "OdinANN", "SPFresh"):
        for workload in ("query", "insert", "delete", "update"):
            for suffix in (".sh", "_prepare.sh"):
                text = (args.repo / f"benchmark/{runner}/sift/{workload}{suffix}").read_text()
                assert '/$DATASET/$BASELINE/$WORKLOAD' in text
                assert '/${DATASET}-${WORKLOAD}-${BASELINE}"' not in text
    # New profile defaults put both kinds of reusable artifact under ann-data.
    result = subprocess.run(
        ["bash", "-euc", 'source "$REPO_DIR/benchmark/datasets/sift100m.sh"; '
         'printf "%s\\n" "$GT_BIN_KCACHED" "$PQ_PREFIX"'],
        env={"PATH": os.environ["PATH"], "HOME": "/fixture-home", "REPO_DIR": str(args.repo),
             "ANN_DATA_ROOT": "/shared/ann-data"}, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "/shared/ann-data/sift100m/groundtruth/sift100m_100000000_gt100.bin",
        "/shared/ann-data/sift100m/pq/pq32-base90000000"]
    print("IVF shell selection, no-overwrite ownership, run tags and prefix-truth guards passed.")


if __name__ == "__main__":
    main()
