#!/usr/bin/env python3
"""Return a coverage-checked GT path; discover, recommend and optionally generate.

Stdout contains only the usable absolute path on success. Progress goes to stderr.
Existing exact-cache provenance remains the caller's responsibility. Offline uint8
squared L2 only; no wrappers, indexes or benchmark timing code are invoked.
"""
from argparse import Namespace
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import check_gt_coverage as coverage
import recommend_gt_depth as depth


class NeedsGeneration(ValueError):
    """Valid inputs but no covered cache; explicit generation consent needed."""


def log(message):
    print(f"[GT] {message}", file=sys.stderr, flush=True)


def save(path, report):
    # Publish complete records without overwriting. An interrupted write remains
    # .partial evidence and cannot poison discovery on the next invocation.
    partial = path.with_name(path.name + ".partial")
    with partial.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    os.link(partial, path)
    partial.unlink()


def new_attempt(directory):
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        attempt = directory / f"run-{number:04d}"
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            number += 1


def candidates(args, inputs):
    """Explicit selection first, successful managed runs next, legacy caches last."""
    if args.cached_gt:
        if args.cached_gt.is_file():
            yield args.cached_gt
        else:
            log(f"Ignoring missing cache: {args.cached_gt}")
    for plan in sorted((args.cache_dir / ".gt").glob("run-*/plan.json"), reverse=True):
        with plan.open() as file:
            previous = json.load(file)
        if previous["inputs"] != inputs:
            continue
        result = plan.parent / "result.json"
        if result.is_file():
            with result.open() as file:
                yield Path(json.load(file)["selected_gt"])
        # Completed, structurally checked generation may be reused as a seed
        # after a later recommendation/admission failure. Partial GT never is.
        for checkpoint in sorted(plan.parent.glob("generated-k*.json")):
            with checkpoint.open() as file:
                yield Path(json.load(file)["cached_gt"])
    # Filenames guide preference only; headers/content determine actual coverage.
    def width(path):
        match = re.fullmatch(r"full-query\.gt(\d+)\.bin", path.name)
        return int(match[1]) if match else -1
    yield from sorted(args.cache_dir.glob("full-query.gt*.bin"), key=lambda p: (width(p), p.name), reverse=True)


def generate(args, attempt, k):
    coverage.require(k <= args.full_count and coverage.fits_row_scratch(k),
                     "GT depth exceeds preflight admission; consider direct per-state GT")
    coverage.require(args.gt_generator is not None and args.gt_generator.is_file(),
                     "GT generator not found; set IVF_BUILD_DIR or pass --gt-generator")
    output = attempt / f"full-query.gt{k}.bin"
    coverage.require(not os.path.lexists(output), "refusing to overwrite GT output")
    command = [str(args.gt_generator.resolve()), "--data_type", "uint8",
               "--base_file", str(args.base_file.resolve()), "--query_file", str(args.query_file.resolve()),
               "--gt_file", str(output), "--K", str(k), "--num_threads", str(args.num_threads)]
    save(attempt / f"generate-k{k}.json", {"command": command})
    log(f"Generating full-source K={k} cache (may take substantial time): {output}")
    # Native helper output must not leak into the command-substitution path.
    with (attempt / f"generate-k{k}.log").open("x") as file:
        subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, check=True)
    coverage.require(output.is_file() and not output.is_symlink(), "generator did not produce a regular GT file")
    return output


def ensure(args, attempt):
    states = coverage.planned_states(args)
    with depth.Snapshot(args.base_file) as source, depth.Snapshot(args.query_file) as queries:
        rows, dim = depth.vector_shape(source)
        query_rows, query_dim = depth.vector_shape(queries)
        coverage.require(rows == args.full_count, "full source row count differs from plan")
        coverage.require(dim == query_dim, "source/query dimensions differ")
        coverage.require(args.expected_queries is None or query_rows == args.expected_queries,
                         "query count differs from plan")
        inputs = {"source": str(source.resolved), "source_stat": list(source.path_signature),
                  "source_rows": rows, "dim": dim, "query": str(queries.resolved),
                  "query_rows": query_rows, "query_sha256": coverage.sha256(queries.file)}
        save(attempt / "plan.json", {"inputs": inputs, "states": [asdict(s) for s in states],
                                    "k_out": args.k_out, "generate_allowed": args.generate})
        checks = []

        def check(path):
            log(f"Checking {path}")
            report = coverage.check_cache(path, rows, states, args.k_out, query_rows, args.worst_queries)
            coverage.require(report["cached_k"] <= rows, "cached depth exceeds source population")
            save(attempt / f"coverage-{len(checks) + 1:02d}.json", report)
            checks.append(report)
            if not report["coverage_passed"]:
                log(f"K={report['cached_k']} underflows; looking for a covered cache")
            return report

        def finish(report):
            source.check()
            queries.check()
            coverage.require(coverage.sha256(queries.file) == inputs["query_sha256"],
                             "queries changed during GT preparation")
            with depth.Snapshot(report["cached_gt"]) as gt:
                coverage.require(coverage.sha256(gt.file) == report["cached_gt_sha256"],
                                 "selected cache changed after coverage check")
                gt.check()
            save(attempt / "result.json", {"inputs": inputs, "selected_gt": report["cached_gt"],
                                          "coverage": report, "exact_cache_provenance_assumed": True})
            log(f"Ready: K={report['cached_k']}; evidence: {attempt}")
            return report["cached_gt"]

        def generated(k):
            report = check(generate(args, attempt, k))
            source.check()
            queries.check()
            save(attempt / f"generated-k{k}.json", report)
            return report

        seen, seed = set(), None
        for path in candidates(args, inputs):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            report = check(path)
            if report["coverage_passed"]:
                return finish(report)
            if seed is None or report["cached_k"] > seed["cached_k"]:
                seed = report

        if seed is None:
            if not args.generate:
                raise NeedsGeneration("No existing cache. Rerun with --generate to create and check full-source GT")
            initial_k = min(rows, max(args.initial_k, args.k_out))
            seed = generated(initial_k)
            if seed["coverage_passed"]:
                return finish(seed)

        recommendation_args = Namespace(**vars(args))
        recommendation_args.cached_gt = Path(seed["cached_gt"])
        recommendation_args.expected_queries = query_rows
        recommendation = depth.recommend(recommendation_args)
        save(attempt / "recommendation.json", recommendation)
        k = recommendation["recommended_k"]
        log(f"Recommended K={k}; tagged bytes={recommendation['tagged_cache_bytes']}")
        coverage.require(recommendation["fits_coverage_row_scratch_guard"],
                         "recommended depth exceeds preflight admission; consider direct per-state GT")
        if not args.generate:
            raise NeedsGeneration(f"Cache underflows; recommended K={k}. Rerun with --generate to generate and check it")
        # Exactly one recommended repair, not an unbounded auto-retry loop.
        coverage.require(k > seed["cached_k"], "recommendation did not increase underflowing depth")
        report = generated(k)
        coverage.require(report["coverage_passed"], "new GT still underflows; preserve evidence and investigate")
        return finish(report)


def parser():
    p = depth.parser(cached_required=False, report_option=False)
    p.description = __doc__
    p.set_defaults(cached_gt=Path(os.environ["GT_BIN_KCACHED"]) if os.environ.get("GT_BIN_KCACHED") else None)
    p.add_argument("--cache-dir", type=Path, help="dataset/cache directory; default: base-file directory")
    p.add_argument("--generate", action="store_true", help="allow expensive exact GT generation only if needed")
    default_generator = (Path(os.environ["IVF_BUILD_DIR"]) / "tests/bufann_gt_computer"
                         if os.environ.get("IVF_BUILD_DIR") else None)
    p.add_argument("--gt-generator", type=Path, default=default_generator)
    p.add_argument("--initial-k", type=coverage.positive, default=1000, help="starting depth only if no cache exists")
    p.add_argument("--num-threads", type=coverage.positive, default=8, help="exact generator thread count")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    attempt = None
    try:
        args.cache_dir = (args.cache_dir or args.base_file.parent).resolve()
        coverage.planned_states(args)
        attempt = new_attempt(args.cache_dir / ".gt")
        log(f"Evidence: {attempt}")
        selected = ensure(args, attempt)
        print(selected)
        return 0
    except (OSError, ValueError, OverflowError, MemoryError, subprocess.CalledProcessError) as error:
        if attempt is not None:
            save(attempt / "error.json", {"error": str(error)})
        log(f"ERROR: {error}")
        return 3 if isinstance(error, NeedsGeneration) else 1


if __name__ == "__main__":
    sys.exit(main())
