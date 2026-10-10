"""IVF build settings and frozen calibration-selection contracts.

Keep environment/CLI defaults separate from the native library's legacy defaults.
Legacy cache-key fields and historical report interpretations are intentional.
"""
import json
import math
import os
from pathlib import Path

from common import digest, require


BUILD_DEFAULTS = {
    "nlist": None, "sample_rate": "1", "max_reps": "15", "seed": "0", "build_threads": "1",
    "memory_budget_gb": "1", "page_size": "4096", "buffer_frames": "64", "router_degree": "16",
    "construction_breadth": "200", "query_breadth": "128", "insert_breadth": "64", "router_seed": "100",
    "optimized_modes": "0", "refine_ef": "128", "pca_variance": "0.95", "routing_work_limit": "1000000000",
    "trainer_mode": "ann-reuse-seeded", "init": "random_partition", "training_breadth": "64",
}

# Calibration has its own sampling, scan-profile and work allowances. Preserve
# its existing spellings/values rather than normalizing historical plan settings.
CALIBRATION_DEFAULTS = {
    "data_type": "float", "dim": None, "base_count": None, "sample_count": None,
    "nlist_grid": None, "synthetic_queries": "100", "topk": "10", "target_recall": ".90", "seed": "0",
    "max_reps": "15", "build_threads": "1", "memory_budget_gb": "1", "trainer_mode": "ann-reuse-seeded",
    "init": "random_partition", "training_breadth": "64", "router_mode": "hnswlib", "optimized_modes": "0",
    "router_degree": "16", "construction_breadth": "200", "query_breadth": "128", "insert_breadth": "64",
    "router_seed": "100", "refine_ef": "128", "pca_variance": ".95", "router_rerank_factor": "3",
    "routing_work_limit": "1000000000", "profile_codes": "100000", "profile_repeats": "3",
    "routing_repeats": "3", "rerank_factor": "4", "work_limit": "1000000000",
    "min_sample_per_centroid": "32", "training_rerank_factor": "3", "training_refine_breadth": "128",
}


def query_router_flags():
    return [part for key in ("router_mode", "router_rerank_factor", "router_breadth")
            if (value := os.environ.get("IVF_" + key.upper())) is not None
            for part in ("--" + key, value)]


def read_cost_model_report(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 << 20,
            "missing/symlink/oversized cost-model report")

    def unique(pairs):
        require(len(dict(pairs)) == len(pairs), "duplicate cost-model report field")
        return dict(pairs)

    record = json.loads(path.read_text(), object_pairs_hook=unique)
    require(record.get("version") in (1, 2) and record.get("kind") == "ivf_cost_model" and record.get("completed") is True,
            "incomplete/unsupported cost-model report")
    chosen = record.get("selected_nlist")
    require(type(chosen) is int and chosen > 0, "invalid model-selected nlist")
    minimum = 32 if record["version"] == 1 else record.get("min_sample_per_centroid")
    require(type(minimum) is int and 1 <= minimum <= 0x7fffffff, "invalid declared sample adequacy minimum")
    if record["version"] == 2:
        require(record.get("candidate_eligibility_policy") == "declared_project_minimum_not_paper_requirement",
                "unsupported candidate eligibility policy")
        mode = record.get("trainer_mode")
        legacy_modes = ("exact", "ann-rebuild", "ann-reuse", "ann-reuse-seeded")
        count_modes = ("ann-reuse-count", "ann-reuse-seeded-count")
        require(mode in legacy_modes + count_modes + ("lv", "lv-plus"), "unsupported report trainer")
        expected = ("pca95_refit_each_pass_initial_refinement" if mode in ("lv", "lv-plus") else
                    "full_count_based" if mode in count_modes else "exact_full_assignment" if mode == "exact" else
                    "legacy_full_breadth_rerank")
        require(record.get("training_search_policy") == expected, "inconsistent training search policy")
    require(type(record["sample_count"]) is int and record["sample_count"] > 0, "invalid calibration sample count")
    for candidate in record["candidates"]:
        n = candidate["nlist"]
        require(type(n) is int and 0 < n <= 0x7fffffff and type(candidate["eligible"]) is bool and
                candidate["eligible"] == (n * minimum <= record["sample_count"]), "incorrect candidate eligibility")
    candidates = [candidate for candidate in record["candidates"] if candidate["eligible"]]
    require(candidates and len({c["nlist"] for c in record["candidates"]}) == len(record["candidates"]),
            "empty/duplicate cost-model candidates")
    for candidate in candidates:
        require(candidate["nlist"] * minimum <= record["sample_count"], "ineligible model candidate")
        a, b = candidate["t1_us"], candidate["t2_us"]
        require(math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0 and
                candidate["total_us"] == a + b, "invalid cost-model candidate estimate")
    require(chosen == min(candidates, key=lambda c: (c["total_us"], c["nlist"]))["nlist"],
            "cost-model selection is not minimum T1+T2")
    return record


def nlist_selection(args, settings, sources):
    path = os.environ.get("IVF_COST_MODEL_REPORT")
    if not path:
        return {"source": "manual", "nlist": int(settings["nlist"])}
    record = read_cost_model_report(path)
    require(record["data_type"] == args.dtype and record["dim"] == args.dim and
            record["source_rows"] == args.full_count and record["base_count"] == args.base_count,
            "cost-model profile does not match build")
    require([record["identities"][key] for key in ("data", "pq_pivots", "pq_codes")] == sources,
            "cost-model source/PQ identity does not match build")
    require(record["objective"] == "mean_T1_plus_T2_excludes_fetch_and_exact_rerank" and
            record["calibration_query_policy"] == "disjoint_base_rows_no_official_queries",
            "unsupported cost-model objective/calibration policy")
    aliases = {"init": "trainer_init", "sample_rate": "target_sample_rate", "nlist": "selected_nlist"}
    keys = ("nlist", "sample_rate", "max_reps", "seed", "build_threads", "trainer_mode", "init", "training_breadth",
            "memory_budget_gb", "router_mode", "optimized_modes", "router_degree", "construction_breadth",
            "query_breadth", "insert_breadth", "router_seed", "refine_ef", "pca_variance",
            "router_rerank_factor", "routing_work_limit")
    for key in keys:
        observed = record[aliases.get(key, key)]
        same = settings[key] == str(observed) if isinstance(observed, str) else float(settings[key]) == observed
        require(same, "cost-model/build setting mismatch: " + key)
    for key, default in (("training_rerank_factor", 3), ("training_refine_breadth", 128)):
        require(float(settings.get(key, default)) == record.get(key, default), "cost-model/build setting mismatch: " + key)
    selection = {"source": "cost_model", "nlist": record["selected_nlist"], "report_path": str(Path(path).resolve()),
                 "report_identity": digest(path), "objective": record["objective"],
                 "topk": record["topk"], "target_recall": record["target_recall"], "rerank_factor": record["rerank_factor"],
                 "validation_completed": record.get("validation_completed", False), "production_qualified": False}
    if record["version"] == 2:
        selection.update(min_sample_per_centroid=record["min_sample_per_centroid"],
                         candidate_eligibility_policy=record["candidate_eligibility_policy"])
    return selection


def prepared_selection(prepared):
    selection = prepared.get("nlist_selection", {"source": "manual", "nlist": int(prepared["settings"]["nlist"])})
    if selection["source"] == "cost_model":
        require(digest(selection["report_path"]) == selection["report_identity"], "prepared cost-model report changed")
    return selection


def profile(args):
    return {key: getattr(args, key) for key in ("dataset", "dtype", "metric", "dim", "full_count", "base_count")}


def build_settings():
    result = {key: os.environ.get("IVF_" + key.upper(), value) for key, value in BUILD_DEFAULTS.items()}
    # Preserve legacy cache keys unless the new trainer/settings are requested.
    for key, default in (("training_rerank_factor", "3"), ("training_refine_breadth", "128")):
        value = os.environ.get("IVF_" + key.upper())
        if result["trainer_mode"] in ("lv", "lv-plus") or value is not None:
            result[key] = default if value is None else value
    # Query-only mode/factor overrides do not alter the build/cache identity.
    result["router_mode"] = os.environ.get("IVF_ASSIGNMENT_ROUTER_MODE", "hnswlib")
    result["router_rerank_factor"] = os.environ.get("IVF_BUILD_ROUTER_RERANK_FACTOR", "3")
    if os.environ.get("IVF_COST_MODEL_REPORT"):
        chosen = read_cost_model_report(os.environ["IVF_COST_MODEL_REPORT"])["selected_nlist"]
        require(result["nlist"] is None or int(result["nlist"]) == chosen, "IVF_NLIST disagrees with cost-model selection")
        result["nlist"] = str(chosen)
    require(result["nlist"] is not None, "set IVF_NLIST explicitly or provide IVF_COST_MODEL_REPORT")
    return result
