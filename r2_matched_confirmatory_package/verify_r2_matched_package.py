"""
r2_matched_confirmatory_package/verify_r2_matched_package.py:
Portable verification harness for the R2_matched confirmatory reproduction package:
  - Job 16911: robust_r2_matched_confirmatory (cost_s = 0.0, N=24 seeds [4001..4024])
  - Job 16914: robust_r2_matched_cost_confirmatory (cost_s = 0.0010 / 10 bps, N=24 seeds [5001..5024])

Explicitly separates 3 reproducibility levels:
  Level 1: SHA-256 integrity, CSV/JSON schema, checkpoint inventory, Berger IUT & sensitivity stats.
  Level 2: Checkpoint reload & JAX re-evaluation against recorded CSV/JSON (runs only if JAX works).
  Level 3: Full re-training from scratch (NOT_RUN_IN_VERIFICATION; prints exact Slurm commands).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PKG_DIR = Path(__file__).resolve().parent
EXP_ROOT = PKG_DIR.parent
SNAP_DIR = PKG_DIR / "deps_snapshot"
C0_DIR = EXP_ROOT / "robust_r2_matched_confirmatory"
C10_DIR = EXP_ROOT / "robust_r2_matched_cost_confirmatory"


def _setup_snapshot_sys_path() -> None:
    """Prefer packaged deps_snapshot modules so standalone clones verify without external folders."""
    ordered = [
        PKG_DIR,
        C0_DIR,
        C10_DIR,
        SNAP_DIR,
        SNAP_DIR / "robust_r3_lr_stability",
        SNAP_DIR / "robust_r3_selection_diagnostics",
        SNAP_DIR / "robust_r3_budget_control",
        SNAP_DIR / "robust_r3_dev",
        SNAP_DIR / "scripts" / "robust_deep_hedging_phase2",
    ]
    for p in reversed(ordered):
        sp = str(p)
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256sums_file(sum_file: Path, base_dir: Path) -> Dict[str, Any]:
    checked = 0
    mismatches: List[str] = []
    missing: List[str] = []
    for raw in sum_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_str = line.split(None, 1)
        rel_clean = rel_str.lstrip("*").strip()
        target = (base_dir / rel_clean).resolve()
        if not target.exists():
            missing.append(rel_clean)
            continue
        actual = sha256_file(target)
        if actual != expected_hash.strip():
            mismatches.append(f"{rel_clean}: expected={expected_hash}, actual={actual}")
        checked += 1
    return {
        "sum_file": str(sum_file.relative_to(EXP_ROOT)).replace("\\", "/"),
        "checked": checked,
        "missing": missing,
        "mismatches": mismatches,
        "passed": len(missing) == 0 and len(mismatches) == 0 and checked > 0,
    }


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_level1_integrity_and_stats() -> Dict[str, Any]:
    _setup_snapshot_sys_path()
    from conf_stats import analyze_confirmatory_pairs, paired_t_summary
    from cost_stats import analyze_cost_confirmatory_pairs

    sha_report: Dict[str, Any] = {}
    top_sums = PKG_DIR / "SHA256SUMS.txt"
    if top_sums.exists():
        sha_report["package_SHA256SUMS"] = verify_sha256sums_file(top_sums, EXP_ROOT)
    for sub in ("robust_r2_matched_confirmatory", "robust_r2_matched_cost_confirmatory"):
        pf = EXP_ROOT / sub / "POSTRUN_SHA256SUMS.txt"
        if pf.exists():
            sha_report[f"{sub}_POSTRUN_SHA256SUMS"] = verify_sha256sums_file(pf, EXP_ROOT / sub)

    exp_checks: Dict[str, Any] = {}
    specs = [
        (
            "Job_16911_0bps",
            C0_DIR,
            list(range(4001, 4025)),
            list(range(4025, 4041)),
            "RESULTS_CONFIRMATORY_FULL_PRECISION.csv",
            "PAIRED_CONFIRMATORY.csv",
            "CONFIRMATORY_STATS_SUMMARY.json",
            False,
        ),
        (
            "Job_16914_10bps",
            C10_DIR,
            list(range(5001, 5025)),
            list(range(5025, 5041)),
            "RESULTS_COST_CONFIRMATORY_FULL_PRECISION.csv",
            "PAIRED_COST_CONFIRMATORY.csv",
            "COST_CONFIRMATORY_STATS_SUMMARY.json",
            True,
        ),
    ]

    for label, d, seeds, unused_seeds, ev_fn, pr_fn, sm_fn, is_cost in specs:
        res_dir = d / "results" / "confirmatory_n24"
        ckpt_dir = d / "checkpoints" / "confirmatory_n24"
        evals = _load_csv(res_dir / ev_fn)
        paired = _load_csv(res_dir / pr_fn)
        summary = json.loads((res_dir / sm_fn).read_text(encoding="utf-8"))
        ledger = json.loads((res_dir / "SEED_LEDGER.json").read_text(encoding="utf-8"))

        assert len(evals) == 576, f"{label}: expected 576 eval rows, got {len(evals)}"
        assert len(paired) == 288, f"{label}: expected 288 paired rows, got {len(paired)}"
        assert len(ledger["seed_records"]) == 24
        assert ledger["failed_seeds"] == []

        best_pkls = list(ckpt_dir.glob("seed_*/*/best_val.pkl"))
        final_pkls = list(ckpt_dir.glob("seed_*/*/final_with_opt.pkl"))
        assert len(best_pkls) == 48, f"{label}: expected 48 best_val.pkl, got {len(best_pkls)}"
        assert len(final_pkls) == 48, f"{label}: expected 48 final_with_opt.pkl, got {len(final_pkls)}"
        for u in unused_seeds:
            assert not (ckpt_dir / f"seed_{u}").exists(), f"{label}: unused seed {u} leaked into checkpoints!"

        recomp = (
            analyze_cost_confirmatory_pairs(paired, planned_seeds=seeds)
            if is_cost
            else analyze_confirmatory_pairs(paired, planned_seeds=seeds)
        )
        assert recomp["verdict"] == "PASS_CONFIRMATORY_IUT"
        assert recomp["verdict"] == summary["stats_summary"]["verdict"]
        assert abs(recomp["p_iut_max"] - summary["stats_summary"]["p_iut_max"]) < 1e-12

        extra: Dict[str, Any] = {}
        if is_cost:
            max_resid = max(float(r["cost_accounting_residual"]) for r in evals)
            min_cost = min(float(r["mean_cost"]) for r in evals)
            assert max_resid < 1e-5
            assert min_cost > 0.0
            p_idx = {(int(r["seed"]), r["ckpt_kind"], r["scenario"]): r for r in paired}
            no5024 = [s for s in seeds if s != 5024]
            sa = paired_t_summary([float(p_idx[(s, "best_val", "fixed_030")]["delta_he_R2_minus_A"]) for s in no5024], 0.0, 0.025)
            sb = paired_t_summary([float(p_idx[(s, "best_val", "fixed_020")]["delta_oce_R2_minus_A"]) for s in no5024], 0.00010, 0.025)
            extra = {
                "max_cost_accounting_residual": max_resid,
                "min_mean_cost_eval": min_cost,
                "p_iut_excl_5024": max(sa["p_one_sided"], sb["p_one_sided"]),
            }

        exp_checks[label] = {
            "eval_rows": len(evals),
            "paired_rows": len(paired),
            "best_val_pkls": len(best_pkls),
            "final_with_opt_pkls": len(final_pkls),
            "verdict": recomp["verdict"],
            "p_iut_max": recomp["p_iut_max"],
            "H1a_mean": recomp["co_primary"]["H1a_HE_030_best_val"]["mean"],
            "H1b_mean": recomp["co_primary"]["H1b_OCE_020_best_val"]["mean"],
            **extra,
        }

    all_sha_ok = all(v["passed"] for v in sha_report.values()) if sha_report else False
    return {
        "level": "Level_1_Integrity_Schema_and_Stats",
        "status": "PASS" if (all_sha_ok or not top_sums.exists()) else "FAIL",
        "sha_checks": sha_report,
        "experiments": exp_checks,
    }


def verify_level2_ckpt_reload(n_seeds_check: int = 24) -> Dict[str, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
    except Exception as exc:
        return {"level": "Level_2_Checkpoint_Reload_Eval", "status": f"SKIPPED ({type(exc).__name__}: {exc})"}

    _setup_snapshot_sys_path()
    from conf_keys import derive_conf_seed_bundle
    from conf_design import get_locked_branch_configs as cfg0
    from conf_run import build_conf_test_for_seed as build_test0
    from cost_keys import derive_cost_seed_bundle
    from cost_design import get_locked_branch_configs as cfg10, COST_S_LOCKED
    from cost_run import build_cost_test_for_seed as build_test10
    from bc_models import rollout_with_oce
    from r3_metrics import evaluate_rollout_metrics_r3

    gym0 = cfg0()["gym_cfg"]
    gym10 = cfg10()["gym_cfg"]
    results: Dict[str, Any] = {}

    for label, d, seeds, res_csv, is_cost in [
        ("Job_16911_0bps", C0_DIR, list(range(4001, 4001 + n_seeds_check)), "RESULTS_CONFIRMATORY_FULL_PRECISION.csv", False),
        ("Job_16914_10bps", C10_DIR, list(range(5001, 5001 + n_seeds_check)), "RESULTS_COST_CONFIRMATORY_FULL_PRECISION.csv", True),
    ]:
        rows = _load_csv(d / "results" / "confirmatory_n24" / res_csv)
        idx = {(int(r["seed"]), r["ckpt_kind"], r["config"], r["scenario"]): r for r in rows}
        gym = gym10 if is_cost else gym0
        max_he_diff = 0.0
        max_oce_diff = 0.0
        checked_evals = 0

        for s in seeds:
            bundle = derive_cost_seed_bundle(s) if is_cost else derive_conf_seed_bundle(s)
            tests = build_test10(bundle, cost_s=COST_S_LOCKED) if is_cost else build_test0(bundle)
            for ck_kind, fn in (("best_val", "best_val.pkl"), ("ep200", "final_with_opt.pkl")):
                for cname, nom_idx in (("A_DECAY", 0), ("R2_DECAY", 1)):
                    pkl_path = d / "checkpoints" / "confirmatory_n24" / f"seed_{s}" / cname / fn
                    obj = pickle.loads(pkl_path.read_bytes())
                    params = jax.tree_util.tree_map(jnp.asarray, obj["params"])
                    for sc in ("fixed_020", "fixed_030"):
                        roll = rollout_with_oce(params, gym, tests[sc], oce_idx=nom_idx)
                        m = evaluate_rollout_metrics_r3(roll)
                        rec = idx[(s, ck_kind, cname, sc)]
                        max_he_diff = max(max_he_diff, abs(float(m["he_net_ddof0"]) - float(rec["he_net_ddof0"])))
                        max_oce_diff = max(max_oce_diff, abs(float(m["hedged_oce_y_nom"]) - float(rec["hedged_oce_y_nom"])))
                        checked_evals += 1

        results[label] = {
            "seeds_checked": len(seeds),
            "checkpoint_scenario_evals_checked": checked_evals,
            "max_he_net_ddof0_abs_diff": max_he_diff,
            "max_hedged_oce_y_nom_abs_diff": max_oce_diff,
            "passed_tol_1e5": max(max_he_diff, max_oce_diff) < 1e-5,
        }

    ok = all(v["passed_tol_1e5"] for v in results.values())
    return {"level": "Level_2_Checkpoint_Reload_Eval", "status": "PASS" if ok else "FAIL", "experiments": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify R2_matched confirmatory package (Levels 1-3)")
    parser.add_argument("--level", choices=("1", "2", "auto"), default="auto")
    parser.add_argument("--n-seeds-l2", type=int, default=24)
    args = parser.parse_args()

    t0 = time.time()
    out: Dict[str, Any] = {
        "level_1": verify_level1_integrity_and_stats(),
        "level_2": (
            verify_level2_ckpt_reload(args.n_seeds_l2)
            if args.level in ("2", "auto")
            else {"level": "Level_2_Checkpoint_Reload_Eval", "status": "NOT_REQUESTED"}
        ),
        "level_3": {
            "level": "Level_3_Full_Retraining_96_Runs",
            "status": "NOT_RUN_IN_PACKAGING_BY_DESIGN",
            "slurm_commands": [
                "sbatch robust_r2_matched_confirmatory/slurm_confirmatory_n24.sbatch",
                "sbatch robust_r2_matched_cost_confirmatory/slurm_cost_confirmatory_n24.sbatch",
            ],
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(json.dumps(out, indent=2))
    if out["level_1"]["status"] != "PASS" or out["level_2"]["status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
