"""
robust_r2_matched_cost_confirmatory/cost_run.py:
Locked runner & dry-run validator for R2_matched_128 vs A_ref_128 at cost_s = 0.0010 (10 bps).

Modes:
  --mode dry_run (DEFAULT):
      Verifies PRNG isolation across all historical + 10 bps keys, validates locked config,
      param parity, cost accounting, and source SHA-256 WITHOUT generating confirmatory test data.
  --mode tech_smoke:
      Isolated end-to-end verification ONLY on COST_SMOKE_SEED = 5999 (never enters [5001..5024]).
  --mode execute_confirmatory --confirm-locked-run --n-seeds 24 [--seed-offset 0]:
      Explicitly gated execution on locked cohort [5001..5024].
      Preserves prior COMPLETED & ALGORITHM_FAILURE records in SEED_LEDGER/CSVs if resuming after
      infrastructure interruption; constructs conf_test (20,000 paths/scenario) ONLY after both
      branches' best_val & ep200 checkpoints are closed and verified (< 1e-6 reload parity).
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import functools
import json
import pickle
import platform
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp

try:
    import resource
except ImportError:
    resource = None

COST_CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = COST_CONF_DIR.parent
LR_DIR = EXP_DIR / "robust_r3_lr_stability"
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"
R2_CONF_DIR = EXP_DIR / "robust_r2_matched_confirmatory"

for _p in (str(COST_CONF_DIR), str(R2_CONF_DIR), str(LR_DIR), str(BC_DIR), str(R3_DIR), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from r3_keys import TRAINING_SCENARIOS, TEST_SCENARIOS
from bc_models import count_param_breakdown, init_full_params, rollout_with_oce


def classify_cost_run_exception(exc: Exception) -> str:
    """Safely classify algorithmic/numerical failures vs OS/infrastructure errors without substring false positives."""
    if isinstance(exc, (FloatingPointError, AssertionError)):
        return "ALGORITHM_FAILURE"
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(tok in msg for tok in ("reload parity", "verify_finite", "cost_accounting_residual")):
        return "ALGORITHM_FAILURE"
    if re.search(r"\b(nan|inf|-inf|\+inf|infinity|non-finite|nonfinite)\b", msg):
        return "ALGORITHM_FAILURE"
    return "INFRA_ERROR"
from cost_keys import (
    COST_SMOKE_SEED,
    COST_LOCKED_COHORT_SEEDS,
    COST_EVAL_SCENARIOS,
    derive_cost_seed_bundle,
    verify_cost_key_isolation,
)
from cost_design import (
    PRIMARY_BRANCHES,
    COST_S_LOCKED,
    TRAIN_PATHS_PER_SCENARIO,
    VAL_PATHS_PER_SCENARIO,
    CONF_TEST_PATHS_PER_SCENARIO,
    EPS_NONINFERIORITY_OCE020,
    ALPHA_ONE_SIDED,
    PRIMARY_N_SEEDS,
    get_locked_train_config,
    get_locked_branch_configs,
    summarize_locked_spec,
)
from cost_world_and_metrics import (
    build_cost_scenario_world,
    evaluate_cost_rollout_metrics,
    hash_batch_spot_and_cost,
    sha256_file,
)
from cost_train import train_cost_config
from cost_stats import analyze_cost_confirmatory_pairs


@functools.partial(jax.jit, static_argnums=(1,))
def _rollout_oce_idx0(params: dict, gym_cfg, batch: dict) -> dict:
    return rollout_with_oce(params, gym_cfg, batch, oce_idx=0)


@functools.partial(jax.jit, static_argnums=(1,))
def _rollout_oce_idx1(params: dict, gym_cfg, batch: dict) -> dict:
    return rollout_with_oce(params, gym_cfg, batch, oce_idx=1)


@functools.partial(jax.jit, static_argnums=(1,))
def _rollout_oce_idx2(params: dict, gym_cfg, batch: dict) -> dict:
    return rollout_with_oce(params, gym_cfg, batch, oce_idx=2)


@functools.partial(jax.jit, static_argnums=(1,))
def _rollout_oce_idx3(params: dict, gym_cfg, batch: dict) -> dict:
    return rollout_with_oce(params, gym_cfg, batch, oce_idx=3)


def rollout_fast(params: dict, gym_cfg, batch: dict, nom_idx: int) -> dict:
    return (_rollout_oce_idx0, _rollout_oce_idx1, _rollout_oce_idx2, _rollout_oce_idx3)[nom_idx](
        params, gym_cfg, batch
    )


def reload_ckpt(path: Path) -> Tuple[dict, dict]:
    obj = pickle.loads(path.read_bytes())
    params_jnp = jax.tree_util.tree_map(jnp.asarray, obj["params"])
    return params_jnp, obj


def build_cost_train_and_val_for_seed(
    bundle: dict,
    cost_s: float = COST_S_LOCKED,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    train_d: Dict[str, dict] = {}
    val_d: Dict[str, dict] = {}
    for sc in TRAINING_SCENARIOS:
        train_d[sc] = build_cost_scenario_world(
            sc, bundle["train_dW"][sc], samples=TRAIN_PATHS_PER_SCENARIO, cost_s=cost_s
        )
        val_d[sc] = build_cost_scenario_world(
            sc, bundle["val_dW"][sc], samples=VAL_PATHS_PER_SCENARIO, cost_s=cost_s
        )
    for sub in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        train_d[sub] = build_cost_scenario_world(
            "fixed_020", bundle["train_dW"][sub], samples=TRAIN_PATHS_PER_SCENARIO, cost_s=cost_s
        )
    return train_d, val_d


def build_cost_test_for_seed(bundle: dict, cost_s: float = COST_S_LOCKED) -> Dict[str, dict]:
    """Confirmatory 20,000-path/scenario test suite under c_s; called ONLY after selection seals."""
    sdw = bundle["conf_test_shared_dW"]
    return {
        sc: build_cost_scenario_world(sc, sdw, samples=CONF_TEST_PATHS_PER_SCENARIO, cost_s=cost_s)
        for sc in COST_EVAL_SCENARIOS
    }


def resolve_locked_cost_seeds(
    n_seeds: int,
    seed_offset: int = 0,
    locked_n_total: int = PRIMARY_N_SEEDS,
    existing_seed_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    locked_pool = list(COST_LOCKED_COHORT_SEEDS[:locked_n_total])
    if seed_offset < 0 or n_seeds < 1 or (seed_offset + n_seeds) > len(locked_pool):
        raise ValueError(
            f"Slice exceeds locked N={locked_n_total} cohort [{locked_pool[0]}..{locked_pool[-1]}]: "
            f"seed_offset={seed_offset}, n_seeds={n_seeds}"
        )
    target_slice = locked_pool[seed_offset : seed_offset + n_seeds]
    by_seed: Dict[int, str] = {}
    if existing_seed_records:
        for rec in existing_seed_records:
            s_id = int(rec["seed"])
            st = str(rec.get("status", "UNKNOWN"))
            if st == "FAILED" and rec.get("error_category") in ("ALGORITHM_FAILURE", "INFRA_ERROR"):
                st = str(rec["error_category"])
            by_seed[s_id] = st

    to_run: List[int] = []
    skipped_completed: List[int] = []
    preserved_algo_failures: List[int] = []
    retried_infra: List[int] = []

    for s in target_slice:
        st = by_seed.get(s)
        if st == "COMPLETED":
            skipped_completed.append(s)
        elif st == "ALGORITHM_FAILURE":
            preserved_algo_failures.append(s)
        elif st == "INFRA_ERROR":
            retried_infra.append(s)
            to_run.append(s)
        else:
            to_run.append(s)

    return to_run, {
        "locked_cohort": locked_pool,
        "target_slice": target_slice,
        "to_run": to_run,
        "skipped_completed": skipped_completed,
        "preserved_algo_failures": preserved_algo_failures,
        "retried_infra": retried_infra,
    }


def _tracked_source_files() -> List[Path]:
    return [
        COST_CONF_DIR / "COST_CONFIRMATORY_DESIGN_LOCK.md",
        COST_CONF_DIR / "COST_ACCOUNTING_AUDIT.md",
        COST_CONF_DIR / "SEED_INDEPENDENCE_AUDIT.md",
        COST_CONF_DIR / "DATA_AND_SEED_MANIFEST.json",
        COST_CONF_DIR / "SAMPLE_SIZE_SENSITIVITY_COST.csv",
        COST_CONF_DIR / "cost_keys.py",
        COST_CONF_DIR / "cost_design.py",
        COST_CONF_DIR / "cost_world_and_metrics.py",
        COST_CONF_DIR / "cost_train.py",
        COST_CONF_DIR / "cost_stats.py",
        COST_CONF_DIR / "cost_run.py",
        COST_CONF_DIR / "scripts" / "cost_power_table.py",
        COST_CONF_DIR / "tests" / "test_cost_conf_suite.py",
        R2_CONF_DIR / "conf_stats.py",
        LR_DIR / "lr_schedule.py",
        LR_DIR / "lr_train.py",
        BC_DIR / "bc_models.py",
        R3_DIR / "r3_models.py",
        R3_DIR / "r3_metrics.py",
        R3_DIR / "r3_train.py",
        R3_DIR / "r3_keys.py",
        EXP_DIR / "scripts" / "robust_deep_hedging_phase2" / "robust_world.py",
    ]


def _collect_tracked_hashes_and_env() -> Dict[str, Any]:
    return {
        "source_sha256": {
            str(p.relative_to(EXP_DIR)).replace("\\", "/"): sha256_file(p)
            for p in _tracked_source_files()
            if p.exists()
        },
        "environment": {
            "python": sys.version,
            "jax_version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "platform": platform.platform(),
        },
    }


def _write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _read_existing_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def run_dry_run_check() -> Dict[str, Any]:
    iso = verify_cost_key_isolation()
    assert iso["isolation_passed"], "Cost PRNG isolation audit failed!"
    spec = summarize_locked_spec()
    tcfg = get_locked_train_config()
    cfgs = get_locked_branch_configs()

    dummy_key = jax.random.PRNGKey(0)
    gym_cfg = cfgs["gym_cfg"]
    feat_dims = cfgs["feature_dims"]
    p_a = init_full_params(dummy_key, gym_cfg, feat_dims, k_oce=1)
    p_r2 = init_full_params(dummy_key, gym_cfg, feat_dims, k_oce=4)
    max_policy_init_diff = max(
        float(jnp.max(jnp.abs(l1 - l2)))
        for l1, l2 in zip(jax.tree_util.tree_leaves(p_a["policy"]), jax.tree_util.tree_leaves(p_r2["policy"]))
    )
    assert max_policy_init_diff == 0.0
    cnt_a = count_param_breakdown(p_a)
    cnt_r2 = count_param_breakdown(p_r2)
    assert cnt_a["policy_total_params"] == cnt_r2["policy_total_params"] == 6762
    assert cnt_a["oce_total_params"] == 2 and cnt_a["grand_total_params"] == 6764
    assert cnt_r2["oce_total_params"] == 8 and cnt_r2["grand_total_params"] == 6770

    prov = _collect_tracked_hashes_and_env()
    report = {
        "mode": "dry_run",
        "cost_s_locked": COST_S_LOCKED,
        "test_sets_5001_5040_status": "NOT_GENERATED_AND_NEVER_OPENED",
        "prior_4001_4040_status": "UNTOUCHED_AND_EXCLUDED",
        "hold_eval_status": "SEALED_AND_NEVER_OPENED",
        "isolation": {k: v for k, v in iso.items() if k != "bundles"},
        "spec": spec,
        "param_parity": {
            "A_DECAY": cnt_a,
            "R2_DECAY": cnt_r2,
            "max_policy_init_diff_dummy": max_policy_init_diff,
        },
        "source_sha256": prov["source_sha256"],
        "environment": prov["environment"],
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_experiment(
    mode: str,
    n_seeds: int,
    seed_offset: int = 0,
    smoke_epochs: int = 2,
    confirm_locked_run: bool = False,
) -> None:
    if mode == "dry_run":
        run_dry_run_check()
        return

    if mode == "tech_smoke":
        out_dir = COST_CONF_DIR / "results" / "tech_smoke_seed5999_NON_CONFIRMATORY"
        ckpt_base = COST_CONF_DIR / "checkpoints" / "tech_smoke_seed5999_NON_CONFIRMATORY"
        seeds_to_run = [COST_SMOKE_SEED]
        epochs = int(smoke_epochs)
        run_tag = out_dir.name
        slice_info = {"mode": "tech_smoke", "to_run": seeds_to_run}
        existing_eval_rows: List[dict] = []
        existing_paired_rows: List[dict] = []
        existing_seed_records: List[dict] = []
        existing_failed_seeds: List[dict] = []
        existing_reload_checks: List[dict] = []
        existing_dataset_hashes: Dict[str, Any] = {}
    elif mode == "execute_confirmatory":
        if not confirm_locked_run:
            raise RuntimeError("Refusing confirmatory execution without --confirm-locked-run.")
        run_tag = "confirmatory_n24" if (seed_offset == 0 and n_seeds == PRIMARY_N_SEEDS) else f"confirmatory_n{PRIMARY_N_SEEDS}_resume"
        # Fixed canonical accumulation folder across any --seed-offset slice in locked [5001..5024]
        out_dir = COST_CONF_DIR / "results" / "confirmatory_n24"
        ckpt_base = COST_CONF_DIR / "checkpoints" / "confirmatory_n24"
        ledger_path = out_dir / "SEED_LEDGER.json"
        if ledger_path.exists():
            prev_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            existing_seed_records = list(prev_ledger.get("seed_records", []))
            existing_failed_seeds = list(prev_ledger.get("failed_seeds", []))
            existing_reload_checks = list(prev_ledger.get("reload_checks", []))
            existing_dataset_hashes = dict(prev_ledger.get("dataset_hashes", {}))
        else:
            existing_seed_records = []
            existing_failed_seeds = []
            existing_reload_checks = []
            existing_dataset_hashes = {}
        existing_eval_rows = _read_existing_csv_rows(out_dir / "RESULTS_COST_CONFIRMATORY_FULL_PRECISION.csv")
        existing_paired_rows = _read_existing_csv_rows(out_dir / "PAIRED_COST_CONFIRMATORY.csv")
        summary_prev_path = out_dir / "COST_CONFIRMATORY_STATS_SUMMARY.json"
        if not existing_reload_checks and summary_prev_path.exists():
            sprev = json.loads(summary_prev_path.read_text(encoding="utf-8"))
            existing_reload_checks = list(sprev.get("reload_checks", []))
            existing_dataset_hashes = dict(sprev.get("dataset_hashes", {}))

        seeds_to_run, slice_info = resolve_locked_cost_seeds(
            n_seeds=n_seeds,
            seed_offset=seed_offset,
            locked_n_total=PRIMARY_N_SEEDS,
            existing_seed_records=existing_seed_records,
        )
        epochs = 200
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    iso = verify_cost_key_isolation()
    assert iso["isolation_passed"], "PRNG isolation audit failed!"
    prov = _collect_tracked_hashes_and_env()
    spec = summarize_locked_spec()

    tcfg = get_locked_train_config()
    if epochs != tcfg.epochs:
        tcfg = replace(tcfg, epochs=epochs)
    cfgs = get_locked_branch_configs()
    gym_cfg = cfgs["gym_cfg"]
    feat_dims = cfgs["feature_dims"]

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base.mkdir(parents=True, exist_ok=True)

    # Filter out prior INFRA_ERROR placeholder rows for retried seeds while preserving COMPLETED & ALGORITHM_FAILURE
    retry_set = set(slice_info.get("retried_infra", []))
    eval_rows: List[Dict[str, Any]] = [r for r in existing_eval_rows if int(r["seed"]) not in retry_set]
    paired_rows: List[Dict[str, Any]] = [r for r in existing_paired_rows if int(r["seed"]) not in retry_set]
    seed_records: List[Dict[str, Any]] = [r for r in existing_seed_records if int(r["seed"]) not in retry_set]
    failed_seeds: List[Dict[str, Any]] = [r for r in existing_failed_seeds if int(r["seed"]) not in retry_set]
    reload_checks: List[Dict[str, Any]] = [r for r in existing_reload_checks if int(r["seed"]) not in retry_set]
    dataset_hashes: Dict[str, Any] = {k: v for k, v in existing_dataset_hashes.items() if int(k) not in retry_set}

    t_global = time.time()
    final_ckpt_label = "ep200" if epochs == 200 else f"ep{epochs:03d}"

    for s in seeds_to_run:
        try:
            bundle = derive_cost_seed_bundle(s)
            train_d, val_d = build_cost_train_and_val_for_seed(bundle, cost_s=COST_S_LOCKED)
            dataset_hashes[str(s)] = {
                "train": {k: hash_batch_spot_and_cost(v) for k, v in train_d.items()},
                "val": {k: hash_batch_spot_and_cost(v) for k, v in val_d.items()},
            }

            recs: Dict[str, Any] = {}
            for cname in PRIMARY_BRANCHES:
                rec = train_cost_config(
                    cname,
                    bundle,
                    train_d,
                    val_d,
                    gym_cfg,
                    tcfg,
                    feat_dims,
                    ckpt_base / f"seed_{s}" / cname,
                )
                recs[cname] = rec
                p_disk, _ = reload_ckpt(Path(rec["ckpt_best_path"]))
                nom_idx = 0 if cname == "A_DECAY" else 1
                roll_disk = rollout_fast(p_disk, gym_cfg, val_d["fixed_020"], nom_idx)
                r_disk = float(jnp.mean(roll_disk["hedged_loss"]))
                diff = abs(r_disk - rec["val_nominal_hedged_at_best"])
                assert diff < 1e-6, f"Reload parity failed {s}/{cname}: {diff}"
                reload_checks.append(
                    {
                        "seed": s,
                        "config": cname,
                        "val_nom_reload_diff": diff,
                        "val_cost_mean_020": float(jnp.mean(roll_disk["cost"])),
                        "actual_update_steps": int(rec["actual_update_steps"]),
                        "actual_paths_processed": int(rec["actual_paths_processed"]),
                    }
                )

            # Selection is now SEALED. Build conf_test ONLY after training & selection close.
            conf_test = build_cost_test_for_seed(bundle, cost_s=COST_S_LOCKED)
            dataset_hashes[str(s)]["conf_test"] = {k: hash_batch_spot_and_cost(v) for k, v in conf_test.items()}

            for ck_kind in ("best_val", final_ckpt_label):
                p_a, obj_a = reload_ckpt(
                    Path(recs["A_DECAY"]["ckpt_best_path"] if ck_kind == "best_val" else recs["A_DECAY"]["final_opt_path"])
                )
                p_r2, obj_r2 = reload_ckpt(
                    Path(recs["R2_DECAY"]["ckpt_best_path"] if ck_kind == "best_val" else recs["R2_DECAY"]["final_opt_path"])
                )

                for sc, batch in conf_test.items():
                    roll_a = rollout_fast(p_a, gym_cfg, batch, 0)
                    m_a = evaluate_cost_rollout_metrics(roll_a, eps=EPS_NONINFERIORITY_OCE020)
                    assert m_a["cost_accounting_residual"] < 1e-5

                    roll_r2_nom = rollout_fast(p_r2, gym_cfg, batch, 1)
                    roll_r2_scen = (
                        rollout_fast(p_r2, gym_cfg, batch, TRAINING_SCENARIOS.index(sc))
                        if sc in TRAINING_SCENARIOS
                        else None
                    )
                    m_r2 = evaluate_cost_rollout_metrics(
                        roll_r2_nom,
                        rollout_scen=roll_r2_scen,
                        ref_hedged_oce_y_nom=m_a["hedged_oce_y_nom"],
                        eps=EPS_NONINFERIORITY_OCE020,
                    )
                    assert m_r2["cost_accounting_residual"] < 1e-5

                    for cname, obj_c, p_c, m_c, nom_i in (
                        ("A_DECAY", obj_a, p_a, m_a, 0),
                        ("R2_DECAY", obj_r2, p_r2, m_r2, 1),
                    ):
                        eval_rows.append(
                            {
                                "seed": s,
                                "ckpt_kind": ck_kind,
                                "epoch": int(obj_c["epoch"]),
                                "best_epoch": recs[cname]["best_epoch"],
                                "best_epoch_if_val020_only": recs[cname]["best_epoch_if_val020_only"],
                                "config": cname,
                                "branch": recs[cname]["branch"],
                                "schedule": "DECAY",
                                "cost_s": COST_S_LOCKED,
                                "scenario": sc,
                                "hedged_oce_y_nom": m_c["hedged_oce_y_nom"],
                                "hedged_oce_y_scen": m_c["hedged_oce_y_scen"],
                                "he_net_ddof0": m_c["he_net_ddof0"],
                                "he_net_ddof1": m_c["he_net_ddof1"],
                                "var95": m_c["var95"],
                                "cvar95": m_c["cvar95"],
                                "mean_gains": m_c["mean_gains"],
                                "mean_gross_gains": m_c["mean_gross_gains"],
                                "mean_pnl": m_c["mean_pnl"],
                                "mean_payoff": m_c["mean_payoff"],
                                "mean_cost": m_c["mean_cost"],
                                "max_abs_cost": m_c["max_abs_cost"],
                                "cost_accounting_residual": m_c["cost_accounting_residual"],
                                "turnover": m_c["turnover"],
                                "active_steps_gt1e5": m_c["active_steps_gt1e5"],
                                "active_trade_rate": m_c["active_trade_rate"],
                                "immobility_rate": m_c["immobility_rate"],
                                "immobility_rate_lt1e4": m_c["immobility_rate_lt1e4"],
                                "immobility_rate_lt1e3": m_c["immobility_rate_lt1e3"],
                                "mean_trade_size_active_pooled": m_c["mean_trade_size_active_pooled"],
                                "mean_trade_size_active_pathwise": m_c["mean_trade_size_active_pathwise"],
                                "zero_active_path_frac": m_c["zero_active_path_frac"],
                                "mean_abs_delta": m_c["mean_abs_delta"],
                                "max_abs_delta": m_c["max_abs_delta"],
                                "p99_abs_delta": m_c["p99_abs_delta"],
                                "frac_steps_abs_delta_gt2": m_c["frac_steps_abs_delta_gt2"],
                                "frac_steps_abs_delta_ge45": m_c["frac_steps_abs_delta_ge45"],
                                "learned_y_nom": float(np.asarray(p_c["oce"]["y"])[nom_i]),
                                "train_time_sec": recs[cname]["total_time_sec"],
                                "actual_update_steps": recs[cname]["actual_update_steps"],
                                "actual_paths_processed": recs[cname]["actual_paths_processed"],
                            }
                        )

                    d_oce = float(m_r2["hedged_oce_y_nom"] - m_a["hedged_oce_y_nom"])
                    d_he = float(m_r2["he_net_ddof0"] - m_a["he_net_ddof0"])
                    d_oce_scen = (
                        float(m_r2["hedged_oce_y_scen"] - m_a["hedged_oce_y_nom"])
                        if np.isfinite(m_r2["hedged_oce_y_scen"])
                        else float("nan")
                    )
                    paired_rows.append(
                        {
                            "seed": s,
                            "ckpt_kind": ck_kind,
                            "scenario": sc,
                            "A_he_net_ddof0": m_a["he_net_ddof0"],
                            "R2_he_net_ddof0": m_r2["he_net_ddof0"],
                            "delta_he_R2_minus_A": d_he,
                            "A_hedged_oce_y_nom": m_a["hedged_oce_y_nom"],
                            "R2_hedged_oce_y_nom": m_r2["hedged_oce_y_nom"],
                            "delta_oce_R2_minus_A": d_oce,
                            "exceed_eps_oce": bool(d_oce > EPS_NONINFERIORITY_OCE020),
                            "R2_hedged_oce_y_scen": m_r2["hedged_oce_y_scen"],
                            "delta_oce_y_scen_R2_minus_A": d_oce_scen,
                            "A_cvar95": m_a["cvar95"],
                            "R2_cvar95": m_r2["cvar95"],
                            "delta_cvar95_R2_minus_A": float(m_r2["cvar95"] - m_a["cvar95"]),
                            "A_var95": m_a["var95"],
                            "R2_var95": m_r2["var95"],
                            "delta_var95_R2_minus_A": float(m_r2["var95"] - m_a["var95"]),
                            "A_mean_gains": m_a["mean_gains"],
                            "R2_mean_gains": m_r2["mean_gains"],
                            "delta_mean_gains_R2_minus_A": float(m_r2["mean_gains"] - m_a["mean_gains"]),
                            "A_mean_gross_gains": m_a["mean_gross_gains"],
                            "R2_mean_gross_gains": m_r2["mean_gross_gains"],
                            "delta_mean_gross_gains_R2_minus_A": float(
                                m_r2["mean_gross_gains"] - m_a["mean_gross_gains"]
                            ),
                            "A_mean_cost": m_a["mean_cost"],
                            "R2_mean_cost": m_r2["mean_cost"],
                            "delta_mean_cost_R2_minus_A": float(m_r2["mean_cost"] - m_a["mean_cost"]),
                            "A_turnover": m_a["turnover"],
                            "R2_turnover": m_r2["turnover"],
                            "delta_turnover_R2_minus_A": float(m_r2["turnover"] - m_a["turnover"]),
                            "A_active_steps_gt1e5": m_a["active_steps_gt1e5"],
                            "R2_active_steps_gt1e5": m_r2["active_steps_gt1e5"],
                            "delta_active_steps_R2_minus_A": float(
                                m_r2["active_steps_gt1e5"] - m_a["active_steps_gt1e5"]
                            ),
                            "A_immobility_rate": m_a["immobility_rate"],
                            "R2_immobility_rate": m_r2["immobility_rate"],
                            "delta_immobility_rate_R2_minus_A": float(
                                m_r2["immobility_rate"] - m_a["immobility_rate"]
                            ),
                            "A_trade_size_active_pooled": m_a["mean_trade_size_active_pooled"],
                            "R2_trade_size_active_pooled": m_r2["mean_trade_size_active_pooled"],
                            "delta_trade_size_active_pooled_R2_minus_A": float(
                                m_r2["mean_trade_size_active_pooled"] - m_a["mean_trade_size_active_pooled"]
                            ),
                            "A_trade_size_active_pathwise": m_a["mean_trade_size_active_pathwise"],
                            "R2_trade_size_active_pathwise": m_r2["mean_trade_size_active_pathwise"],
                            "delta_trade_size_active_pathwise_R2_minus_A": float(
                                m_r2["mean_trade_size_active_pathwise"] - m_a["mean_trade_size_active_pathwise"]
                            ),
                            "A_max_abs_delta": m_a["max_abs_delta"],
                            "R2_max_abs_delta": m_r2["max_abs_delta"],
                            "delta_max_abs_delta_R2_minus_A": float(m_r2["max_abs_delta"] - m_a["max_abs_delta"]),
                            "A_p99_abs_delta": m_a["p99_abs_delta"],
                            "R2_p99_abs_delta": m_r2["p99_abs_delta"],
                            "delta_p99_abs_delta_R2_minus_A": float(m_r2["p99_abs_delta"] - m_a["p99_abs_delta"]),
                            "A_frac_steps_abs_delta_gt2": m_a["frac_steps_abs_delta_gt2"],
                            "R2_frac_steps_abs_delta_gt2": m_r2["frac_steps_abs_delta_gt2"],
                            "delta_frac_steps_abs_delta_gt2_R2_minus_A": float(
                                m_r2["frac_steps_abs_delta_gt2"] - m_a["frac_steps_abs_delta_gt2"]
                            ),
                        }
                    )

            seed_records.append(
                {
                    "seed": s,
                    "status": "COMPLETED",
                    "A_DECAY_time_sec": recs["A_DECAY"]["total_time_sec"],
                    "R2_DECAY_time_sec": recs["R2_DECAY"]["total_time_sec"],
                    "A_best_epoch": recs["A_DECAY"]["best_epoch"],
                    "R2_best_epoch": recs["R2_DECAY"]["best_epoch"],
                    "A_actual_steps": recs["A_DECAY"]["actual_update_steps"],
                    "R2_actual_steps": recs["R2_DECAY"]["actual_update_steps"],
                    "A_train_cost_mean_020": recs["A_DECAY"]["train_cost_means"]["fixed_020"],
                    "R2_train_cost_mean_020": recs["R2_DECAY"]["train_cost_means"]["fixed_020"],
                }
            )
        except Exception as exc:
            # Clean up any partial rows appended for seed s before recording failure NaNs
            eval_rows = [r for r in eval_rows if int(r["seed"]) != s]
            paired_rows = [r for r in paired_rows if int(r["seed"]) != s]
            reload_checks = [r for r in reload_checks if int(r["seed"]) != s]
            err_cat = classify_cost_run_exception(exc)
            failed_seeds.append({"seed": s, "error_category": err_cat, "error": f"{type(exc).__name__}: {exc}"})
            seed_records.append(
                {
                    "seed": s,
                    "status": "FAILED",
                    "error_category": err_cat,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            nan_v = float("nan")
            for ck_kind in ("best_val", final_ckpt_label):
                for sc in COST_EVAL_SCENARIOS:
                    paired_rows.append(
                        {
                            "seed": s,
                            "ckpt_kind": ck_kind,
                            "scenario": sc,
                            "A_he_net_ddof0": nan_v,
                            "R2_he_net_ddof0": nan_v,
                            "delta_he_R2_minus_A": nan_v,
                            "A_hedged_oce_y_nom": nan_v,
                            "R2_hedged_oce_y_nom": nan_v,
                            "delta_oce_R2_minus_A": nan_v,
                            "exceed_eps_oce": False,
                            "R2_hedged_oce_y_scen": nan_v,
                            "delta_oce_y_scen_R2_minus_A": nan_v,
                            "A_cvar95": nan_v,
                            "R2_cvar95": nan_v,
                            "delta_cvar95_R2_minus_A": nan_v,
                            "A_var95": nan_v,
                            "R2_var95": nan_v,
                            "delta_var95_R2_minus_A": nan_v,
                            "A_mean_gains": nan_v,
                            "R2_mean_gains": nan_v,
                            "delta_mean_gains_R2_minus_A": nan_v,
                            "A_mean_gross_gains": nan_v,
                            "R2_mean_gross_gains": nan_v,
                            "delta_mean_gross_gains_R2_minus_A": nan_v,
                            "A_mean_cost": nan_v,
                            "R2_mean_cost": nan_v,
                            "delta_mean_cost_R2_minus_A": nan_v,
                            "A_turnover": nan_v,
                            "R2_turnover": nan_v,
                            "delta_turnover_R2_minus_A": nan_v,
                            "A_active_steps_gt1e5": nan_v,
                            "R2_active_steps_gt1e5": nan_v,
                            "delta_active_steps_R2_minus_A": nan_v,
                            "A_immobility_rate": nan_v,
                            "R2_immobility_rate": nan_v,
                            "delta_immobility_rate_R2_minus_A": nan_v,
                            "A_trade_size_active_pooled": nan_v,
                            "R2_trade_size_active_pooled": nan_v,
                            "delta_trade_size_active_pooled_R2_minus_A": nan_v,
                            "A_trade_size_active_pathwise": nan_v,
                            "R2_trade_size_active_pathwise": nan_v,
                            "delta_trade_size_active_pathwise_R2_minus_A": nan_v,
                            "A_max_abs_delta": nan_v,
                            "R2_max_abs_delta": nan_v,
                            "delta_max_abs_delta_R2_minus_A": nan_v,
                            "A_p99_abs_delta": nan_v,
                            "R2_p99_abs_delta": nan_v,
                            "delta_p99_abs_delta_R2_minus_A": nan_v,
                            "A_frac_steps_abs_delta_gt2": nan_v,
                            "R2_frac_steps_abs_delta_gt2": nan_v,
                            "delta_frac_steps_abs_delta_gt2_R2_minus_A": nan_v,
                        }
                    )

        _write_csv(eval_rows, out_dir / "RESULTS_COST_CONFIRMATORY_FULL_PRECISION.csv")
        _write_csv(paired_rows, out_dir / "PAIRED_COST_CONFIRMATORY.csv")
        (out_dir / "SEED_LEDGER.json").write_text(
            json.dumps(
                {
                    "seed_records": seed_records,
                    "failed_seeds": failed_seeds,
                    "reload_checks": reload_checks,
                    "dataset_hashes": dataset_hashes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if mode == "execute_confirmatory":
        stats_summary = analyze_cost_confirmatory_pairs(
            paired_rows,
            planned_seeds=list(COST_LOCKED_COHORT_SEEDS[:PRIMARY_N_SEEDS]),
            eps_margin=EPS_NONINFERIORITY_OCE020,
            alpha_one_sided=ALPHA_ONE_SIDED,
            require_complete_clean=True,
        )
    else:
        stats_summary = {
            "mode": mode,
            "note": "NON_CONFIRMATORY_TECH_SMOKE_STATISTICAL_TEST_NOT_APPLIED",
            "seeds": seeds_to_run,
            "epochs": epochs,
        }

    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource is not None else -1
    summary = {
        "mode": mode,
        "run_tag": run_tag,
        "cost_s_locked": COST_S_LOCKED,
        "slice_info": slice_info,
        "seeds_executed_this_invocation": seeds_to_run,
        "seed_offset": int(seed_offset),
        "epochs": epochs,
        "wall_time_sec": float(time.time() - t_global),
        "max_rss_kb": max_rss_kb,
        "source_sha256": prov["source_sha256"],
        "environment": prov["environment"],
        "spec": spec,
        "isolation": {k: v for k, v in iso.items() if k != "bundles"},
        "failed_seeds": failed_seeds,
        "reload_checks": reload_checks,
        "seed_records": seed_records,
        "dataset_hashes": dataset_hashes,
        "stats_summary": stats_summary,
    }
    (out_dir / "COST_CONFIRMATORY_STATS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Completed mode={mode}, out_dir={out_dir}, wall={summary['wall_time_sec']:.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirmatory runner for R2_matched_128 vs A_ref_128 under 10 bps cost")
    parser.add_argument("--mode", choices=("dry_run", "tech_smoke", "execute_confirmatory"), default="dry_run")
    parser.add_argument("--n-seeds", type=int, default=PRIMARY_N_SEEDS)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--confirm-locked-run", action="store_true")
    args = parser.parse_args()
    run_experiment(
        mode=args.mode,
        n_seeds=args.n_seeds,
        seed_offset=args.seed_offset,
        smoke_epochs=args.smoke_epochs,
        confirm_locked_run=args.confirm_locked_run,
    )


if __name__ == "__main__":
    main()
