"""
conf_run.py: Locked runner & dry-run validator for the independent R2_matched confirmatory experiment.

Modes:
  --mode dry_run (DEFAULT):
      Verifies PRNG isolation, locked configs, parameter breakdown, LR schedule, and source SHA-256
      WITHOUT generating any confirmatory datasets ([4001..4040]) or touching HOLD_EVAL.
  --mode tech_smoke:
      Isolated pipeline verification ONLY on TECH_SMOKE_SEED=3999 (never enters confirmatory pool).
  --mode execute_confirmatory --confirm-locked-run --n-seeds 24:
      Explicitly gated confirmatory execution on CONF_RESERVED_SEEDS[:n_seeds].
      Builds conf_test (20,000 paths/scenario) ONLY after best_val & final checkpoints are sealed.
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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp

try:
    import resource
except ImportError:
    resource = None

CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = CONF_DIR.parent
LR_DIR = EXP_DIR / "robust_r3_lr_stability"
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"
P2_DIR = EXP_DIR / "scripts" / "robust_deep_hedging_phase2"

for _p in (str(CONF_DIR), str(LR_DIR), str(BC_DIR), str(R3_DIR), str(EXP_DIR), str(P2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from r3_keys import TRAINING_SCENARIOS, TEST_SCENARIOS
from r3_metrics import evaluate_rollout_metrics_r3
from bc_models import count_param_breakdown, init_full_params, rollout_with_oce
from run_r3_pilot import build_scenario_world, sha256_file, hash_batch_spot
from lr_train import train_lr_config
from conf_keys import (
    TECH_SMOKE_SEED,
    CONF_RESERVED_SEEDS,
    CONF_EVAL_SCENARIOS,
    derive_conf_seed_bundle,
    verify_conf_key_isolation,
)
from conf_design import (
    PRIMARY_BRANCHES,
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
from conf_stats import analyze_confirmatory_pairs


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
    fn_map = (_rollout_oce_idx0, _rollout_oce_idx1, _rollout_oce_idx2, _rollout_oce_idx3)
    return fn_map[nom_idx](params, gym_cfg, batch)


def reload_ckpt(path: Path) -> Tuple[dict, dict]:
    obj = pickle.loads(path.read_bytes())
    params_jnp = jax.tree_util.tree_map(jnp.asarray, obj["params"])
    return params_jnp, obj


def build_train_and_val_for_seed(bundle: dict) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    train_d: Dict[str, dict] = {}
    val_d: Dict[str, dict] = {}
    for sc in TRAINING_SCENARIOS:
        train_d[sc] = build_scenario_world(sc, bundle["train_dW"][sc], samples=TRAIN_PATHS_PER_SCENARIO)
        val_d[sc] = build_scenario_world(sc, bundle["val_dW"][sc], samples=VAL_PATHS_PER_SCENARIO)
    for sub in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        train_d[sub] = build_scenario_world("fixed_020", bundle["train_dW"][sub], samples=TRAIN_PATHS_PER_SCENARIO)
    return train_d, val_d


def build_conf_test_for_seed(bundle: dict) -> Dict[str, dict]:
    """Sealed confirmatory evaluation set (20,000 paths/scenario, shared dW); built ONLY after selection closes."""
    sdw = bundle["conf_test_shared_dW"]
    return {sc: build_scenario_world(sc, sdw, samples=CONF_TEST_PATHS_PER_SCENARIO) for sc in CONF_EVAL_SCENARIOS}


def run_dry_run_check() -> Dict[str, Any]:
    iso = verify_conf_key_isolation()
    assert iso["isolation_passed"], "PRNG isolation failed!"
    spec = summarize_locked_spec()
    tcfg = get_locked_train_config()
    cfgs = get_locked_branch_configs()

    # Structural init parity check using synthetic dummy seed 0 (NOT any reserved/test seed)
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

    tracked = [
        CONF_DIR / "R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md",
        CONF_DIR / "SAMPLE_SIZE_SENSITIVITY.csv",
        CONF_DIR / "conf_keys.py",
        CONF_DIR / "conf_design.py",
        CONF_DIR / "conf_stats.py",
        CONF_DIR / "conf_run.py",
        CONF_DIR / "scripts" / "sample_size_and_power.py",
        CONF_DIR / "tests" / "test_conf_suite.py",
        CONF_DIR / "DATA_AND_SEED_PLAN.json",
        LR_DIR / "lr_train.py",
        LR_DIR / "lr_schedule.py",
        LR_DIR / "lr_keys.py",
        BC_DIR / "bc_models.py",
        R3_DIR / "r3_models.py",
        R3_DIR / "r3_metrics.py",
        R3_DIR / "r3_train.py",
        R3_DIR / "r3_keys.py",
    ]
    hashes = {str(p.relative_to(EXP_DIR)).replace("\\", "/"): sha256_file(p) for p in tracked if p.exists()}

    report = {
        "mode": "dry_run",
        "test_sets_4001_4040_status": "NOT_GENERATED_AND_NEVER_OPENED",
        "hold_eval_status": "SEALED_AND_NEVER_OPENED",
        "isolation": {k: v for k, v in iso.items() if k != "bundles"},
        "spec": spec,
        "param_parity": {
            "A_DECAY": cnt_a,
            "R2_DECAY": cnt_r2,
            "max_policy_init_diff_dummy": max_policy_init_diff,
        },
        "source_sha256": hashes,
        "python": sys.version,
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "platform": platform.platform(),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def _collect_tracked_hashes_and_env() -> Dict[str, Any]:
    tracked = [
        CONF_DIR / "R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md",
        CONF_DIR / "SAMPLE_SIZE_SENSITIVITY.csv",
        CONF_DIR / "conf_keys.py",
        CONF_DIR / "conf_design.py",
        CONF_DIR / "conf_stats.py",
        CONF_DIR / "conf_run.py",
        CONF_DIR / "scripts" / "sample_size_and_power.py",
        CONF_DIR / "tests" / "test_conf_suite.py",
        CONF_DIR / "DATA_AND_SEED_PLAN.json",
        LR_DIR / "lr_train.py",
        LR_DIR / "lr_schedule.py",
        LR_DIR / "lr_keys.py",
        BC_DIR / "bc_models.py",
        R3_DIR / "r3_models.py",
        R3_DIR / "r3_metrics.py",
        R3_DIR / "r3_train.py",
        R3_DIR / "r3_keys.py",
    ]
    return {
        "source_sha256": {
            str(p.relative_to(EXP_DIR)).replace("\\", "/"): sha256_file(p)
            for p in tracked
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def resolve_locked_confirmatory_seeds(
    n_seeds: int,
    seed_offset: int = 0,
    locked_n_total: int = PRIMARY_N_SEEDS,
    existing_seed_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Strictly constrain --n-seeds / --seed-offset to the locked cohort [4001..4024].
    When continuing after infrastructure failures, preserve ALL COMPLETED or
    ALGORITHM_FAILURE seed records untouched (never drop/replace adverse seeds)
    and return ONLY missing or INFRA_ERROR seeds within the locked cohort slice.
    """
    locked_pool = list(CONF_RESERVED_SEEDS[:locked_n_total])
    if seed_offset < 0 or n_seeds < 1 or (seed_offset + n_seeds) > len(locked_pool):
        raise ValueError(
            f"Slice exceeds locked N={locked_n_total} cohort [4001..{4000 + locked_n_total}]: "
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

    info = {
        "locked_cohort": locked_pool,
        "target_slice": target_slice,
        "to_run": to_run,
        "skipped_completed": skipped_completed,
        "preserved_algo_failures": preserved_algo_failures,
        "retried_infra": retried_infra,
    }
    return to_run, info


def classify_run_exception(exc: Exception) -> str:
    """Distinguish algorithmic/numerical failure (non-finite, parity) from infrastructure failure."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    algo_tokens = ("non-finite", "nonfinite", "nan", "inf", "reload parity", "verify_finite")
    if any(tok in msg for tok in algo_tokens):
        return "ALGORITHM_FAILURE"
    return "INFRA_ERROR"


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
        seeds = [TECH_SMOKE_SEED]
        epochs = int(smoke_epochs)
        run_tag = "tech_smoke_seed3999_NON_CONFIRMATORY"
    elif mode == "execute_confirmatory":
        if not confirm_locked_run:
            raise RuntimeError(
                "Refusing to run confirmatory training without explicit --confirm-locked-run flag."
            )
        seeds, slice_info = resolve_locked_confirmatory_seeds(
            n_seeds=n_seeds,
            seed_offset=seed_offset,
            locked_n_total=PRIMARY_N_SEEDS,
            existing_seed_records=None,
        )
        epochs = 200
        run_tag = (
            f"confirmatory_n{n_seeds}"
            if seed_offset == 0
            else f"confirmatory_slice_{seeds[0]}_{seeds[-1]}"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    iso = verify_conf_key_isolation()
    assert iso["isolation_passed"], "PRNG isolation audit failed!"
    prov = _collect_tracked_hashes_and_env()
    spec = summarize_locked_spec()

    tcfg = get_locked_train_config()
    if epochs != tcfg.epochs:
        from dataclasses import replace
        tcfg = replace(tcfg, epochs=epochs)
    cfgs = get_locked_branch_configs()
    gym_cfg = cfgs["gym_cfg"]
    feat_dims = cfgs["feature_dims"]

    out_dir = CONF_DIR / "results" / run_tag
    ckpt_base = CONF_DIR / "checkpoints" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base.mkdir(parents=True, exist_ok=True)

    eval_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    reload_checks: List[Dict[str, Any]] = []
    seed_records: List[Dict[str, Any]] = []
    failed_seeds: List[Dict[str, Any]] = []
    dataset_hashes: Dict[str, Any] = {}

    t_global = time.time()
    final_ckpt_label = "ep200" if epochs == 200 else f"ep{epochs:03d}"

    for s in seeds:
        try:
            bundle = derive_conf_seed_bundle(s)
            train_d, val_d = build_train_and_val_for_seed(bundle)
            dataset_hashes[str(s)] = {
                "train": {k: hash_batch_spot(v) for k, v in train_d.items()},
                "val": {k: hash_batch_spot(v) for k, v in val_d.items()},
            }

            recs: Dict[str, Any] = {}
            for cname in PRIMARY_BRANCHES:
                rec = train_lr_config(
                    cname,
                    bundle,
                    train_d,
                    val_d,
                    gym_cfg,
                    tcfg,
                    feat_dims,
                    ckpt_base / f"seed_{s}" / cname,
                    save_extra_ckpts=False,
                )
                recs[cname] = rec
                p_disk, _ = reload_ckpt(Path(rec["ckpt_best_path"]))
                nom_idx = 0 if cname.startswith("A_") else 1
                r_disk = float(jnp.mean(rollout_fast(p_disk, gym_cfg, val_d["fixed_020"], nom_idx)["hedged_loss"]))
                diff = abs(r_disk - rec["val_nominal_hedged_at_best"])
                assert diff < 1e-6, f"Reload parity failed {s}/{cname}: {diff}"
                reload_checks.append({"seed": s, "config": cname, "val_nom_reload_diff": diff})

            # Selection is now SEALED. Build conf_test ONLY after training & selection complete.
            conf_test = build_conf_test_for_seed(bundle)
            dataset_hashes[str(s)]["conf_test"] = {k: hash_batch_spot(v) for k, v in conf_test.items()}

            for ck_kind in ("best_val", final_ckpt_label):
                p_a, obj_a = reload_ckpt(
                    Path(recs["A_DECAY"]["ckpt_best_path"] if ck_kind == "best_val" else recs["A_DECAY"]["final_opt_path"])
                )
                p_r2, obj_r2 = reload_ckpt(
                    Path(recs["R2_DECAY"]["ckpt_best_path"] if ck_kind == "best_val" else recs["R2_DECAY"]["final_opt_path"])
                )

                for sc, batch in conf_test.items():
                    roll_a = rollout_fast(p_a, gym_cfg, batch, 0)
                    m_a = evaluate_rollout_metrics_r3(roll_a, eps=EPS_NONINFERIORITY_OCE020)

                    roll_r2_nom = rollout_fast(p_r2, gym_cfg, batch, 1)
                    roll_r2_scen = (
                        rollout_fast(p_r2, gym_cfg, batch, TRAINING_SCENARIOS.index(sc))
                        if sc in TRAINING_SCENARIOS
                        else None
                    )
                    m_r2 = evaluate_rollout_metrics_r3(
                        roll_r2_nom,
                        rollout_scen=roll_r2_scen,
                        ref_hedged_oce_y_nom=m_a["hedged_oce_y_nom"],
                        eps=EPS_NONINFERIORITY_OCE020,
                    )

                    for cname, obj_c, p_c, m_c, nom_i in (
                        ("A_DECAY", obj_a, p_a, m_a, 0),
                        ("R2_DECAY", obj_r2, p_r2, m_r2, 1),
                    ):
                        eval_rows.append({
                            "seed": s,
                            "ckpt_kind": ck_kind,
                            "epoch": int(obj_c["epoch"]),
                            "best_epoch": recs[cname]["best_epoch"],
                            "best_epoch_if_val020_only": recs[cname]["best_epoch_if_val020_only"],
                            "config": cname,
                            "branch": recs[cname]["branch"],
                            "schedule": "DECAY",
                            "scenario": sc,
                            "hedged_oce_y_nom": m_c["hedged_oce_y_nom"],
                            "hedged_oce_y_scen": m_c["hedged_oce_y_scen"],
                            "he_net_ddof0": m_c["he_net_ddof0"],
                            "he_net_ddof1": m_c["he_net_ddof1"],
                            "var95": m_c["var95"],
                            "cvar95": m_c["cvar95"],
                            "mean_gains": m_c["mean_gains"],
                            "turnover": m_c["turnover"],
                            "max_abs_delta": m_c["max_abs_delta"],
                            "learned_y_nom": float(np.asarray(p_c["oce"]["y"])[nom_i]),
                        })

                    d_oce = float(m_r2["hedged_oce_y_nom"] - m_a["hedged_oce_y_nom"])
                    d_he = float(m_r2["he_net_ddof0"] - m_a["he_net_ddof0"])
                    d_cvar95 = float(m_r2["cvar95"] - m_a["cvar95"])
                    d_turn = float(m_r2["turnover"] - m_a["turnover"])
                    d_oce_scen = (
                        float(m_r2["hedged_oce_y_scen"] - m_a["hedged_oce_y_nom"])
                        if np.isfinite(m_r2["hedged_oce_y_scen"])
                        else float("nan")
                    )
                    paired_rows.append({
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
                        "delta_cvar95_R2_minus_A": d_cvar95,
                        "A_turnover": m_a["turnover"],
                        "R2_turnover": m_r2["turnover"],
                        "delta_turnover_R2_minus_A": d_turn,
                    })

            seed_records.append({
                "seed": s,
                "status": "COMPLETED",
                "A_DECAY_time_sec": recs["A_DECAY"]["total_time_sec"],
                "R2_DECAY_time_sec": recs["R2_DECAY"]["total_time_sec"],
                "A_best_epoch": recs["A_DECAY"]["best_epoch"],
                "R2_best_epoch": recs["R2_DECAY"]["best_epoch"],
            })
        except Exception as exc:
            err_cat = classify_run_exception(exc)
            failed_seeds.append({"seed": s, "error_category": err_cat, "error": f"{type(exc).__name__}: {exc}"})
            seed_records.append({"seed": s, "status": "FAILED", "error_category": err_cat, "error": f"{type(exc).__name__}: {exc}"})
            for ck_kind in ("best_val", final_ckpt_label):
                for sc in CONF_EVAL_SCENARIOS:
                    paired_rows.append({
                        "seed": s,
                        "ckpt_kind": ck_kind,
                        "scenario": sc,
                        "A_he_net_ddof0": float("nan"),
                        "R2_he_net_ddof0": float("nan"),
                        "delta_he_R2_minus_A": float("nan"),
                        "A_hedged_oce_y_nom": float("nan"),
                        "R2_hedged_oce_y_nom": float("nan"),
                        "delta_oce_R2_minus_A": float("nan"),
                        "exceed_eps_oce": False,
                        "R2_hedged_oce_y_scen": float("nan"),
                        "delta_oce_y_scen_R2_minus_A": float("nan"),
                        "A_cvar95": float("nan"),
                        "R2_cvar95": float("nan"),
                        "delta_cvar95_R2_minus_A": float("nan"),
                        "A_turnover": float("nan"),
                        "R2_turnover": float("nan"),
                        "delta_turnover_R2_minus_A": float("nan"),
                    })

        # Flush incrementally after every seed so mid-run failures preserve complete ledger
        _write_csv(eval_rows, out_dir / "RESULTS_CONFIRMATORY_FULL_PRECISION.csv")
        _write_csv(paired_rows, out_dir / "PAIRED_CONFIRMATORY.csv")
        (out_dir / "SEED_LEDGER.json").write_text(
            json.dumps({"seed_records": seed_records, "failed_seeds": failed_seeds}, indent=2),
            encoding="utf-8",
        )

    if mode == "execute_confirmatory":
        # Always judge against the full locked N=24 cohort [4001..4024]; a partial slice cannot emit PASS!
        stats_summary = analyze_confirmatory_pairs(
            paired_rows,
            planned_seeds=list(CONF_RESERVED_SEEDS[:PRIMARY_N_SEEDS]),
            eps_margin=EPS_NONINFERIORITY_OCE020,
            alpha_one_sided=ALPHA_ONE_SIDED,
            require_complete_clean=True,
        )
    else:
        stats_summary = {
            "mode": mode,
            "note": "NON_CONFIRMATORY_TECH_SMOKE_STATISTICAL_TEST_NOT_APPLIED",
            "seeds": seeds,
            "epochs": epochs,
        }

    max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource is not None else -1
    summary = {
        "mode": mode,
        "run_tag": run_tag,
        "seeds": seeds,
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
    (out_dir / "CONFIRMATORY_STATS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Completed mode={mode}, out_dir={out_dir}, wall={summary['wall_time_sec']:.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirmatory runner for R2_matched vs A_ref (Cosine LR Decay)")
    parser.add_argument(
        "--mode",
        choices=("dry_run", "tech_smoke", "execute_confirmatory"),
        default="dry_run",
        help="Default is dry_run (zero training, zero test data simulation).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=PRIMARY_N_SEEDS,
        help="Number of reserved confirmatory seeds from [4001..4040] when --mode execute_confirmatory.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Optional slice offset in CONF_RESERVED_SEEDS (for sliced cluster execution).",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=2,
        help="Number of epochs when --mode tech_smoke (seed=3999 only).",
    )
    parser.add_argument(
        "--confirm-locked-run",
        action="store_true",
        help="Explicit safety switch required for --mode execute_confirmatory.",
    )
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
