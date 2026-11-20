"""
run_r3_pilot.py: Main execution script for R3 Development Smoke Test and 3-Seed Pilot.

Usage:
  python3 run_r3_pilot.py --mode smoke   # Runs pytest + 1-seed 5-epoch timing & reload verification
  python3 run_r3_pilot.py --mode pilot   # Runs locked 3-seed [3101, 3102, 3103] x 200-epoch pilot
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import csv
import json
import time
import pickle
import hashlib
import platform
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp

R3_DIR = Path(__file__).resolve().parent
EXP_DIR = R3_DIR.parent
sys.path.insert(0, str(R3_DIR))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "scripts" / "robust_deep_hedging_phase2"))

from robust_world import RobustWorldConfig, simulate_robust_world
from core.train import as_batch
from r3_keys import (
    TRAINING_SCENARIOS,
    TEST_SCENARIOS,
    VOL_SEED_OFFSET,
    verify_key_isolation,
)
from r3_models import make_canonical_configs, rollout_with_oce, count_param_breakdown, smooth_max_logsumexp
from r3_metrics import evaluate_rollout_metrics_r3
from r3_train import R3TrainConfig, R3Spec, train_single_stream_branch, train_r2_matched, train_r3_constrained


DEV_SEEDS = [3101, 3102, 3103]

# Two pre-locked R3 candidates from R3_DESIGN_LOCK.md
R3_SPECS = [
    R3Spec(name="R3_cfg1", tau=0.012, lam_max=10.0, eps=0.00010, nominal_idx=1),
    R3Spec(name="R3_cfg2", tau=0.005, lam_max=20.0, eps=0.00010, nominal_idx=1),
]

SCENARIO_SPECS: Dict[str, dict] = {
    "fixed_015": {"sigma_mode": "fixed", "rvol": 0.15, "category": "in_train_lower"},
    "fixed_020": {"sigma_mode": "fixed", "rvol": 0.20, "category": "in_train_nominal"},
    "fixed_025": {"sigma_mode": "fixed", "rvol": 0.25, "category": "in_train_upper"},
    "rs_015_025": {
        "sigma_mode": "regime_switch",
        "regime_split_step": 10,
        "regime_sigma_low": 0.15,
        "regime_sigma_high": 0.25,
        "category": "in_train_regime_switch",
    },
    "regime_switch_015_028": {
        "sigma_mode": "regime_switch",
        "regime_split_step": 10,
        "regime_sigma_low": 0.15,
        "regime_sigma_high": 0.28,
        "category": "ood_regime_switch_028",
    },
    "fixed_030": {"sigma_mode": "fixed", "rvol": 0.30, "category": "ood_high_030"},
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_batch_spot(batch: dict) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(batch["market"]["payoff"]).tobytes())
    h.update(np.asarray(batch["features"]["price"]).tobytes())
    return h.hexdigest()[:16]


def build_scenario_world(scen_name: str, rng_seed: int, samples: int = 5000) -> dict:
    sp = SCENARIO_SPECS[scen_name]
    if sp["sigma_mode"] == "fixed":
        w = simulate_robust_world(
            RobustWorldConfig(
                samples=samples,
                steps=20,
                sigma_mode="fixed",
                rvol=sp["rvol"],
                drift=0.0,
                cost_s=0.0,
                seed=rng_seed,
                normalize_batch_mean=True,
            )
        )
    elif sp["sigma_mode"] == "regime_switch":
        w = simulate_robust_world(
            RobustWorldConfig(
                samples=samples,
                steps=20,
                sigma_mode="regime_switch",
                regime_split_step=sp["regime_split_step"],
                regime_sigma_low=sp["regime_sigma_low"],
                regime_sigma_high=sp["regime_sigma_high"],
                drift=0.0,
                cost_s=0.0,
                seed=rng_seed,
                normalize_batch_mean=True,
            )
        )
    else:
        raise ValueError(sp["sigma_mode"])
    return as_batch(w)


def build_train_val_for_seed(bundle: dict) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    train_dict: Dict[str, dict] = {}
    val_dict: Dict[str, dict] = {}
    for sc in TRAINING_SCENARIOS:
        train_dict[sc] = build_scenario_world(sc, bundle["train_dW"][sc], samples=5000)
        val_dict[sc] = build_scenario_world(sc, bundle["val_dW"][sc], samples=5000)

    w_r2 = simulate_robust_world(
        RobustWorldConfig(
            samples=5000,
            steps=20,
            sigma_mode="static_path",
            sigma_min=0.15,
            sigma_max=0.25,
            vol_seed_offset=VOL_SEED_OFFSET,
            drift=0.0,
            cost_s=0.0,
            seed=bundle["train_dW"]["r2_static_path"],
            normalize_batch_mean=True,
        )
    )
    train_dict["r2_static_path"] = as_batch(w_r2)
    return train_dict, val_dict


def build_test_for_seed(bundle: dict) -> Dict[str, dict]:
    test_seed = bundle["test_dW_shared_pair"]
    return {sc: build_scenario_world(sc, test_seed, samples=5000) for sc in TEST_SCENARIOS}


def reload_ckpt_jnp(ckpt_path: str) -> Tuple[dict, dict]:
    obj = pickle.loads(Path(ckpt_path).read_bytes())
    params_jnp = jax.tree_util.tree_map(jnp.asarray, obj["params"])
    return params_jnp, obj


def evaluate_split_all_branches(
    split_name: str,
    seed_label: int,
    eval_batches: Dict[str, dict],
    branch_records: Dict[str, dict],
    gym_cfg,
    eps: float = 0.00010,
) -> List[dict]:
    """
    Evaluates reloaded checkpoints cleanly on a dataset split (`train`, `val`, or `test`).
    """
    rows: List[dict] = []
    # First compute A_ref's hedged_oce_y_nom on each scenario in this split
    aref_rec = branch_records["A_ref"]
    aref_params, _ = reload_ckpt_jnp(aref_rec["ckpt_best_path"])
    aref_y_nom_by_scen: Dict[str, float] = {}
    for sc, batch in eval_batches.items():
        r_ref = rollout_with_oce(aref_params, gym_cfg, batch, oce_idx=0)
        aref_y_nom_by_scen[sc] = float(jnp.mean(r_ref["hedged_loss"]))

    spec_lam = {"A_ref": 0.0, "R2": 0.0, "R2_matched": 0.0, "R3_cfg1": 10.0, "R3_cfg2": 20.0}

    for bname, rec in branch_records.items():
        params_jnp, ckpt_obj = reload_ckpt_jnp(rec["ckpt_best_path"])
        nom_oce_idx = 0 if bname in ("A_ref", "R2") else 1
        y_nom_val = params_jnp["oce"]["y"][nom_oce_idx]
        y0_nom_val = params_jnp["oce"]["y0"][nom_oce_idx]
        has_scen_y = bname in ("R2_matched", "R3_cfg1", "R3_cfg2")

        for sc, batch in eval_batches.items():
            roll_nom = rollout_with_oce(
                params_jnp, gym_cfg, batch,
                oce_idx=nom_oce_idx,
                eval_y_override=y_nom_val,
                eval_y0_override=y0_nom_val,
            )
            roll_scen = None
            if has_scen_y and sc in TRAINING_SCENARIOS:
                sc_idx = TRAINING_SCENARIOS.index(sc)
                roll_scen = rollout_with_oce(params_jnp, gym_cfg, batch, oce_idx=sc_idx)

            m = evaluate_rollout_metrics_r3(
                roll_nom,
                rollout_scen=roll_scen,
                ref_hedged_oce_y_nom=aref_y_nom_by_scen[sc],
                eps=eps,
                lam_penalty=spec_lam[bname],
            )
            row = {
                "split": split_name,
                "seed": seed_label,
                "branch": bname,
                "scenario": sc,
                "scenario_category": SCENARIO_SPECS[sc]["category"],
                "best_epoch": rec["best_epoch"],
                "feasible_val": rec["feasible_val"],
                "selection_status": rec["selection_status"],
                "train_time_sec": rec["total_time_sec"],
                "total_paths_processed": rec["total_paths_processed"],
                "total_update_steps": rec["total_update_steps"],
                "policy_total_params": rec["param_counts"]["policy_total_params"],
                "oce_total_params": rec["param_counts"]["oce_total_params"],
                "grand_total_params": rec["param_counts"]["grand_total_params"],
                "learned_y_nom": float(y_nom_val),
                "learned_y_vec": json.dumps([float(v) for v in np.asarray(params_jnp["oce"]["y"])]),
            }
            row.update(m)
            rows.append(row)
    return rows


def run(mode: str) -> None:
    epochs = 5 if mode == "smoke" else 200
    seeds = [DEV_SEEDS[0]] if mode == "smoke" else DEV_SEEDS

    if mode == "smoke":
        ckpt_dir = R3_DIR / "checkpoints" / "smoke"
        res_dir = R3_DIR / "results" / "smoke"
    else:
        ckpt_dir = R3_DIR / "checkpoints" / "pilot"
        res_dir = R3_DIR / "results" / "pilot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"R3 DEVELOPMENT EXPERIMENT | MODE={mode.upper()} | SEEDS={seeds} | EPOCHS={epochs}")
    print(f"JAX devices: {jax.devices()} | Platform: {platform.platform()}")
    print("=" * 80, flush=True)

    # 1. Verify PRNG key isolation BEFORE simulating or training anything
    iso = verify_key_isolation("R3_DEV", DEV_SEEDS)
    assert iso["isolation_passed"], f"PRNG isolation failure: {iso}"
    bundle_by_seed = {b["seed_label"]: b for b in iso["bundles"]}

    _, gym_cfg, feature_dims = make_canonical_configs(lmbda=10.0)
    train_cfg = R3TrainConfig(epochs=epochs)

    all_rows: List[dict] = []
    seed_summaries: List[dict] = []
    data_hashes: Dict[str, Any] = {}
    reload_checks: List[dict] = []

    t_global = time.time()
    for s in seeds:
        bundle = bundle_by_seed[s]
        print(f"\n>>> SEED {s}: Simulating Train & Val (Test NOT simulated until selection ends) <<<", flush=True)
        train_dict, val_dict = build_train_val_for_seed(bundle)
        train_list = [train_dict[sc] for sc in TRAINING_SCENARIOS]
        val_list = [val_dict[sc] for sc in TRAINING_SCENARIOS]

        data_hashes[str(s)] = {
            "train": {k: hash_batch_spot(v) for k, v in train_dict.items()},
            "val": {k: hash_batch_spot(v) for k, v in val_dict.items()},
        }

        branch_records: Dict[str, dict] = {}

        # Branch 1: A_ref (sigma=0.20)
        rec_a = train_single_stream_branch(
            "A_ref", bundle, train_dict["fixed_020"], val_dict, ("fixed_020",),
            gym_cfg, train_cfg, feature_dims, ckpt_dir
        )
        branch_records["A_ref"] = rec_a
        print(f"  [A_ref] best_ep={rec_a['best_epoch']} val_nom={rec_a['val_nominal_hedged_at_best']:.6f} time={rec_a['total_time_sec']:.2f}s", flush=True)

        # Branch 2: R2 (static_path U[0.15, 0.25])
        rec_r2 = train_single_stream_branch(
            "R2", bundle, train_dict["r2_static_path"], val_dict, TRAINING_SCENARIOS,
            gym_cfg, train_cfg, feature_dims, ckpt_dir
        )
        branch_records["R2"] = rec_r2
        print(f"  [R2] best_ep={rec_r2['best_epoch']} val_mean={rec_r2['best_val_hedged']:.6f} time={rec_r2['total_time_sec']:.2f}s", flush=True)

        # Branch 3: R2_matched (4 scenarios, uniform ERM)
        rec_r2m = train_r2_matched(
            bundle, train_list, val_list, gym_cfg, train_cfg, feature_dims, ckpt_dir
        )
        branch_records["R2_matched"] = rec_r2m
        print(f"  [R2_matched] best_ep={rec_r2m['best_epoch']} val_mean={rec_r2m['best_val_hedged']:.6f} time={rec_r2m['total_time_sec']:.2f}s", flush=True)

        # Branches 4 & 5: R3_cfg1 and R3_cfg2 (frozen A_ref from disk to prove zero coupling!)
        aref_frozen_disk, _ = reload_ckpt_jnp(rec_a["ckpt_best_path"])
        ref_val_nom = float(jnp.mean(rollout_with_oce(aref_frozen_disk, gym_cfg, val_dict["fixed_020"], oce_idx=0)["hedged_loss"]))
        assert abs(ref_val_nom - rec_a["val_nominal_hedged_at_best"]) < 1e-6

        for spec in R3_SPECS:
            rec_r3 = train_r3_constrained(
                spec, bundle, aref_frozen_disk, ref_val_nom,
                train_list, val_list, gym_cfg, train_cfg, feature_dims, ckpt_dir
            )
            branch_records[spec.name] = rec_r3
            print(
                f"  [{spec.name}] status={rec_r3['selection_status']} best_ep={rec_r3['best_epoch']} "
                f"n_feas={rec_r3['n_feasible_epochs']}/{epochs} val_dNom={rec_r3['val_delta_nom_at_best']:+.6f} "
                f"lam_end={rec_r3['final_lam_dual']:.3f} time={rec_r3['total_time_sec']:.2f}s",
                flush=True,
            )

        # Reload parity verification on Val before touching Test
        for bname, rec in branch_records.items():
            p_disk, ck = reload_ckpt_jnp(rec["ckpt_best_path"])
            nom_idx = 0 if bname in ("A_ref", "R2") else 1
            r_disk_nom = float(jnp.mean(rollout_with_oce(p_disk, gym_cfg, val_dict["fixed_020"], oce_idx=nom_idx)["hedged_loss"]))
            mem_nom = rec["history"]["val_hedged_per_scen"]["fixed_020"][rec["best_epoch"] - 1]
            diff = abs(r_disk_nom - mem_nom)
            assert diff < 1e-5, f"Reload parity failed for {s}/{bname}: {diff}"
            reload_checks.append({"seed": s, "branch": bname, "best_epoch": rec["best_epoch"], "val_nom_reload_diff": diff})

        # Evaluate Train & Val summaries
        all_rows.extend(evaluate_split_all_branches("train", s, {sc: train_dict[sc] for sc in TRAINING_SCENARIOS}, branch_records, gym_cfg))
        all_rows.extend(evaluate_split_all_branches("val", s, val_dict, branch_records, gym_cfg))

        # ONLY NOW generate Test split (strict post-selection isolation!) and evaluate
        test_dict = build_test_for_seed(bundle)
        data_hashes[str(s)]["test"] = {k: hash_batch_spot(v) for k, v in test_dict.items()}
        all_rows.extend(evaluate_split_all_branches("test", s, test_dict, branch_records, gym_cfg))

        seed_summaries.append({
            "seed": s,
            "branches": {
                bname: {k: v for k, v in rec.items() if k != "best_params"}
                for bname, rec in branch_records.items()
            },
        })

    total_wall = float(time.time() - t_global)

    # Write RESULTS CSV (full precision!)
    csv_path = (res_dir / "RESULTS_R3_SMOKE.csv") if mode == "smoke" else (R3_DIR / "RESULTS_R3_FULL_PRECISION.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    summary = {
        "mode": mode,
        "seeds": seeds,
        "epochs": epochs,
        "total_wall_clock_sec": total_wall,
        "jax_devices": [str(d) for d in jax.devices()],
        "reload_checks": reload_checks,
        "max_reload_diff": float(max(x["val_nom_reload_diff"] for x in reload_checks)),
        "seed_summaries": seed_summaries,
    }
    with open(res_dir / f"r3_{mode}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if mode == "pilot":
        code_files = [
            R3_DIR / "R3_DESIGN_LOCK.md",
            R3_DIR / "R3_IMPLEMENTATION_NOTES.md",
            R3_DIR / "r3_keys.py",
            R3_DIR / "r3_models.py",
            R3_DIR / "r3_metrics.py",
            R3_DIR / "r3_train.py",
            R3_DIR / "run_r3_pilot.py",
            R3_DIR / "tests" / "test_r3_suite.py",
        ]
        manifest = {
            "experiment": "robust_r3_dev_pilot",
            "dev_seeds": DEV_SEEDS,
            "epochs": epochs,
            "r3_specs": [{"name": sp.name, "tau": sp.tau, "lam_max": sp.lam_max, "eps": sp.eps} for sp in R3_SPECS],
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "jax_version": jax.__version__,
                "numpy_version": np.__version__,
                "jax_devices": [str(d) for d in jax.devices()],
            },
            "code_sha256": {str(p.relative_to(R3_DIR)): sha256_file(p) for p in code_files if p.exists()},
            "prng_isolation_audit": {k: v for k, v in iso.items() if k != "bundles"},
            "seed_bundles": iso["bundles"],
            "dataset_spot_hashes": data_hashes,
        }
        with open(R3_DIR / "R3_DATA_AND_SEED_MANIFEST.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"COMPLETED {mode.upper()} IN {total_wall:.2f}s | MAX RELOAD DIFF = {summary['max_reload_diff']:.3e}")
    print(f"CSV written: {csv_path}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args()
    run(args.mode)
