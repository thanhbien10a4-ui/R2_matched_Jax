"""
robust_r2_matched_cost_confirmatory/cost_train.py:
Audited training engine for A_DECAY (A_ref_128) and R2_DECAY (R2_matched_128)
under locked transaction cost c_s = 0.0010 (10 bps) in training & validation selection.
"""

from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import jax
import jax.numpy as jnp

COST_CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = COST_CONF_DIR.parent
LR_DIR = EXP_DIR / "robust_r3_lr_stability"
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"

for _p in (str(COST_CONF_DIR), str(LR_DIR), str(BC_DIR), str(R3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from r3_keys import TRAINING_SCENARIOS
from r3_train import R3TrainConfig, adam_init, _take_batch, _atomic_pickle_dump, verify_finite_tree
from bc_models import (
    GymConfigOriginal,
    init_full_params,
    count_param_breakdown,
    rollout_with_oce,
    aref_128_step_loss,
    r2_matched_step_loss,
)
from lr_schedule import STEPS_PER_EPOCH, compute_lr_float
from lr_train import adam_update_scheduled
from cost_design import COST_S_LOCKED


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _run_epoch_compiled(
    carry: tuple,
    gym_cfg: GymConfigOriginal,
    train_cfg: R3TrainConfig,
    is_a: bool,
    sub_train_batches: List[dict],
    perm_stacked: jnp.ndarray,
) -> tuple:
    loss_fn = aref_128_step_loss if is_a else r2_matched_step_loss

    def _scan_step(carry_in, perm_row):
        params_in, opt_in = carry_in
        batch_slices = [_take_batch(sub_train_batches[k], perm_row[k]) for k in range(4)]
        loss_val, grads = jax.value_and_grad(loss_fn)(params_in, gym_cfg, batch_slices)
        new_params, new_opt = adam_update_scheduled(train_cfg, "DECAY", grads, opt_in, params_in)
        return (new_params, new_opt), loss_val

    return jax.lax.scan(_scan_step, carry, perm_stacked)


@functools.partial(jax.jit, static_argnums=(1, 3))
def _eval_oce_compiled(params: dict, gym_cfg: GymConfigOriginal, batch: dict, oce_idx: int) -> dict:
    return rollout_with_oce(params, gym_cfg, batch, oce_idx=oce_idx)


def train_cost_config(
    config_name: str,
    seed_bundle: dict,
    train_data: Dict[str, dict],
    val_batches: Dict[str, dict],
    gym_cfg: GymConfigOriginal,
    train_cfg: R3TrainConfig,
    feature_dims: dict,
    ckpt_run_dir: Path,
) -> Dict[str, Any]:
    if config_name not in ("A_DECAY", "R2_DECAY"):
        raise ValueError(f"Unsupported locked cost config: {config_name}")
    is_a = config_name == "A_DECAY"
    seed_label = int(seed_bundle["seed_label"])
    ckpt_run_dir.mkdir(parents=True, exist_ok=True)

    if is_a:
        sub_train_batches = [
            train_data["fixed_020_sub0"],
            train_data["fixed_020"],
            train_data["fixed_020_sub2"],
            train_data["fixed_020_sub3"],
        ]
        perm_keys = [
            seed_bundle["train_perm"]["fixed_020_sub0"],
            seed_bundle["train_perm"]["fixed_020"],
            seed_bundle["train_perm"]["fixed_020_sub2"],
            seed_bundle["train_perm"]["fixed_020_sub3"],
        ]
        k_oce = 1
    else:
        sub_train_batches = [train_data[sc] for sc in TRAINING_SCENARIOS]
        perm_keys = [seed_bundle["train_perm"][sc] for sc in TRAINING_SCENARIOS]
        k_oce = len(TRAINING_SCENARIOS)

    # Runtime verification that training and validation worlds carry positive proportional fee when COST_S_LOCKED > 0
    train_cost_means = {sc: float(np.mean(np.asarray(b["market"]["cost"]))) for sc, b in train_data.items()}
    val_cost_means = {sc: float(np.mean(np.asarray(b["market"]["cost"]))) for sc, b in val_batches.items()}
    if COST_S_LOCKED > 0.0:
        assert all(v > 0.0 for v in train_cost_means.values()), f"Zero cost in train_data: {train_cost_means}"
        assert all(v > 0.0 for v in val_cost_means.values()), f"Zero cost in val_batches: {val_cost_means}"

    bs_sub = int(train_cfg.batch_size)
    n_sub = int(sub_train_batches[0]["market"]["payoff"].shape[0])
    assert n_sub // bs_sub == STEPS_PER_EPOCH, f"Expected {STEPS_PER_EPOCH} steps/epoch, got {n_sub // bs_sub}"

    init_key = jax.random.PRNGKey(int(seed_bundle["init_policy"]))
    params = init_full_params(init_key, gym_cfg, feature_dims, k_oce=k_oce)
    opt = adam_init(params)
    carry = (params, opt)

    best_epoch = -1
    best_val_metric = float("inf")
    best_params_np = None

    history: Dict[str, Any] = {
        "epoch": [],
        "lr_end": [],
        "train_loss": [],
        "val_sel_metric": [],
        "val_hedged_per_scen": {sc: [] for sc in TRAINING_SCENARIOS},
        "val_cost_mean_020": [],
        "epoch_time_sec": [],
    }

    actual_update_steps = 0
    actual_paths_processed = 0
    t0_all = time.time()

    for ep in range(int(train_cfg.epochs)):
        t_ep = time.time()
        perms = [
            np.random.default_rng([int(perm_keys[k]), ep]).permutation(n_sub)[: STEPS_PER_EPOCH * bs_sub].reshape(
                STEPS_PER_EPOCH, bs_sub
            )
            for k in range(4)
        ]
        perm_stacked = jnp.asarray(np.stack(perms, axis=1))
        carry, ep_losses = _run_epoch_compiled(carry, gym_cfg, train_cfg, is_a, sub_train_batches, perm_stacked)
        actual_update_steps += int(perm_stacked.shape[0])
        actual_paths_processed += int(perm_stacked.shape[0] * perm_stacked.shape[1] * perm_stacked.shape[2])

        cur_params = carry[0]
        verify_finite_tree(cur_params)
        mean_train_loss = float(jnp.mean(ep_losses))
        if not np.isfinite(mean_train_loss):
            raise FloatingPointError(f"Non-finite train loss at seed={seed_label}, config={config_name}, ep={ep + 1}")

        if is_a:
            roll_020 = _eval_oce_compiled(cur_params, gym_cfg, val_batches["fixed_020"], 0)
            v020 = float(jnp.mean(roll_020["hedged_loss"]))
            val_cost_020 = float(jnp.mean(roll_020["cost"]))
            if not np.isfinite(v020):
                raise FloatingPointError(f"Non-finite val loss at seed={seed_label}, config={config_name}, ep={ep + 1}")
            for sc in TRAINING_SCENARIOS:
                history["val_hedged_per_scen"][sc].append(v020 if sc == "fixed_020" else float("nan"))
            sel_metric = v020
        else:
            sc_vals: List[float] = []
            val_cost_020 = 0.0
            for idx_sc, sc in enumerate(TRAINING_SCENARIOS):
                roll_sc = _eval_oce_compiled(cur_params, gym_cfg, val_batches[sc], idx_sc)
                v_sc = float(jnp.mean(roll_sc["hedged_loss"]))
                if not np.isfinite(v_sc):
                    raise FloatingPointError(
                        f"Non-finite val loss at seed={seed_label}, config={config_name}, ep={ep + 1}, sc={sc}"
                    )
                history["val_hedged_per_scen"][sc].append(v_sc)
                sc_vals.append(v_sc)
                if sc == "fixed_020":
                    val_cost_020 = float(jnp.mean(roll_sc["cost"]))
            sel_metric = float(np.mean(sc_vals))

        ep_num = ep + 1
        history["epoch"].append(ep_num)
        history["lr_end"].append(compute_lr_float(ep_num * STEPS_PER_EPOCH, "DECAY"))
        history["train_loss"].append(mean_train_loss)
        history["val_sel_metric"].append(sel_metric)
        history["val_cost_mean_020"].append(val_cost_020)
        history["epoch_time_sec"].append(float(time.time() - t_ep))

        if sel_metric < best_val_metric:
            best_val_metric = sel_metric
            best_epoch = ep_num
            best_params_np = jax.tree_util.tree_map(np.asarray, cur_params)

    total_time = float(time.time() - t0_all)
    expected_steps = int(train_cfg.epochs) * STEPS_PER_EPOCH
    expected_paths = expected_steps * 4 * bs_sub
    assert actual_update_steps == expected_steps, f"Step mismatch: {actual_update_steps} != {expected_steps}"
    assert actual_paths_processed == expected_paths, f"Path mismatch: {actual_paths_processed} != {expected_paths}"
    assert best_epoch >= 1 and best_params_np is not None

    best_ckpt_path = ckpt_run_dir / "best_val.pkl"
    _atomic_pickle_dump(
        {
            "config_name": config_name,
            "seed_label": seed_label,
            "cost_s": COST_S_LOCKED,
            "epoch": best_epoch,
            "update_step": best_epoch * STEPS_PER_EPOCH,
            "lr_end": compute_lr_float(best_epoch * STEPS_PER_EPOCH, "DECAY"),
            "val_sel_metric": best_val_metric,
            "best_val_hedged": best_val_metric,
            "params": best_params_np,
        },
        best_ckpt_path,
    )

    final_params_np = jax.tree_util.tree_map(np.asarray, carry[0])
    final_opt_np = jax.tree_util.tree_map(np.asarray, carry[1])
    final_opt_path = ckpt_run_dir / "final_with_opt.pkl"
    _atomic_pickle_dump(
        {
            "config_name": config_name,
            "seed_label": seed_label,
            "cost_s": COST_S_LOCKED,
            "epoch": int(train_cfg.epochs),
            "update_step": actual_update_steps,
            "lr_end": compute_lr_float(actual_update_steps, "DECAY"),
            "params": final_params_np,
            "opt": final_opt_np,
            "history": history,
        },
        final_opt_path,
    )

    val_020_curve = history["val_hedged_per_scen"]["fixed_020"]
    best_ep_val020 = int(np.argmin(val_020_curve)) + 1

    ckpt_manifest = {
        "config_name": config_name,
        "seed_label": seed_label,
        "cost_s": COST_S_LOCKED,
        "best_epoch": best_epoch,
        "best_epoch_if_val020_only": best_ep_val020,
        "best_ckpt_file": best_ckpt_path.name,
        "final_with_opt_file": final_opt_path.name,
        "actual_update_steps": actual_update_steps,
        "actual_paths_processed": actual_paths_processed,
    }
    (ckpt_run_dir / "checkpoint_manifest.json").write_text(json.dumps(ckpt_manifest, indent=2), encoding="utf-8")

    return {
        "config_name": config_name,
        "branch": "A_ref_128" if is_a else "R2_matched_128",
        "schedule": "DECAY",
        "seed_label": seed_label,
        "param_counts": count_param_breakdown(best_params_np),
        "distinct_train_paths": 20000,
        "paths_per_step": 128,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "actual_update_steps": actual_update_steps,
        "actual_paths_processed": actual_paths_processed,
        "total_time_sec": total_time,
        "best_epoch": best_epoch,
        "best_epoch_if_val020_only": best_ep_val020,
        "best_val_hedged": best_val_metric,
        "val_nominal_hedged_at_best": float(val_020_curve[best_epoch - 1]),
        "val_cost_mean_020_at_best": float(history["val_cost_mean_020"][best_epoch - 1]),
        "train_cost_means": train_cost_means,
        "val_cost_means": val_cost_means,
        "ckpt_best_path": str(best_ckpt_path),
        "final_opt_path": str(final_opt_path),
        "checkpoint_manifest": ckpt_manifest,
        "history": history,
    }
