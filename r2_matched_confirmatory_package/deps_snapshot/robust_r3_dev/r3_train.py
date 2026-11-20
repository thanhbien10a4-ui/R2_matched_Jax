"""
r3_train.py: Unified Training Engine for A_ref, R2, R2_matched, and R3 (EMA Primal-Dual).

Budget accounting & methodological strictness:
- Drops 8-path tail batch uniformly across all branches: 156 full batches x 32 = 4,992 paths/scen/epoch.
- Validation selection uses ONLY hedged OCE risk (-utility), strictly excluding -u0 and Test sets.
- R3 selects checkpoints minimizing R_tau_val among Val-feasible epochs (delta_nom <= eps);
  if infeasible across all epochs, marks `feasible = False` (`INFEASIBLE_FALLBACK`) without relaxing eps.
"""

from dataclasses import dataclass, replace
from typing import Dict, Any, List, Tuple, Optional
import time
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from r3_keys import TRAINING_SCENARIOS
from r3_models import (
    GymConfigOriginal,
    init_full_params,
    count_param_breakdown,
    rollout_with_oce,
    smooth_max_logsumexp,
    single_scenario_step_loss,
    r2_matched_step_loss,
    r3_constrained_step_loss,
)

BATCH_SIZE = 32
N_SAMPLES_TRAIN = 5000
N_FULL_BATCHES = N_SAMPLES_TRAIN // BATCH_SIZE  # 156 batches -> 4,992 paths/scen/epoch (tail 8 dropped)


@dataclass(frozen=True)
class R3TrainConfig:
    epochs: int = 200
    batch_size: int = BATCH_SIZE
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    ema_beta: float = 0.05
    eta_dual: float = 0.02


@dataclass(frozen=True)
class R3Spec:
    name: str
    tau: float
    lam_max: float
    eps: float
    nominal_idx: int = 1  # index of fixed_020 inside TRAINING_SCENARIOS


def adam_init(params: dict) -> dict:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=zeros, v=zeros, t=jnp.zeros((), dtype=jnp.float32))


def adam_update(cfg: R3TrainConfig, grads: dict, opt: dict, params: dict) -> Tuple[dict, dict]:
    t = opt["t"] + 1.0
    alpha = cfg.learning_rate * jnp.sqrt(1.0 - cfg.beta_2 ** t) / (1.0 - cfg.beta_1 ** t)
    m = jax.tree_util.tree_map(lambda m, g: m + (g - m) * (1.0 - cfg.beta_1), opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: v + (g * g - v) * (1.0 - cfg.beta_2), opt["v"], grads)
    new_params = jax.tree_util.tree_map(
        lambda p, m, v: p - m * alpha / (jnp.sqrt(v) + cfg.epsilon), params, m, v
    )
    return new_params, dict(m=m, v=v, t=t)


def _take_batch(data: dict, idx: jnp.ndarray) -> dict:
    return jax.tree_util.tree_map(lambda x: x[idx], data)


def _atomic_pickle_dump(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def verify_finite_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if not np.all(np.isfinite(np.asarray(leaf))):
            raise FloatingPointError("Non-finite NaN/Inf detected in parameters!")


def train_single_stream_branch(
    branch_name: str,
    seed_bundle: dict,
    train_batch: dict,
    val_batches: Dict[str, dict],
    val_eval_scenarios: Tuple[str, ...],
    gym_cfg: GymConfigOriginal,
    train_cfg: R3TrainConfig,
    feature_dims: dict,
    ckpt_dir: Path,
) -> Dict[str, Any]:
    """
    Trains `A_ref` (on fixed_020, validated on fixed_020) or `R2` (on static_path U[0.15, 0.25],
    validated on mean of 4 Val scenarios) using hedged OCE (-u) only for checkpoint selection.
    """
    perm_key_name = "fixed_020" if branch_name == "A_ref" else "r2_static_path"
    perm_seed = seed_bundle["train_perm"][perm_key_name]

    rng_key = jax.random.PRNGKey(seed_bundle["init_policy"])
    params = init_full_params(rng_key, gym_cfg, feature_dims, k_oce=1, n_inst=1)
    opt = adam_init(params)

    @jax.jit
    def run_epoch(carry, perm_matrix):
        def _step(cr, idx):
            p, o = cr
            b = _take_batch(train_batch, idx)
            lval, grads = jax.value_and_grad(single_scenario_step_loss)(p, gym_cfg, b)
            p_next, o_next = adam_update(train_cfg, grads, o, p)
            return (p_next, o_next), lval
        return jax.lax.scan(_step, carry, perm_matrix)

    @jax.jit
    def eval_val(p, b):
        r = rollout_with_oce(p, gym_cfg, b, oce_idx=0)
        return r["hedged_loss_mean"], r["total_loss_mean"]

    best_val_hedged = float("inf")
    best_epoch = -1
    best_params = None
    history = {
        "train_loss_step_mean": [],
        "val_hedged_mean": [],
        "val_orig_mean": [],
        "val_hedged_per_scen": {sc: [] for sc in val_eval_scenarios},
        "epoch_time_sec": [],
    }

    seed_label = seed_bundle["seed_label"]
    ckpt_best = ckpt_dir / f"seed_{seed_label}_branch_{branch_name}_best.pkl"
    ckpt_latest = ckpt_dir / f"seed_{seed_label}_branch_{branch_name}_latest.pkl"

    t0_all = time.time()
    carry = (params, opt)
    n_train = int(train_batch["market"]["payoff"].shape[0])

    for ep in range(train_cfg.epochs):
        t0 = time.time()
        perm = np.random.default_rng([perm_seed, ep]).permutation(n_train)
        perm_mat = jnp.asarray(perm[: N_FULL_BATCHES * BATCH_SIZE].reshape(N_FULL_BATCHES, BATCH_SIZE))
        carry, step_losses = run_epoch(carry, perm_mat)

        val_h_list, val_o_list = [], []
        for sc in val_eval_scenarios:
            vh, vo = eval_val(carry[0], val_batches[sc])
            vh_f, vo_f = float(vh), float(vo)
            history["val_hedged_per_scen"][sc].append(vh_f)
            val_h_list.append(vh_f)
            val_o_list.append(vo_f)

        mean_vh = float(np.mean(val_h_list))
        mean_vo = float(np.mean(val_o_list))
        history["train_loss_step_mean"].append(float(jnp.mean(step_losses)))
        history["val_hedged_mean"].append(mean_vh)
        history["val_orig_mean"].append(mean_vo)
        history["epoch_time_sec"].append(float(time.time() - t0))

        if mean_vh < best_val_hedged:
            best_val_hedged = mean_vh
            best_epoch = ep + 1
            best_params = jax.tree_util.tree_map(lambda x: np.asarray(x), carry[0])
            _atomic_pickle_dump(
                {
                    "branch": branch_name,
                    "seed_label": seed_label,
                    "epoch": best_epoch,
                    "best_val_hedged": best_val_hedged,
                    "selection_status": "FEASIBLE",
                    "params": best_params,
                },
                ckpt_best,
            )

    verify_finite_tree(carry[0])
    total_time = float(time.time() - t0_all)
    latest_params = jax.tree_util.tree_map(lambda x: np.asarray(x), carry[0])
    _atomic_pickle_dump(
        {"branch": branch_name, "seed_label": seed_label, "epoch": train_cfg.epochs, "params": latest_params, "history": history},
        ckpt_latest,
    )

    best_params_jnp = jax.tree_util.tree_map(jnp.asarray, best_params)
    return {
        "branch": branch_name,
        "seed_label": seed_label,
        "param_counts": count_param_breakdown(best_params),
        "paths_per_epoch": N_FULL_BATCHES * BATCH_SIZE,
        "total_paths_processed": train_cfg.epochs * N_FULL_BATCHES * BATCH_SIZE,
        "update_steps_per_epoch": N_FULL_BATCHES,
        "total_update_steps": train_cfg.epochs * N_FULL_BATCHES,
        "rollouts_per_step": 1 * BATCH_SIZE,
        "total_rollouts": train_cfg.epochs * N_FULL_BATCHES * BATCH_SIZE,
        "total_time_sec": total_time,
        "best_epoch": best_epoch,
        "feasible_val": True,
        "selection_status": "FEASIBLE",
        "best_val_hedged": best_val_hedged,
        "val_nominal_hedged_at_best": float(history["val_hedged_per_scen"].get("fixed_020", [float("nan")])[best_epoch - 1]),
        "best_params": best_params_jnp,
        "ckpt_best_path": str(ckpt_best),
        "history": history,
    }


def train_r2_matched(
    seed_bundle: dict,
    train_batches: List[dict],
    val_batches: List[dict],
    gym_cfg: GymConfigOriginal,
    train_cfg: R3TrainConfig,
    feature_dims: dict,
    ckpt_dir: Path,
) -> Dict[str, Any]:
    """Trains R2-matched on the exact same K=4 scenarios with uniform ERM objective and y_k (shape 4,)."""
    k_scen = len(TRAINING_SCENARIOS)
    perm_seeds = [seed_bundle["train_perm"][sc] for sc in TRAINING_SCENARIOS]

    rng_key = jax.random.PRNGKey(seed_bundle["init_policy"])
    params = init_full_params(rng_key, gym_cfg, feature_dims, k_oce=k_scen, n_inst=1)
    opt = adam_init(params)

    @jax.jit
    def run_epoch(carry, perm_matrices):
        def _step(cr, idx_tuple):
            p, o = cr
            bs = [_take_batch(train_batches[k], idx_tuple[k]) for k in range(k_scen)]
            lval, grads = jax.value_and_grad(r2_matched_step_loss)(p, gym_cfg, bs)
            p_next, o_next = adam_update(train_cfg, grads, o, p)
            return (p_next, o_next), lval
        return jax.lax.scan(_step, carry, perm_matrices)

    @jax.jit
    def eval_all_val(p):
        vh_list, vo_list = [], []
        for k in range(k_scen):
            r = rollout_with_oce(p, gym_cfg, val_batches[k], oce_idx=k)
            vh_list.append(r["hedged_loss_mean"])
            vo_list.append(r["total_loss_mean"])
        return jnp.stack(vh_list), jnp.stack(vo_list)

    best_val_hedged = float("inf")
    best_epoch = -1
    best_params = None
    history = {
        "train_loss_step_mean": [],
        "val_hedged_mean": [],
        "val_orig_mean": [],
        "val_hedged_per_scen": {sc: [] for sc in TRAINING_SCENARIOS},
        "epoch_time_sec": [],
    }

    seed_label = seed_bundle["seed_label"]
    ckpt_best = ckpt_dir / f"seed_{seed_label}_branch_R2_matched_best.pkl"
    ckpt_latest = ckpt_dir / f"seed_{seed_label}_branch_R2_matched_latest.pkl"

    t0_all = time.time()
    carry = (params, opt)
    n_train = int(train_batches[0]["market"]["payoff"].shape[0])

    for ep in range(train_cfg.epochs):
        t0 = time.time()
        perms = []
        for k in range(k_scen):
            pm = np.random.default_rng([perm_seeds[k], ep]).permutation(n_train)
            perms.append(jnp.asarray(pm[: N_FULL_BATCHES * BATCH_SIZE].reshape(N_FULL_BATCHES, BATCH_SIZE)))
        # shape: (N_FULL_BATCHES, k_scen, BATCH_SIZE) -> tuple per scan step via swapaxes
        perm_stacked = jnp.stack(perms, axis=1)
        carry, step_losses = run_epoch(carry, perm_stacked)

        vh_vec, vo_vec = eval_all_val(carry[0])
        vh_np, vo_np = np.asarray(vh_vec, dtype=np.float64), np.asarray(vo_vec, dtype=np.float64)
        for k, sc in enumerate(TRAINING_SCENARIOS):
            history["val_hedged_per_scen"][sc].append(float(vh_np[k]))

        mean_vh = float(np.mean(vh_np))
        history["train_loss_step_mean"].append(float(jnp.mean(step_losses)))
        history["val_hedged_mean"].append(mean_vh)
        history["val_orig_mean"].append(float(np.mean(vo_np)))
        history["epoch_time_sec"].append(float(time.time() - t0))

        if mean_vh < best_val_hedged:
            best_val_hedged = mean_vh
            best_epoch = ep + 1
            best_params = jax.tree_util.tree_map(lambda x: np.asarray(x), carry[0])
            _atomic_pickle_dump(
                {
                    "branch": "R2_matched",
                    "seed_label": seed_label,
                    "epoch": best_epoch,
                    "best_val_hedged": best_val_hedged,
                    "selection_status": "FEASIBLE",
                    "params": best_params,
                },
                ckpt_best,
            )

    verify_finite_tree(carry[0])
    total_time = float(time.time() - t0_all)
    latest_params = jax.tree_util.tree_map(lambda x: np.asarray(x), carry[0])
    _atomic_pickle_dump(
        {"branch": "R2_matched", "seed_label": seed_label, "epoch": train_cfg.epochs, "params": latest_params, "history": history},
        ckpt_latest,
    )

    best_params_jnp = jax.tree_util.tree_map(jnp.asarray, best_params)
    return {
        "branch": "R2_matched",
        "seed_label": seed_label,
        "param_counts": count_param_breakdown(best_params),
        "paths_per_epoch": k_scen * N_FULL_BATCHES * BATCH_SIZE,
        "total_paths_processed": train_cfg.epochs * k_scen * N_FULL_BATCHES * BATCH_SIZE,
        "update_steps_per_epoch": N_FULL_BATCHES,
        "total_update_steps": train_cfg.epochs * N_FULL_BATCHES,
        "rollouts_per_step": k_scen * BATCH_SIZE,
        "total_rollouts": train_cfg.epochs * k_scen * N_FULL_BATCHES * BATCH_SIZE,
        "total_time_sec": total_time,
        "best_epoch": best_epoch,
        "feasible_val": True,
        "selection_status": "FEASIBLE",
        "best_val_hedged": best_val_hedged,
        "val_nominal_hedged_at_best": float(history["val_hedged_per_scen"]["fixed_020"][best_epoch - 1]),
        "best_params": best_params_jnp,
        "ckpt_best_path": str(ckpt_best),
        "history": history,
    }


def train_r3_constrained(
    spec: R3Spec,
    seed_bundle: dict,
    ref_params_frozen: dict,
    ref_val_nominal_hedged: float,
    train_batches: List[dict],
    val_batches: List[dict],
    gym_cfg: GymConfigOriginal,
    train_cfg: R3TrainConfig,
    feature_dims: dict,
    ckpt_dir: Path,
) -> Dict[str, Any]:
    """
    Trains R3 with Smooth-Max R_tau + Unbiased EMA Primal-Dual Nominal Constraint.
    `ref_params_frozen` is strictly frozen (stop_gradient + excluded from optimizer state).
    Selection rule:
      - Among epochs with val_nominal_hedged - ref_val_nominal_hedged <= eps:
        select epoch minimizing R_tau_val (`FEASIBLE`).
      - If NO epoch satisfies constraint: mark `feasible_val = False` (`INFEASIBLE_FALLBACK`)
        and save lexicographic minimum-violation fallback checkpoint.
    """
    k_scen = len(TRAINING_SCENARIOS)
    perm_seeds = [seed_bundle["train_perm"][sc] for sc in TRAINING_SCENARIOS]
    ref_frozen = jax.tree_util.tree_map(jax.lax.stop_gradient, ref_params_frozen)

    rng_key = jax.random.PRNGKey(seed_bundle["init_policy"])
    params = init_full_params(rng_key, gym_cfg, feature_dims, k_oce=k_scen, n_inst=1)
    opt = adam_init(params)

    ema_beta = jnp.asarray(train_cfg.ema_beta, dtype=jnp.float32)
    eta_dual = jnp.asarray(train_cfg.eta_dual, dtype=jnp.float32)
    eps_jnp = jnp.asarray(spec.eps, dtype=jnp.float32)
    lam_max_jnp = jnp.asarray(spec.lam_max, dtype=jnp.float32)

    @jax.jit
    def run_epoch(carry, perm_matrices):
        def _step(cr, idx_tuple):
            p, o, ema_d, lam_d = cr
            bs = [_take_batch(train_batches[k], idx_tuple[k]) for k in range(k_scen)]

            def _loss(pp):
                return r3_constrained_step_loss(
                    pp,
                    ref_frozen,
                    gym_cfg,
                    bs,
                    tau=spec.tau,
                    lam_dual=lam_d,
                    nominal_idx=spec.nominal_idx,
                )

            (_, aux), grads = jax.value_and_grad(_loss, has_aux=True)(p)
            p_next, o_next = adam_update(train_cfg, grads, o, p)
            new_ema = (1.0 - ema_beta) * ema_d + ema_beta * aux["batch_delta_nom"]
            new_lam = jnp.clip(lam_d + eta_dual * (new_ema - eps_jnp) / eps_jnp, 0.0, lam_max_jnp)
            return (p_next, o_next, new_ema, new_lam), aux

        return jax.lax.scan(_step, carry, perm_matrices)

    @jax.jit
    def eval_all_val(p):
        vh_list, vo_list = [], []
        for k in range(k_scen):
            r = rollout_with_oce(p, gym_cfg, val_batches[k], oce_idx=k)
            vh_list.append(r["hedged_loss_mean"])
            vo_list.append(r["total_loss_mean"])
        vh_vec = jnp.stack(vh_list)
        r_tau_val, w_val = smooth_max_logsumexp(vh_vec, tau=spec.tau)
        return vh_vec, jnp.stack(vo_list), r_tau_val, w_val

    best_feasible_rtau = float("inf")
    best_feasible_epoch = -1
    best_feasible_params = None

    best_fallback_delta = float("inf")
    best_fallback_epoch = -1
    best_fallback_params = None

    history = {
        "train_r_tau_step_mean": [],
        "train_delta_nom_step_mean": [],
        "ema_delta_end": [],
        "lam_dual_end": [],
        "train_weights_last": [],
        "val_r_tau": [],
        "val_max_hedged": [],
        "val_delta_nom": [],
        "val_violation": [],
        "val_feasible": [],
        "val_orig_mean": [],
        "val_hedged_per_scen": {sc: [] for sc in TRAINING_SCENARIOS},
        "epoch_time_sec": [],
    }

    seed_label = seed_bundle["seed_label"]
    ckpt_best = ckpt_dir / f"seed_{seed_label}_branch_{spec.name}_best.pkl"
    ckpt_latest = ckpt_dir / f"seed_{seed_label}_branch_{spec.name}_latest.pkl"

    t0_all = time.time()
    carry = (params, opt, jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32))
    n_train = int(train_batches[0]["market"]["payoff"].shape[0])

    for ep in range(train_cfg.epochs):
        t0 = time.time()
        perms = []
        for k in range(k_scen):
            pm = np.random.default_rng([perm_seeds[k], ep]).permutation(n_train)
            perms.append(jnp.asarray(pm[: N_FULL_BATCHES * BATCH_SIZE].reshape(N_FULL_BATCHES, BATCH_SIZE)))
        perm_stacked = jnp.stack(perms, axis=1)
        carry, aux_hist = run_epoch(carry, perm_stacked)

        vh_vec, vo_vec, r_tau_val, w_val = eval_all_val(carry[0])
        vh_np = np.asarray(vh_vec, dtype=np.float64)
        vo_np = np.asarray(vo_vec, dtype=np.float64)
        rtau_f = float(r_tau_val)
        l_nom_val = float(vh_np[spec.nominal_idx])
        delta_nom_val = float(l_nom_val - ref_val_nominal_hedged)
        violation_val = float(max(0.0, delta_nom_val - spec.eps))
        is_feas = bool(delta_nom_val <= spec.eps)

        for k, sc in enumerate(TRAINING_SCENARIOS):
            history["val_hedged_per_scen"][sc].append(float(vh_np[k]))

        history["train_r_tau_step_mean"].append(float(jnp.mean(aux_hist["r_tau"])))
        history["train_delta_nom_step_mean"].append(float(jnp.mean(aux_hist["batch_delta_nom"])))
        history["ema_delta_end"].append(float(carry[2]))
        history["lam_dual_end"].append(float(carry[3]))
        history["train_weights_last"].append([float(x) for x in aux_hist["weights"][-1]])
        history["val_r_tau"].append(rtau_f)
        history["val_max_hedged"].append(float(np.max(vh_np)))
        history["val_delta_nom"].append(delta_nom_val)
        history["val_violation"].append(violation_val)
        history["val_feasible"].append(is_feas)
        history["val_orig_mean"].append(float(np.mean(vo_np)))
        history["epoch_time_sec"].append(float(time.time() - t0))

        # Track lexicographic minimum-violation fallback regardless of feasibility
        if delta_nom_val < best_fallback_delta:
            best_fallback_delta = delta_nom_val
            best_fallback_epoch = ep + 1
            best_fallback_params = jax.tree_util.tree_map(lambda x: np.asarray(x), carry[0])

        # Track constrained feasible minimum R_tau
        if is_feas and rtau_f < best_feasible_rtau:
            best_feasible_rtau = rtau_f
            best_feasible_epoch = ep + 1
            best_feasible_params = jax.tree_util.tree_map(lambda x: np.asarray(x), carry[0])

    verify_finite_tree(carry[0])
    total_time = float(time.time() - t0_all)

    if best_feasible_params is not None:
        feasible_overall = True
        sel_status = "FEASIBLE"
        best_epoch = best_feasible_epoch
        chosen_params = best_feasible_params
    else:
        feasible_overall = False
        sel_status = "INFEASIBLE_FALLBACK"
        best_epoch = best_fallback_epoch
        chosen_params = best_fallback_params

    _atomic_pickle_dump(
        {
            "branch": spec.name,
            "seed_label": seed_label,
            "epoch": best_epoch,
            "feasible_val": feasible_overall,
            "selection_status": sel_status,
            "val_r_tau_at_best": history["val_r_tau"][best_epoch - 1],
            "val_delta_nom_at_best": history["val_delta_nom"][best_epoch - 1],
            "val_violation_at_best": history["val_violation"][best_epoch - 1],
            "n_feasible_epochs": int(sum(history["val_feasible"])),
            "spec": {"tau": spec.tau, "lam_max": spec.lam_max, "eps": spec.eps},
            "params": chosen_params,
        },
        ckpt_best,
    )

    latest_params = jax.tree_util.tree_map(lambda x: np.asarray(x), carry[0])
    _atomic_pickle_dump(
        {"branch": spec.name, "seed_label": seed_label, "epoch": train_cfg.epochs, "params": latest_params, "history": history},
        ckpt_latest,
    )

    chosen_params_jnp = jax.tree_util.tree_map(jnp.asarray, chosen_params)
    return {
        "branch": spec.name,
        "seed_label": seed_label,
        "param_counts": count_param_breakdown(chosen_params),
        "paths_per_epoch": k_scen * N_FULL_BATCHES * BATCH_SIZE,
        "total_paths_processed": train_cfg.epochs * k_scen * N_FULL_BATCHES * BATCH_SIZE,
        "update_steps_per_epoch": N_FULL_BATCHES,
        "total_update_steps": train_cfg.epochs * N_FULL_BATCHES,
        # Policy processes 4*32 rollouts + 1*32 frozen forward ref rollout per step
        "rollouts_per_step_policy": k_scen * BATCH_SIZE,
        "rollouts_per_step_with_frozen_ref": (k_scen + 1) * BATCH_SIZE,
        "total_rollouts_with_frozen_ref": train_cfg.epochs * N_FULL_BATCHES * (k_scen + 1) * BATCH_SIZE,
        "total_time_sec": total_time,
        "best_epoch": best_epoch,
        "feasible_val": feasible_overall,
        "selection_status": sel_status,
        "n_feasible_epochs": int(sum(history["val_feasible"])),
        "val_r_tau_at_best": float(history["val_r_tau"][best_epoch - 1]),
        "val_delta_nom_at_best": float(history["val_delta_nom"][best_epoch - 1]),
        "val_violation_at_best": float(history["val_violation"][best_epoch - 1]),
        "final_ema_delta": float(history["ema_delta_end"][-1]),
        "final_lam_dual": float(history["lam_dual_end"][-1]),
        "best_params": chosen_params_jnp,
        "ckpt_best_path": str(ckpt_best),
        "history": history,
    }
