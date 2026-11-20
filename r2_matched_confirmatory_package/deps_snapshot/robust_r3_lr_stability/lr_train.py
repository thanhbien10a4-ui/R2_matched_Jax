"""
lr_train.py: 2x2 Training Engine for `A_FIXED`, `A_DECAY`, `R2_FIXED`, `R2_DECAY`.

Strict Invariants:
- Reuses canonical `GymConfigOriginal`, `R3TrainConfig` (`lr=1e-3, b1=0.9, b2=0.999, eps=1e-7`),
  `init_full_params`, `aref_128_step_loss`, `r2_matched_step_loss`, and `rollout_with_oce`.
- All 4 configurations execute strictly 156 steps/epoch x 200 epochs = 31,200 update steps,
  128 paths/step (4 x 32), 20,000 distinct training paths, 3,993,600 processed path-rollouts.
- Checkpoints saved strictly without affecting carry/RNG/trajectory:
  * Epochs {160, 170, 180} and every epoch {181..200}.
  * `best_val` (deduplicated via checkpoint_manifest.json if best_epoch in extra epochs;
    written to `best_val.pkl` otherwise).
  * `final_with_opt.pkl` at epoch 200 storing params, optimizer state (m, v, t=31200), and full history.
"""

from pathlib import Path
from typing import Dict, Any, List
import sys
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp

LR_DIR = Path(__file__).resolve().parent
EXP_DIR = LR_DIR.parent
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"
for p in (str(LR_DIR), str(BC_DIR), str(R3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

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
from lr_schedule import STEPS_PER_EPOCH, compute_lr_jnp, compute_lr_float

EXTRA_CKPT_EPOCHS = (160, 170, 180) + tuple(range(181, 201))


def adam_update_scheduled(cfg: R3TrainConfig, schedule_mode: str, grads: dict, opt: dict, params: dict):
    """Adam step matching r3_train.adam_update identically for FIXED, with step-based LR for DECAY."""
    t = opt["t"] + 1.0
    lr_t = compute_lr_jnp(t, schedule_mode)
    alpha = lr_t * jnp.sqrt(1.0 - cfg.beta_2 ** t) / (1.0 - cfg.beta_1 ** t)
    m = jax.tree_util.tree_map(lambda m_arr, g: m_arr + (g - m_arr) * (1.0 - cfg.beta_1), opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v_arr, g: v_arr + (g * g - v_arr) * (1.0 - cfg.beta_2), opt["v"], grads)
    new_params = jax.tree_util.tree_map(
        lambda p, m_arr, v_arr: p - m_arr * alpha / (jnp.sqrt(v_arr) + cfg.epsilon),
        params, m, v,
    )
    return new_params, dict(m=m, v=v, t=t)


def train_lr_config(
    config_name: str,          # "A_FIXED", "A_DECAY", "R2_FIXED", "R2_DECAY"
    seed_bundle: dict,
    train_data: dict,          # For A_*: list of 4 sub_batches; for R2_*: list of 4 scenario batches
    val_batches: Dict[str, dict],
    gym_cfg: GymConfigOriginal,
    train_cfg: R3TrainConfig,
    feature_dims: dict,
    ckpt_run_dir: Path,
    save_extra_ckpts: bool = True,
) -> Dict[str, Any]:
    is_a = config_name.startswith("A_")
    schedule_mode = "DECAY" if config_name.endswith("_DECAY") else "FIXED"
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
        val_020 = val_batches["fixed_020"]
        k_oce = 1
    else:
        sub_train_batches = [train_data[sc] for sc in TRAINING_SCENARIOS]
        perm_keys = [seed_bundle["train_perm"][sc] for sc in TRAINING_SCENARIOS]
        val_list = [val_batches[sc] for sc in TRAINING_SCENARIOS]
        k_oce = len(TRAINING_SCENARIOS)

    bs_sub = 32
    n_sub = int(sub_train_batches[0]["market"]["payoff"].shape[0])
    assert n_sub == 5000

    rng_key = jax.random.PRNGKey(seed_bundle["init_policy"])
    params = init_full_params(rng_key, gym_cfg, feature_dims, k_oce=k_oce, n_inst=1)
    opt = adam_init(params)

    loss_fn = aref_128_step_loss if is_a else r2_matched_step_loss

    @jax.jit
    def run_epoch(carry, perm_matrices):
        def _step(cr, idx_tuple):
            p, o = cr
            bs = [_take_batch(sub_train_batches[k], idx_tuple[k]) for k in range(4)]
            lval, grads = jax.value_and_grad(loss_fn)(p, gym_cfg, bs)
            p_next, o_next = adam_update_scheduled(train_cfg, schedule_mode, grads, o, p)
            return (p_next, o_next), lval
        return jax.lax.scan(_step, carry, perm_matrices)

    if is_a:
        @jax.jit
        def eval_val(p):
            r = rollout_with_oce(p, gym_cfg, val_020, oce_idx=0)
            return r["hedged_loss_mean"], r["total_loss_mean"]
    else:
        @jax.jit
        def eval_val(p):
            vh_l, vo_l = [], []
            for k in range(4):
                r = rollout_with_oce(p, gym_cfg, val_list[k], oce_idx=k)
                vh_l.append(r["hedged_loss_mean"])
                vo_l.append(r["total_loss_mean"])
            return jnp.stack(vh_l), jnp.stack(vo_l)

    history = {
        "train_loss": [],
        "lr_end_of_epoch": [],
        "val_hedged_mean": [],
        "val_orig_mean": [],
        "val_hedged_per_scen": {"fixed_020": []} if is_a else {sc: [] for sc in TRAINING_SCENARIOS},
        "epoch_time_sec": [],
    }

    best_val_metric = float("inf")
    best_epoch = -1
    best_params_np = None
    saved_epochs_map: Dict[str, str] = {}

    t0_all = time.time()
    carry = (params, opt)

    for ep in range(train_cfg.epochs):
        ep_num = ep + 1
        t0 = time.time()
        perms = []
        for k in range(4):
            pm = np.random.default_rng([perm_keys[k], ep]).permutation(n_sub)
            perms.append(jnp.asarray(pm[: STEPS_PER_EPOCH * bs_sub].reshape(STEPS_PER_EPOCH, bs_sub)))
        perm_stacked = jnp.stack(perms, axis=1)
        carry, step_losses = run_epoch(carry, perm_stacked)

        step_end = ep_num * STEPS_PER_EPOCH
        lr_end = compute_lr_float(step_end, schedule_mode)
        history["train_loss"].append(float(jnp.mean(step_losses)))
        history["lr_end_of_epoch"].append(lr_end)

        if is_a:
            vh, vo = eval_val(carry[0])
            vh_f, vo_f = float(vh), float(vo)
            history["val_hedged_per_scen"]["fixed_020"].append(vh_f)
            history["val_hedged_mean"].append(vh_f)
            history["val_orig_mean"].append(vo_f)
            sel_metric = vh_f
        else:
            vh_vec, vo_vec = eval_val(carry[0])
            vh_np = np.asarray(vh_vec, dtype=np.float64)
            vo_np = np.asarray(vo_vec, dtype=np.float64)
            for k, sc in enumerate(TRAINING_SCENARIOS):
                history["val_hedged_per_scen"][sc].append(float(vh_np[k]))
            sel_metric = float(np.mean(vh_np))
            history["val_hedged_mean"].append(sel_metric)
            history["val_orig_mean"].append(float(np.mean(vo_np)))

        history["epoch_time_sec"].append(float(time.time() - t0))

        is_new_best = sel_metric < best_val_metric
        need_snap = save_extra_ckpts and (ep_num in EXTRA_CKPT_EPOCHS)
        if is_new_best or need_snap:
            snap_params_np = jax.tree_util.tree_map(np.asarray, carry[0])
        if is_new_best:
            best_val_metric = sel_metric
            best_epoch = ep_num
            best_params_np = snap_params_np
        if need_snap:
            ep_path = ckpt_run_dir / f"ep_{ep_num:03d}.pkl"
            _atomic_pickle_dump(
                {
                    "config_name": config_name,
                    "seed_label": seed_label,
                    "epoch": ep_num,
                    "update_step": step_end,
                    "lr_end": lr_end,
                    "val_sel_metric": sel_metric,
                    "best_val_hedged": sel_metric,
                    "params": snap_params_np,
                },
                ep_path,
            )
            saved_epochs_map[str(ep_num)] = str(ep_path.name)

    verify_finite_tree(carry[0])
    total_time = float(time.time() - t0_all)

    # Deduplicate best_val checkpoint if already saved in EXTRA_CKPT_EPOCHS; else write best_val.pkl
    stale_best = ckpt_run_dir / "best_val.pkl"
    if save_extra_ckpts and str(best_epoch) in saved_epochs_map:
        best_ckpt_path = ckpt_run_dir / saved_epochs_map[str(best_epoch)]
        best_deduplicated = True
        if stale_best.exists():
            stale_best.unlink()
    else:
        best_ckpt_path = stale_best
        _atomic_pickle_dump(
            {
                "config_name": config_name,
                "seed_label": seed_label,
                "epoch": best_epoch,
                "update_step": best_epoch * STEPS_PER_EPOCH,
                "lr_end": compute_lr_float(best_epoch * STEPS_PER_EPOCH, schedule_mode),
                "val_sel_metric": best_val_metric,
                "best_val_hedged": best_val_metric,
                "params": best_params_np,
            },
            best_ckpt_path,
        )
        best_deduplicated = False

    # Always save final epoch 200 + optimizer state for full reproducibility
    final_params_np = jax.tree_util.tree_map(np.asarray, carry[0])
    final_opt_np = jax.tree_util.tree_map(np.asarray, carry[1])
    final_opt_path = ckpt_run_dir / "final_with_opt.pkl"
    _atomic_pickle_dump(
        {
            "config_name": config_name,
            "seed_label": seed_label,
            "epoch": train_cfg.epochs,
            "update_step": train_cfg.epochs * STEPS_PER_EPOCH,
            "lr_end": compute_lr_float(train_cfg.epochs * STEPS_PER_EPOCH, schedule_mode),
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
        "best_epoch": best_epoch,
        "best_epoch_if_val020_only": best_ep_val020,
        "best_ckpt_file": best_ckpt_path.name,
        "best_deduplicated_to_epoch_snap": best_deduplicated,
        "final_with_opt_file": final_opt_path.name,
        "saved_epoch_snaps": saved_epochs_map,
    }
    (ckpt_run_dir / "checkpoint_manifest.json").write_text(
        __import__("json").dumps(ckpt_manifest, indent=2), encoding="utf-8"
    )

    return {
        "config_name": config_name,
        "branch": "A_ref_128" if is_a else "R2_matched_128",
        "schedule": schedule_mode,
        "seed_label": seed_label,
        "param_counts": count_param_breakdown(best_params_np),
        "distinct_train_paths": 20000,
        "paths_per_step": 128,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_update_steps": train_cfg.epochs * STEPS_PER_EPOCH,
        "total_paths_processed": train_cfg.epochs * STEPS_PER_EPOCH * 128,
        "total_time_sec": total_time,
        "best_epoch": best_epoch,
        "best_epoch_if_val020_only": best_ep_val020,
        "best_val_hedged": best_val_metric,
        "val_nominal_hedged_at_best": float(val_020_curve[best_epoch - 1]),
        "ckpt_best_path": str(best_ckpt_path),
        "final_opt_path": str(final_opt_path),
        "checkpoint_manifest": ckpt_manifest,
        "history": history,
    }
