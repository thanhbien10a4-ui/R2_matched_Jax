"""
core/train.py: Paired Training Engine with Validation Loss Checkpoint Selection.

Ensures 100% fair benchmark parity between Branch A (Original JAX) and Branch B (JAX + NTBN):
- Synchronized sample permutations across minibatches: np.random.default_rng([c.seed, epoch])
- Pre-locked checkpoint selection based strictly on validation loss (val_loss)
- Test set strictly isolated and locked until after training is complete
- Exact Adam optimizer with clipnorm / global_clipnorm / clipvalue precedence matching TF 2.11
"""

from dataclasses import dataclass, replace
from typing import Dict, Any, Tuple, Optional, Callable
import time
import os
import hashlib
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from .model import (
    GymConfigOriginal, GymConfigNTBN,
    init_params_original, init_params_ntbn,
    rollout_original, rollout_ntbn,
    loss_original, loss_ntbn
)
from .metrics import evaluate_rollout_metrics


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    clipnorm: Optional[float] = None
    global_clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None
    seed: int = 0


@dataclass
class TrainResult:
    branch: str
    params: dict
    last_params: dict
    best_epoch: int
    best_val_loss: float
    history: dict
    train_time_sec: float
    completed: bool = True
    last_epoch: int = -1


HISTORY_KEYS = ("loss", "val_loss", "utility", "utility0", "val_utility", "val_utility0", "time")


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=zeros, v=zeros, t=jnp.zeros((), jnp.float32))


def adam_update(c: TrainConfig, grads, opt, params):
    if c.clipnorm:
        grads = jax.tree_util.tree_map(lambda g: g * c.clipnorm / jnp.maximum(jnp.linalg.norm(g), c.clipnorm), grads)
    elif c.global_clipnorm:
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
        grads = jax.tree_util.tree_map(lambda g: g * c.global_clipnorm / jnp.maximum(norm, c.global_clipnorm), grads)
    elif c.clipvalue:
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -c.clipvalue, c.clipvalue), grads)
        
    t = opt["t"] + 1.0
    alpha = c.learning_rate * jnp.sqrt(1.0 - c.beta_2 ** t) / (1.0 - c.beta_1 ** t)
    m = jax.tree_util.tree_map(lambda m, g: m + (g - m) * (1.0 - c.beta_1), opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: v + (g * g - v) * (1.0 - c.beta_2), opt["v"], grads)
    params = jax.tree_util.tree_map(lambda p, m, v: p - m * alpha / (jnp.sqrt(v) + c.epsilon), params, m, v)
    return params, dict(m=m, v=v, t=t)


def _take(batch, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], batch)


def as_batch(world) -> dict:
    return dict(
        market={k: jnp.asarray(v) for k, v in world.market.items()},
        features={k: jnp.asarray(v) for k, v in world.features.items()}
    )


def train_paired(
    branch: str,
    gym_cfg: Any,
    train_cfg: TrainConfig,
    train_data: dict,
    val_data: dict,
    key: int = 0,
    feature_dims: dict = None,
    n_inst: int = 1,
    log_every: int = 0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
    checkpoint_every: int = 50,
    resume: bool = True,
    max_session_seconds: Optional[float] = None
) -> TrainResult:
    """
    Unified training loop for either Branch A ('A') or Branch B ('B').
    Locks checkpoint selection strictly to validation loss.
    Supports atomic checkpointing and auto-resume if GPU reaches limits.
    """
    n = train_data["market"]["payoff"].shape[0]
    n_full = n // train_cfg.batch_size
    rng_key = jax.random.PRNGKey(key)

    if branch == "A":
        params = init_params_original(rng_key, gym_cfg, feature_dims, n_inst)
        loss_fn = loss_original
        rollout_fn = rollout_original
    elif branch == "B":
        params = init_params_ntbn(rng_key, gym_cfg, feature_dims, n_inst)
        loss_fn = loss_ntbn
        rollout_fn = rollout_ntbn
    else:
        raise ValueError(f"Unknown branch '{branch}'. Must be 'A' or 'B'.")

    def step(carry, idx, data):
        p, opt = carry
        g = jax.grad(loss_fn)(p, gym_cfg, _take(data, idx))
        return adam_update(train_cfg, g, opt, p), None

    @jax.jit
    def full_batches(carry, perm, data):
        carry, _ = jax.lax.scan(lambda cr, idx: step(cr, idx, data), carry, perm.reshape(n_full, train_cfg.batch_size))
        return carry

    tail_step = jax.jit(lambda carry, idx, data: step(carry, idx, data)[0])

    @jax.jit
    def evaluate(p, data):
        r = rollout_fn(p, gym_cfg, data)
        return jnp.mean(r["loss"]), jnp.mean(r["utility"]), jnp.mean(r["utility0"])

    # Checkpoint paths
    prefix = checkpoint_prefix or f"branch_{branch}"
    latest_chk_path = os.path.join(checkpoint_dir, f"{prefix}_latest.pkl") if checkpoint_dir else None
    best_chk_path = os.path.join(checkpoint_dir, f"{prefix}_best.pkl") if checkpoint_dir else None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(path, epoch_num, current_carry, b_params, b_vl, b_ep, hist_dict, el_time):
        if not path:
            return
        tmp_path = path + ".tmp"
        chk_data = {
            "epoch": epoch_num,
            "params": current_carry[0],
            "opt_state": current_carry[1],
            "best_params": b_params,
            "best_val_loss": b_vl,
            "best_epoch": b_ep,
            "history": hist_dict,
            "elapsed_time": el_time,
            "branch": branch,
            "seed": train_cfg.seed,
            "target_epochs": train_cfg.epochs
        }
        with open(tmp_path, "wb") as f:
            pickle.dump(chk_data, f)
        os.replace(tmp_path, path)

    resumed = False
    start_epoch = 0
    hist = {k: [] for k in HISTORY_KEYS}
    accumulated_time = 0.0

    if resume and latest_chk_path and os.path.exists(latest_chk_path):
        try:
            with open(latest_chk_path, "rb") as f:
                chk = pickle.load(f)
            last_saved_epoch = chk["epoch"]
            carry = (chk["params"], chk["opt_state"])
            best_params = chk["best_params"]
            best_val_loss = float(chk["best_val_loss"])
            best_epoch = chk["best_epoch"]
            hist = chk["history"]
            accumulated_time = float(chk.get("elapsed_time", 0.0))
            start_epoch = last_saved_epoch + 1
            resumed = True
            print(f"[{branch}] Resumed from checkpoint at epoch {last_saved_epoch + 1}/{train_cfg.epochs} "
                  f"(Best Val Loss: {best_val_loss:.5f} @ Epoch {best_epoch + 1})", flush=True)

            if start_epoch >= train_cfg.epochs:
                print(f"[{branch}] Training already completed ({last_saved_epoch + 1}/{train_cfg.epochs} epochs). "
                      f"Using saved best checkpoint.", flush=True)
                return TrainResult(
                    branch=branch,
                    params=best_params,
                    last_params=carry[0],
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    history=hist,
                    train_time_sec=accumulated_time,
                    completed=True,
                    last_epoch=last_saved_epoch
                )
        except Exception as e:
            print(f"[{branch}] Warning: Failed to load checkpoint ({e}). Starting fresh.", flush=True)
            resumed = False

    if not resumed:
        carry = (params, adam_init(params))
        init_tl, init_tu, init_tu0 = evaluate(params, train_data)
        init_vl, init_vu, init_vu0 = evaluate(params, val_data)
        best_params = params
        best_val_loss = float(init_vl)
        best_epoch = -1

    session_start = time.time()
    completed = True
    last_completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, train_cfg.epochs):
        t0 = time.time()
        
        # Exact reproducible permutation shared identically between A and B
        perm = np.random.default_rng([train_cfg.seed, epoch]).permutation(n)
        
        if n_full:
            carry = full_batches(carry, jnp.asarray(perm[:n_full * train_cfg.batch_size]), train_data)
        if n % train_cfg.batch_size:
            carry = tail_step(carry, jnp.asarray(perm[n_full * train_cfg.batch_size:]), train_data)
            
        tl, tu, tu0 = evaluate(carry[0], train_data)
        vl, vu, vu0 = evaluate(carry[0], val_data)
        
        for k, v in zip(HISTORY_KEYS[:-1], (tl, vl, tu, tu0, vu, vu0)):
            hist[k].append(float(v))
        hist["time"].append(time.time() - t0)
        accumulated_time += hist["time"][-1]
        last_completed_epoch = epoch
        
        # Checkpoint selection strictly on validation loss (Requirement 4)
        is_best = False
        if hist["val_loss"][-1] < best_val_loss:
            best_val_loss = hist["val_loss"][-1]
            best_params = carry[0]
            best_epoch = epoch
            is_best = True
            
        # Periodic and best checkpoint saving
        if checkpoint_dir and ((epoch + 1) % checkpoint_every == 0 or (epoch + 1) == train_cfg.epochs or is_best):
            save_checkpoint(latest_chk_path, epoch, carry, best_params, best_val_loss, best_epoch, hist, accumulated_time)
            if is_best and best_chk_path:
                save_checkpoint(best_chk_path, epoch, carry, best_params, best_val_loss, best_epoch, hist, accumulated_time)
            
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[{branch}] Epoch {epoch + 1}/{train_cfg.epochs} - Train Loss: {tl:.5f}, Val Loss: {vl:.5f} (Best Val@{best_epoch + 1}: {best_val_loss:.5f}) - {hist['time'][-1]:.2f}s", flush=True)

        # Check GPU session limit
        if max_session_seconds is not None and (time.time() - session_start) >= max_session_seconds:
            print(f"[{branch}] GPU session time limit reached ({time.time() - session_start:.1f}s >= {max_session_seconds}s) at epoch {epoch + 1}/{train_cfg.epochs}. Checkpoint saved cleanly.", flush=True)
            save_checkpoint(latest_chk_path, epoch, carry, best_params, best_val_loss, best_epoch, hist, accumulated_time)
            completed = False
            break

    return TrainResult(
        branch=branch,
        params=best_params,
        last_params=carry[0],
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=hist,
        train_time_sec=accumulated_time,
        completed=completed,
        last_epoch=last_completed_epoch
    )
