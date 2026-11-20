"""
lr_schedule.py: Exact step-based Cosine Learning Rate Schedule for `robust_r3_lr_stability/`.

- Epochs: 200, Steps per epoch: 156, Total updates: 31,200.
- Uses 1-based update step `t = completed_steps + 1.0` inside Adam:
  * FIXED: lr(t) = 1e-3 for all t in [1..31200].
  * DECAY: lr(t) = 1e-3 for t <= 15600 (epochs 1..100),
           for t in [15601..31200] (epochs 101..200):
             u = (t - 15600) / (31200 - 15600)
             lr(t) = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + cos(pi * u))
- Exact checkpoint values:
  * t = 1      (ep 1, step 1):     1.000000e-03
  * t = 15600  (ep 100, step 156): 1.000000e-03
  * t = 15601  (ep 101, step 1):   ~9.99999991e-04
  * t = 23400  (ep 150, step 156): 5.500000e-04  (exact midpoint)
  * t = 31200  (ep 200, step 156): 1.000000e-04  (exact final update)
"""

import math
import jax.numpy as jnp

LR_MAX = 1e-3
LR_MIN = 1e-4
STEPS_PER_EPOCH = 156
CONST_EPOCHS = 100
TOTAL_EPOCHS = 200
T_CONST = CONST_EPOCHS * STEPS_PER_EPOCH   # 15,600
T_TOTAL = TOTAL_EPOCHS * STEPS_PER_EPOCH   # 31,200


def compute_lr_jnp(step_t: jnp.ndarray, schedule_mode: str) -> jnp.ndarray:
    """
    Computes scalar float32 learning rate as a pure function of 1-based update step `step_t = opt['t'] + 1.0`.
    """
    if schedule_mode == "FIXED":
        return jnp.asarray(LR_MAX, dtype=jnp.float32)
    if schedule_mode == "DECAY":
        t = jnp.asarray(step_t, dtype=jnp.float32)
        u = jnp.clip((t - float(T_CONST)) / float(T_TOTAL - T_CONST), 0.0, 1.0)
        decay_val = float(LR_MIN) + 0.5 * float(LR_MAX - LR_MIN) * (1.0 + jnp.cos(jnp.pi * u))
        return jnp.where(t <= float(T_CONST), jnp.asarray(LR_MAX, dtype=jnp.float32), decay_val)
    raise ValueError(f"Unknown schedule_mode: {schedule_mode}")


def compute_lr_float(step_t: int, schedule_mode: str) -> float:
    if schedule_mode == "FIXED":
        return float(LR_MAX)
    if schedule_mode == "DECAY":
        if step_t <= T_CONST:
            return float(LR_MAX)
        u = min(1.0, max(0.0, float(step_t - T_CONST) / float(T_TOTAL - T_CONST)))
        return float(LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * u)))
    raise ValueError(f"Unknown schedule_mode: {schedule_mode}")
