"""
bc_models.py: Model definitions and step loss for A_ref_32, A_ref_128, R2_matched_32, R2_matched_128.

Reuses canonical policy rollout & OCE utilities directly from robust_r3_dev/r3_models.py
and adds `aref_128_step_loss` matching the exact 4x32 accumulation structure of `r2_matched_step_loss`
while sharing scalar (y, y0) at index 0.
"""

from typing import List
import sys
from pathlib import Path
import jax
import jax.numpy as jnp

EXP_DIR = Path(__file__).resolve().parent.parent
R3_DIR = EXP_DIR / "robust_r3_dev"
if str(R3_DIR) not in sys.path:
    sys.path.insert(0, str(R3_DIR))

from r3_models import (
    GymConfigOriginal,
    make_canonical_configs,
    init_full_params,
    count_param_breakdown,
    rollout_policy_paths,
    rollout_with_oce,
    oce_hedged_loss_from_gains,
    oce_unhedged_loss_from_payoff,
    single_scenario_step_loss,
    r2_matched_step_loss,
)


def aref_128_step_loss(params: dict, gym_cfg: GymConfigOriginal, sub_batches: List[dict]) -> jnp.ndarray:
    """
    A_ref_128 step loss over 4 independent sigma=0.20 sub-batches (4 x 32 = 128 paths/step).
    Uses exact mean-of-4-rollouts structure matching r2_matched_step_loss float32 summation order,
    while sharing the single nominal (y[0], y0[0]) pair across all 4 sigma=0.20 sub-pools.
    """
    losses_h = []
    losses_u0 = []
    for b in sub_batches:
        pr = rollout_policy_paths(params["policy"], gym_cfg, b)
        losses_h.append(oce_hedged_loss_from_gains(pr["gains"], params["oce"]["y"][0], gym_cfg))
        losses_u0.append(oce_unhedged_loss_from_payoff(jax.lax.stop_gradient(pr["payoff"]), params["oce"]["y0"][0], gym_cfg))
    return jnp.mean(jnp.stack(losses_h)) + jnp.mean(jnp.stack(losses_u0))
