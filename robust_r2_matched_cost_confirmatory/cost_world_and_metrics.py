"""
robust_r2_matched_cost_confirmatory/cost_world_and_metrics.py:
Explicit transaction-cost market simulator wrapper (`cost_s = 0.0010`) with
global NumPy RNG state preservation and extended post-fee + trading-mechanism metrics.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

COST_CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = COST_CONF_DIR.parent
R3_DIR = EXP_DIR / "robust_r3_dev"
P2_DIR = EXP_DIR / "scripts" / "robust_deep_hedging_phase2"

for _p in (str(COST_CONF_DIR), str(R3_DIR), str(EXP_DIR), str(P2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.train import as_batch
from robust_world import RobustWorldConfig, simulate_robust_world
from r3_keys import VOL_SEED_OFFSET
from r3_metrics import evaluate_rollout_metrics_r3
from run_r3_pilot import SCENARIO_SPECS
from cost_design import (
    COST_S_LOCKED,
    EPS_NONINFERIORITY_OCE020,
    ACTION_ACTIVE_THRESHOLD,
    DELTA_EXTREME_THRESHOLD,
    DELTA_CLIP_NEAR_THRESHOLD,
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_batch_spot_and_cost(batch: dict) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(batch["market"]["payoff"]).tobytes())
    h.update(np.asarray(batch["market"]["cost"]).tobytes())
    h.update(np.asarray(batch["features"]["price"]).tobytes())
    return h.hexdigest()[:16]


def build_cost_scenario_world(
    scen_name: str,
    rng_seed: int,
    samples: int = 5000,
    cost_s: float = COST_S_LOCKED,
) -> dict:
    """
    Build a JAX batch dict under SCENARIO_SPECS[scen_name] with explicit `cost_s`.
    Saves and restores legacy global np.random state around simulate_robust_world
    because upstream robust_world.py calls np.random.seed(c.seed).
    """
    if scen_name not in SCENARIO_SPECS:
        raise KeyError(f"Unknown scenario name: {scen_name}")
    sp = SCENARIO_SPECS[scen_name]
    saved_np_state = np.random.get_state()
    try:
        if sp["sigma_mode"] == "fixed":
            w = simulate_robust_world(
                RobustWorldConfig(
                    samples=int(samples),
                    steps=20,
                    sigma_mode="fixed",
                    rvol=float(sp["rvol"]),
                    drift=0.0,
                    cost_s=float(cost_s),
                    ubnd_as=5.0,
                    lbnd_as=-5.0,
                    strike=1.0,
                    seed=int(rng_seed),
                    vol_seed_offset=VOL_SEED_OFFSET,
                    normalize_batch_mean=True,
                )
            )
        elif sp["sigma_mode"] == "regime_switch":
            w = simulate_robust_world(
                RobustWorldConfig(
                    samples=int(samples),
                    steps=20,
                    sigma_mode="regime_switch",
                    regime_split_step=int(sp["regime_split_step"]),
                    regime_sigma_low=float(sp["regime_sigma_low"]),
                    regime_sigma_high=float(sp["regime_sigma_high"]),
                    drift=0.0,
                    cost_s=float(cost_s),
                    ubnd_as=5.0,
                    lbnd_as=-5.0,
                    strike=1.0,
                    seed=int(rng_seed),
                    vol_seed_offset=VOL_SEED_OFFSET,
                    normalize_batch_mean=True,
                )
            )
        else:
            raise ValueError(f"Unsupported sigma_mode: {sp['sigma_mode']}")
    finally:
        np.random.set_state(saved_np_state)

    return as_batch(w)


def evaluate_cost_rollout_metrics(
    rollout_nom: Dict[str, Any],
    rollout_scen: Optional[Dict[str, Any]] = None,
    ref_hedged_oce_y_nom: Optional[float] = None,
    eps: float = EPS_NONINFERIORITY_OCE020,
    action_thresh: float = ACTION_ACTIVE_THRESHOLD,
    delta_extreme_thresh: float = DELTA_EXTREME_THRESHOLD,
    delta_clip_thresh: float = DELTA_CLIP_NEAR_THRESHOLD,
) -> Dict[str, float]:
    """
    Compute complete confirmatory + diagnostic + cost/trading-mechanism metrics.
    Verifies gross gains - cost == net gains internally (zero double counting).
    """
    base = evaluate_rollout_metrics_r3(
        rollout_nom=rollout_nom,
        rollout_scen=rollout_scen,
        ref_hedged_oce_y_nom=ref_hedged_oce_y_nom,
        eps=eps,
    )

    payoff = np.asarray(rollout_nom["payoff"], dtype=np.float64)
    pnl = np.asarray(rollout_nom["pnl"], dtype=np.float64)
    cost = np.asarray(rollout_nom["cost"], dtype=np.float64)
    gains = np.asarray(rollout_nom["gains"], dtype=np.float64)
    actions = np.asarray(rollout_nom["actions"], dtype=np.float64)
    deltas = np.asarray(rollout_nom["deltas"], dtype=np.float64)
    if actions.ndim == 3 and actions.shape[-1] == 1:
        actions = actions[:, :, 0]
    if deltas.ndim == 3 and deltas.shape[-1] == 1:
        deltas = deltas[:, :, 0]

    gross_gains = payoff + pnl
    mean_gross_gains = float(np.mean(gross_gains))
    net_reconstructed = float(np.mean(gross_gains - cost))
    cost_accounting_residual = float(abs(net_reconstructed - float(np.mean(gains))))

    abs_act = np.abs(actions)
    active_mask = abs_act >= float(action_thresh)
    active_rate = float(np.mean(active_mask))
    immobility_rate = float(1.0 - active_rate)

    # Secondary threshold immobility diagnostics (descriptive only)
    immobility_rate_lt1e4 = float(1.0 - np.mean(abs_act >= 1e-4))
    immobility_rate_lt1e3 = float(1.0 - np.mean(abs_act >= 1e-3))

    # Average trade size when active: BOTH pooled ratio and pathwise mean-of-ratios
    n_active_cells = int(np.sum(active_mask))
    if n_active_cells > 0:
        mean_trade_size_active_pooled = float(np.sum(abs_act[active_mask]) / n_active_cells)
    else:
        mean_trade_size_active_pooled = 0.0

    active_per_path = np.sum(active_mask, axis=1)
    active_vol_per_path = np.sum(abs_act * active_mask, axis=1)
    has_active = active_per_path > 0
    if np.any(has_active):
        pathwise_sizes = active_vol_per_path[has_active] / active_per_path[has_active]
        mean_trade_size_active_pathwise = float(np.mean(pathwise_sizes))
    else:
        mean_trade_size_active_pathwise = 0.0
    zero_active_path_frac = float(np.mean(~has_active))

    abs_delta = np.abs(deltas)
    p99_abs_delta = float(np.percentile(abs_delta, 99.0))
    frac_steps_abs_delta_gt2 = float(np.mean(abs_delta > float(delta_extreme_thresh)))
    frac_steps_abs_delta_ge45 = float(np.mean(abs_delta >= float(delta_clip_thresh)))

    out = dict(base)
    out.update(
        {
            "mean_gross_gains": mean_gross_gains,
            "cost_accounting_residual": cost_accounting_residual,
            "immobility_rate": immobility_rate,
            "immobility_rate_lt1e4": immobility_rate_lt1e4,
            "immobility_rate_lt1e3": immobility_rate_lt1e3,
            "mean_trade_size_active_pooled": mean_trade_size_active_pooled,
            "mean_trade_size_active_pathwise": mean_trade_size_active_pathwise,
            "zero_active_path_frac": zero_active_path_frac,
            "p99_abs_delta": p99_abs_delta,
            "frac_steps_abs_delta_gt2": frac_steps_abs_delta_gt2,
            "frac_steps_abs_delta_ge45": frac_steps_abs_delta_ge45,
        }
    )
    return out
