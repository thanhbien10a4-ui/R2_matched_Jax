"""
r3_metrics.py: Unified financial, hedging, position-stability, and constraint-violation metrics.

All branches (A_ref, R2, R2_matched, R3_cfg1, R3_cfg2) pass through this exact function.
Strictly NO parameter optimization is performed on Validation or Test sets.
"""

from typing import Dict, Any, Optional
import sys
from pathlib import Path
import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.objectives import compute_var_cvar


def evaluate_rollout_metrics_r3(
    rollout_nom: Dict[str, Any],
    rollout_scen: Optional[Dict[str, Any]] = None,
    ref_hedged_oce_y_nom: Optional[float] = None,
    eps: float = 0.00010,
    lam_penalty: float = 0.0,
) -> Dict[str, float]:
    """
    Computes unified metrics from frozen rollout dictionaries.
    - `rollout_nom`: evaluated with each branch's frozen nominal (sigma=0.20) offset y_nom.
    - `rollout_scen`: optional, evaluated with scenario-specific frozen y_k (only for in-training scenarios).
    """
    gains = np.asarray(rollout_nom["gains"], dtype=np.float64)
    pnl = np.asarray(rollout_nom["pnl"], dtype=np.float64)
    cost = np.asarray(rollout_nom["cost"], dtype=np.float64)
    payoff = np.asarray(rollout_nom["payoff"], dtype=np.float64)
    actions = np.asarray(rollout_nom["actions"], dtype=np.float64)
    deltas = np.asarray(rollout_nom["deltas"], dtype=np.float64)

    # Positive net loss L_net = -Gains (Rockafellar-Uryasev convention)
    net_losses = -gains
    var95, cvar95 = compute_var_cvar(net_losses, alpha=0.95)

    # Hedging errors: since cost_s = 0.0 in Stage 1 R3, net == gross hedging error
    he_net_ddof0 = float(np.std(gains, ddof=0))
    he_net_ddof1 = float(np.std(gains, ddof=1))

    # Trading volume & rebalancing activity
    abs_act = np.abs(actions)
    turnover_per_path = np.sum(abs_act, axis=(1, 2))
    active_steps_gt1e5 = np.mean(np.sum(abs_act >= 1e-5, axis=1))
    active_trade_rate = float(np.mean(abs_act >= 1e-5))

    # Position stability / extreme positions
    abs_delta = np.abs(deltas)
    max_abs_delta = float(np.max(abs_delta))
    p99_abs_delta = float(np.percentile(abs_delta, 99.0))
    mean_abs_delta = float(np.mean(abs_delta))
    mean_delta_t0 = float(np.mean(deltas[:, 0, 0]))
    max_abs_action = float(np.max(abs_act))

    hedged_oce_y_nom = float(np.mean(np.asarray(rollout_nom["hedged_loss"], dtype=np.float64)))
    unhedged_y_nom = float(np.mean(np.asarray(rollout_nom["unhedged_loss"], dtype=np.float64)))
    loss_orig_y_nom = float(np.mean(np.asarray(rollout_nom["loss"], dtype=np.float64)))

    if rollout_scen is not None:
        hedged_oce_y_scen = float(np.mean(np.asarray(rollout_scen["hedged_loss"], dtype=np.float64)))
        unhedged_y_scen = float(np.mean(np.asarray(rollout_scen["unhedged_loss"], dtype=np.float64)))
        loss_orig_y_scen = float(np.mean(np.asarray(rollout_scen["loss"], dtype=np.float64)))
    else:
        hedged_oce_y_scen = float("nan")
        unhedged_y_scen = float("nan")
        loss_orig_y_scen = float("nan")

    if ref_hedged_oce_y_nom is not None and np.isfinite(ref_hedged_oce_y_nom):
        delta_vs_aref = float(hedged_oce_y_nom - ref_hedged_oce_y_nom)
        excess_violation = float(max(0.0, delta_vs_aref - eps))
        raw_penalty_term = float(lam_penalty * excess_violation)
    else:
        delta_vs_aref = float("nan")
        excess_violation = float("nan")
        raw_penalty_term = float("nan")

    return {
        "hedged_oce_y_nom": hedged_oce_y_nom,
        "hedged_oce_y_scen": hedged_oce_y_scen,
        "unhedged_oce_y_nom": unhedged_y_nom,
        "unhedged_oce_y_scen": unhedged_y_scen,
        "loss_orig_y_nom": loss_orig_y_nom,
        "loss_orig_y_scen": loss_orig_y_scen,
        "he_net_ddof0": he_net_ddof0,
        "he_net_ddof1": he_net_ddof1,
        "var95": var95,
        "cvar95": cvar95,
        "mean_gains": float(np.mean(gains)),
        "std_gains": he_net_ddof0,
        "mean_pnl": float(np.mean(pnl)),
        "mean_cost": float(np.mean(cost)),
        "max_abs_cost": float(np.max(np.abs(cost))),
        "mean_payoff": float(np.mean(payoff)),
        "turnover": float(np.mean(turnover_per_path)),
        "std_turnover": float(np.std(turnover_per_path, ddof=0)),
        "active_steps_gt1e5": float(active_steps_gt1e5),
        "active_trade_rate": active_trade_rate,
        "max_abs_delta": max_abs_delta,
        "p99_abs_delta": p99_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "mean_delta_t0": mean_delta_t0,
        "max_abs_action": max_abs_action,
        "delta_hedged_oce_vs_aref_y_nom": delta_vs_aref,
        "constraint_violation_above_eps": excess_violation,
        "raw_penalty_term": raw_penalty_term,
    }
