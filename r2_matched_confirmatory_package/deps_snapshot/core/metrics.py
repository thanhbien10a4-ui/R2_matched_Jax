"""
core/metrics.py: Evaluation metrics for Deep Hedging & NTBN experiments.

Strictly follows @gng5 Epistemic Gate standards:
- Exact Rockafellar-Uryasev (2000) variational CVaR95 & VaR95.
- Correct sign conventions: Net Loss L = - Gains = - (Payoff + PnL - Cost).
- Pre-locked metric schema for both Branch A (Original JAX) and Branch B (JAX + NTBN).
- Pathwise paired comparisons.
"""

from typing import Dict, Any, Tuple
import numpy as np
from .objectives import compute_var_cvar


def evaluate_rollout_metrics(rollout_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes all standard financial and deep hedging metrics from a rollout dictionary.
    """
    gains = np.asarray(rollout_dict["gains"], dtype=np.float64)
    pnl = np.asarray(rollout_dict["pnl"], dtype=np.float64)
    cost = np.asarray(rollout_dict["cost"], dtype=np.float64)
    payoff = np.asarray(rollout_dict["payoff"], dtype=np.float64)
    loss = np.asarray(rollout_dict["loss"], dtype=np.float64)
    actions = np.asarray(rollout_dict["actions"], dtype=np.float64)

    # Net loss for risk measure evaluation: L = - Gains
    net_losses = -gains

    # VaR95 and CVaR95 via Rockafellar-Uryasev
    var95, cvar95 = compute_var_cvar(net_losses, alpha=0.95)

    # Turnover: sum of absolute actions across steps and instruments per path
    turnover_per_path = np.sum(np.abs(actions), axis=(1, 2))
    turnover = float(np.mean(turnover_per_path))

    return {
        "loss": float(np.mean(loss)),
        "utility": float(np.mean(rollout_dict["utility"])),
        "utility0": float(np.mean(rollout_dict["utility0"])),
        "mean_gains": float(np.mean(gains)),
        "std_gains": float(np.std(gains)),
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl)),
        "mean_cost": float(np.mean(cost)),
        "std_cost": float(np.std(cost)),
        "turnover": turnover,
        "std_turnover": float(np.std(turnover_per_path)),
        "var95": var95,
        "cvar95": cvar95,
        "hedging_error": float(np.std(payoff + pnl)),
        "mean_payoff": float(np.mean(payoff)),
        "std_payoff": float(np.std(payoff)),
    }


def compute_paired_comparison(metrics_a: Dict[str, float], metrics_b: Dict[str, float],
                              rollout_a: Dict[str, Any], rollout_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes rigorous pathwise paired differences:
      Diff = Branch B (NTBN) - Branch A (Original)
    """
    gains_a = np.asarray(rollout_a["gains"], dtype=np.float64)
    gains_b = np.asarray(rollout_b["gains"], dtype=np.float64)
    diff_gains = gains_b - gains_a

    cost_a = np.asarray(rollout_a["cost"], dtype=np.float64)
    cost_b = np.asarray(rollout_b["cost"], dtype=np.float64)
    diff_cost = cost_b - cost_a

    act_a = np.asarray(rollout_a["actions"], dtype=np.float64)
    act_b = np.asarray(rollout_b["actions"], dtype=np.float64)
    turn_a = np.sum(np.abs(act_a), axis=(1, 2))
    turn_b = np.sum(np.abs(act_b), axis=(1, 2))
    diff_turn = turn_b - turn_a

    return {
        "delta_loss": metrics_b["loss"] - metrics_a["loss"],
        "delta_cvar95": metrics_b["cvar95"] - metrics_a["cvar95"],
        "delta_var95": metrics_b["var95"] - metrics_a["var95"],
        "delta_mean_cost": metrics_b["mean_cost"] - metrics_a["mean_cost"],
        "delta_turnover": metrics_b["turnover"] - metrics_a["turnover"],
        "pathwise_gains_diff_mean": float(np.mean(diff_gains)),
        "pathwise_gains_diff_std": float(np.std(diff_gains)),
        "pathwise_cost_diff_mean": float(np.mean(diff_cost)),
        "pathwise_cost_diff_std": float(np.std(diff_cost)),
        "pathwise_turnover_diff_mean": float(np.mean(diff_turn)),
        "pathwise_turnover_diff_std": float(np.std(diff_turn)),
    }
