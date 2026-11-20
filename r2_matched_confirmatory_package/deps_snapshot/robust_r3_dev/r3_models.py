"""
r3_models.py: JAX Policy Rollout, Numerically Stable Smooth-Max R_tau, and Decoupled OCE Losses.

Key design guarantees (@gng5 Design Lock):
1. Policy architecture IDENTICAL to Branch A / R2:
   - AgentConfigOriginal(features=("price", "delta", "time_left"), width=40, depth=5, activation="softplus")
   - Strictly 6,762 policy parameters (dense=6,761 + idelta=1); 0% volatility leakage, causal rollout.
2. Separate policy parameters vs auxiliary OCE shift parameters:
   - `A_ref` and `R2`: y shape (1,), y0 shape (1,) -> 6,764 total params.
   - `R2_matched` and `R3`: y shape (4,), y0 shape (4,) -> 6,770 total params (same 6,762 policy params!).
3. Decoupled OCE gradient flow for R3:
   - Inside R_tau(theta_pol), y_k is stop-gradiented so softmax weights w_k do NOT crush dy_k/dL_k at small tau.
   - Simultaneously, y_k receives exact unweighted gradient sum_k L_k(sg(theta_pol), y_k),
     and y0_k receives (1/K) sum_k (-u0_k) with zero policy gradient.
4. Numerically stable LogSumExp: R_tau = tau * (jax.nn.logsumexp(L / tau) - log(K)).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from core.model import (
    AgentConfigOriginal,
    GymConfigOriginal,
    ACTIVATIONS,
    input_features_original,
    softclip,
    _glorot,
)
from core.objectives import utility


def make_canonical_configs(lmbda: float = 10.0) -> Tuple[AgentConfigOriginal, GymConfigOriginal, dict]:
    agent_cfg = AgentConfigOriginal(
        features=("price", "delta", "time_left"),
        width=40,
        depth=5,
        activation="softplus",
        init_delta=True,
    )
    gym_cfg = GymConfigOriginal(agent=agent_cfg, utility="exp2", lmbda=lmbda)
    feature_dims = {"price": 1, "time_left": 1}
    return agent_cfg, gym_cfg, feature_dims


def init_policy_params(key: jax.Array, gym_cfg: GymConfigOriginal, feature_dims: dict, n_inst: int = 1) -> dict:
    """Initializes ONLY policy network parameters (dense + idelta): strictly 6,762 parameters."""
    a = gym_cfg.agent
    dims = dict(feature_dims, action=n_inst, delta=n_inst)
    sizes = [sum(dims[f] for f in input_features_original(a))] + [a.width] * a.depth + [n_inst]
    keys = jax.random.split(key, len(sizes) + 1)
    params = dict(
        dense=[
            dict(W=_glorot(k, (i, o)), b=jnp.zeros((o,), jnp.float32))
            for k, i, o in zip(keys[:-2], sizes[:-1], sizes[1:])
        ]
    )
    if a.init_delta:
        params["idelta"] = _glorot(keys[-1], (n_inst,))
    return params


def init_full_params(
    key: jax.Array,
    gym_cfg: GymConfigOriginal,
    feature_dims: dict,
    k_oce: int = 1,
    n_inst: int = 1,
) -> dict:
    """Initializes full parameter pytree with separated `policy` and `oce` leaves."""
    pol = init_policy_params(key, gym_cfg, feature_dims, n_inst=n_inst)
    oce = dict(
        y=jnp.zeros((k_oce,), jnp.float32),
        y0=jnp.zeros((k_oce,), jnp.float32),
    )
    return dict(policy=pol, oce=oce)


def count_param_breakdown(params: dict) -> Dict[str, int]:
    pol_dense = int(sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(params["policy"]["dense"])))
    pol_idelta = int(np.asarray(params["policy"]["idelta"]).size) if "idelta" in params["policy"] else 0
    y_cnt = int(np.asarray(params["oce"]["y"]).size)
    y0_cnt = int(np.asarray(params["oce"]["y0"]).size)
    return {
        "policy_dense_params": pol_dense,
        "policy_idelta_params": pol_idelta,
        "policy_total_params": pol_dense + pol_idelta,
        "oce_y_params": y_cnt,
        "oce_y0_params": y0_cnt,
        "oce_total_params": y_cnt + y0_cnt,
        "grand_total_params": pol_dense + pol_idelta + y_cnt + y0_cnt,
    }


def rollout_policy_paths(policy_params: dict, gym_cfg: GymConfigOriginal, batch: dict) -> dict:
    """
    Executes causal policy rollout over one scenario batch, independent of OCE offsets y, y0.
    Matches core/model.py rollout_original bit-for-bit on policy forward pass and PnL.
    """
    a_cfg = gym_cfg.agent
    act = ACTIVATIONS[a_cfg.activation]
    feat_names = input_features_original(a_cfg)
    m = batch["market"]
    payoff, hedges = m["payoff"], m["hedges"]
    n, _, n_inst = hedges.shape
    zeros_i = jnp.zeros((n, n_inst), jnp.float32)
    idelta = policy_params["idelta"][None, :] if a_cfg.init_delta else zeros_i
    tm = lambda x: jnp.swapaxes(x, 0, 1)
    xs = (
        tm(hedges),
        tm(m["cost"]),
        tm(m["lbnd_a"]),
        tm(m["ubnd_a"]),
        {k: tm(v) for k, v in batch["features"].items()},
    )

    def body(carry, x):
        delta, action, cost, pnl, first = carry
        h, c, lb, ub, f = x
        feats = dict(action=action, delta=delta, **f)
        z = jnp.concatenate([feats[name] for name in feat_names], axis=1)
        for layer in policy_params["dense"][:-1]:
            z = act(z @ layer["W"] + layer["b"])
        out = z @ policy_params["dense"][-1]["W"] + policy_params["dense"][-1]["b"]
        a = softclip(out + idelta * first, lb, ub)
        delta = delta + a
        cost = cost + jnp.sum(jnp.abs(a) * c, axis=1)
        pnl = pnl + jnp.sum(a * h, axis=1)
        return (delta, a, cost, pnl, 0.0), (jax.lax.stop_gradient(a), jax.lax.stop_gradient(delta))

    z = jnp.zeros_like(payoff)
    (_, _, cost, pnl, _), (actions, deltas) = jax.lax.scan(body, (zeros_i, zeros_i, z, z, 1.0), xs)
    gains = payoff + pnl - cost
    return dict(
        gains=gains,
        payoff=payoff,
        pnl=pnl,
        cost=cost,
        actions=jnp.swapaxes(actions, 0, 1),
        deltas=jnp.swapaxes(deltas, 0, 1),
    )


def oce_hedged_loss_from_gains(gains: jnp.ndarray, y_scalar: jnp.ndarray, gym_cfg: GymConfigOriginal) -> jnp.ndarray:
    """Computes L_k(theta, y) = E[-u_tilde(gains, y)] (smaller is better)."""
    u = utility(gym_cfg.utility, gym_cfg.lmbda, gains, y_scalar)
    return -jnp.mean(u)


def oce_unhedged_loss_from_payoff(payoff: jnp.ndarray, y0_scalar: jnp.ndarray, gym_cfg: GymConfigOriginal) -> jnp.ndarray:
    """Computes -E[u_tilde(payoff, y0)] (gradient w.r.t. policy is identically zero)."""
    u0 = utility(gym_cfg.utility, gym_cfg.lmbda, payoff, y0_scalar)
    return -jnp.mean(u0)


def rollout_with_oce(
    params: dict,
    gym_cfg: GymConfigOriginal,
    batch: dict,
    oce_idx: int = 0,
    eval_y_override: Optional[jnp.ndarray] = None,
    eval_y0_override: Optional[jnp.ndarray] = None,
) -> dict:
    """Full rollout returning pathwise gains, actions, deltas, and OCE risk terms."""
    path_res = rollout_policy_paths(params["policy"], gym_cfg, batch)
    y_val = eval_y_override if eval_y_override is not None else params["oce"]["y"][oce_idx]
    y0_val = eval_y0_override if eval_y0_override is not None else params["oce"]["y0"][oce_idx]
    u = utility(gym_cfg.utility, gym_cfg.lmbda, path_res["gains"], y_val)
    u0 = utility(gym_cfg.utility, gym_cfg.lmbda, path_res["payoff"], y0_val)
    sg = jax.lax.stop_gradient
    return dict(
        hedged_loss_mean=-jnp.mean(u),
        unhedged_loss_mean=-jnp.mean(u0),
        total_loss_mean=-jnp.mean(u) - jnp.mean(u0),
        hedged_loss=sg(-u),
        unhedged_loss=sg(-u0),
        loss=sg(-u - u0),
        utility=sg(u),
        utility0=sg(u0),
        gains=sg(path_res["gains"]),
        payoff=path_res["payoff"],
        pnl=sg(path_res["pnl"]),
        cost=sg(path_res["cost"]),
        actions=path_res["actions"],
        deltas=path_res["deltas"],
    )


def smooth_max_logsumexp(risks: jnp.ndarray, tau: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Numerically stable smooth maximum:
      R_tau(L) = tau * ( logsumexp(L / tau) - log(K) )
    Returns (R_tau, softmax_weights).
    Note: jax.nn.logsumexp subtracts max(L / tau) internally to prevent overflow/underflow.
    """
    k = risks.shape[0]
    scaled = risks / tau
    lse = jax.nn.logsumexp(scaled)
    r_tau = tau * (lse - jnp.log(jnp.asarray(k, dtype=risks.dtype)))
    weights = jax.nn.softmax(scaled)
    return r_tau, weights


def single_scenario_step_loss(params: dict, gym_cfg: GymConfigOriginal, batch: dict) -> jnp.ndarray:
    """Objective for A_ref and R2: hedged OCE + auxiliary unhedged y0 (no gradient coupling to policy)."""
    path_res = rollout_policy_paths(params["policy"], gym_cfg, batch)
    l_hedged = oce_hedged_loss_from_gains(path_res["gains"], params["oce"]["y"][0], gym_cfg)
    l_unhedged = oce_unhedged_loss_from_payoff(jax.lax.stop_gradient(path_res["payoff"]), params["oce"]["y0"][0], gym_cfg)
    return l_hedged + l_unhedged


def r2_matched_step_loss(params: dict, gym_cfg: GymConfigOriginal, scenario_batches: List[dict]) -> jnp.ndarray:
    """
    Objective for R2-matched: uniform average over the K=4 scenarios with scenario-specific y_k.
      (1/K) * sum_k L_k(theta, y_k) + (1/K) * sum_k (-u0_k)
    """
    k = len(scenario_batches)
    losses_h = []
    losses_u0 = []
    for idx, b in enumerate(scenario_batches):
        pr = rollout_policy_paths(params["policy"], gym_cfg, b)
        losses_h.append(oce_hedged_loss_from_gains(pr["gains"], params["oce"]["y"][idx], gym_cfg))
        losses_u0.append(oce_unhedged_loss_from_payoff(jax.lax.stop_gradient(pr["payoff"]), params["oce"]["y0"][idx], gym_cfg))
    return jnp.mean(jnp.stack(losses_h)) + jnp.mean(jnp.stack(losses_u0))


def r3_constrained_step_loss(
    params: dict,
    ref_params_frozen: dict,
    gym_cfg: GymConfigOriginal,
    scenario_batches: List[dict],
    tau: float,
    lam_dual: jnp.ndarray,
    nominal_idx: int = 1,
) -> Tuple[jnp.ndarray, dict]:
    """
    Unbiased Primal-Dual Surrogate for R3 (@gng5 Fix 1 & Fix 2):
    - Computes L_k(theta_pol, sg(y_k)) for policy gradient inside smooth_max_logsumexp.
    - Updates each y_k with unweighted gradient sum_k L_k(sg(theta_pol), y_k) so softmax
      weights near 0 on low-vol scenarios NEVER stall Adam updates on y_0.
    - Adds linear penalty term stop_grad(lam_dual) * L_nom(theta_pol, sg(y_nom); B_nom),
      where lam_dual is updated via EMA-smoothed paired nominal violations outside autodiff,
      eliminating mini-batch Jensen false-gate bias completely.
    - Evaluates frozen A_ref on the exact same paired nominal mini-batch B_nom under stop_gradient.
    """
    sg = jax.lax.stop_gradient
    y_sg = sg(params["oce"]["y"])

    risks_for_theta = []
    risks_for_y = []
    losses_u0 = []

    for idx, b in enumerate(scenario_batches):
        pr = rollout_policy_paths(params["policy"], gym_cfg, b)
        gains_theta = pr["gains"]
        gains_sg = sg(gains_theta)
        # Gradient flows ONLY into policy weights
        risks_for_theta.append(oce_hedged_loss_from_gains(gains_theta, y_sg[idx], gym_cfg))
        # Gradient flows ONLY into y_k (weight 1.0, unaffected by tau softmax weights!)
        risks_for_y.append(oce_hedged_loss_from_gains(gains_sg, params["oce"]["y"][idx], gym_cfg))
        losses_u0.append(oce_unhedged_loss_from_payoff(sg(pr["payoff"]), params["oce"]["y0"][idx], gym_cfg))

    risks_theta_vec = jnp.stack(risks_for_theta)
    r_tau, weights = smooth_max_logsumexp(risks_theta_vec, tau=tau)
    l_nom_theta = risks_theta_vec[nominal_idx]

    # Paired frozen nominal reference rollout (zero gradient to A_ref!)
    ref_pol_frozen = sg(ref_params_frozen["policy"])
    ref_y_frozen = sg(ref_params_frozen["oce"]["y"][0])
    ref_pr = rollout_policy_paths(ref_pol_frozen, gym_cfg, scenario_batches[nominal_idx])
    l_nom_ref = sg(oce_hedged_loss_from_gains(ref_pr["gains"], ref_y_frozen, gym_cfg))

    batch_delta_nom = sg(l_nom_theta - l_nom_ref)
    surrogate = (
        r_tau
        + sg(lam_dual) * l_nom_theta
        + jnp.sum(jnp.stack(risks_for_y))
        + jnp.mean(jnp.stack(losses_u0))
    )
    aux = dict(
        r_tau=sg(r_tau),
        risks=sg(risks_theta_vec),
        weights=sg(weights),
        l_nom_theta=sg(l_nom_theta),
        l_nom_ref=l_nom_ref,
        batch_delta_nom=batch_delta_nom,
    )
    return surrogate, aux
