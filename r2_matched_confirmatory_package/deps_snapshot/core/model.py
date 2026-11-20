"""
core/model.py: Dual-Branch Deep Hedging Architecture in JAX.

Supports:
- Branch A: Canonical JAX Deep Hedging (Buehler et al. 2019 / NdQuan 2024)
- Branch B: JAX + No-Transaction Band Network (NTBN) (Imaki et al. 2021)

Both branches support:
- Multi-asset hedging (n_inst >= 1, e.g. spot + call option in trainer/sv)
- Recurrent memory states (classic, aggregate, past_repr, event, bound_aggr_states)
- All 6 OCE utilities (mean, exp, exp2, vicky, cvar, quad)
- Finite difference, boundary constraints, and non-anticipative rollout.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import jax
import jax.numpy as jnp

from .ntbn_primitives import ntbn_clamp, ntbn_bounds_center_width, ntbn_action
from .objectives import utility, UTILITIES

STATE = "recurrent_state"

ACTIVATIONS = dict(
    relu=jax.nn.relu,
    softplus=jax.nn.softplus,
    tanh=jnp.tanh,
    sigmoid=jax.nn.sigmoid,
    linear=lambda x: x
)


# ==============================================================================
# CONFIGURATIONS
# ==============================================================================

@dataclass(frozen=True)
class AgentConfigOriginal:
    """Configuration for Branch A (Canonical JAX Agent)."""
    features: tuple = ("price", "delta", "time_left")
    width: int = 20
    depth: int = 3
    activation: str = "relu"
    classic: int = 0
    aggregate: int = 0
    past_repr: int = 0
    event: int = 0
    bound_aggr_states: bool = False
    init_delta: bool = True

    @property
    def n_states(self) -> int:
        return self.classic + self.aggregate + self.past_repr + self.event


@dataclass(frozen=True)
class GymConfigOriginal:
    agent: AgentConfigOriginal = field(default_factory=AgentConfigOriginal)
    utility: str = "exp2"
    lmbda: float = 1.0


@dataclass(frozen=True)
class AgentConfigNTBN:
    """Configuration for Branch B (JAX + NTBN Agent)."""
    features: tuple = ("price", "time_left")
    width: int = 20
    depth: int = 3
    activation: str = "relu"
    band_mode: str = "center_width"  # "center_width" (strictly bl <= bu) or "direct"
    classic: int = 0
    aggregate: int = 0
    past_repr: int = 0
    event: int = 0
    bound_aggr_states: bool = False
    init_delta: bool = False         # Pure NTBN starts with delta_0 = 0
    action_clipping: str = "none"    # "none" (pure NTBN), "hard" (clip), "soft" (softclip)

    @property
    def n_states(self) -> int:
        return self.classic + self.aggregate + self.past_repr + self.event


@dataclass(frozen=True)
class GymConfigNTBN:
    agent: AgentConfigNTBN = field(default_factory=AgentConfigNTBN)
    utility: str = "exp2"
    lmbda: float = 1.0


# ==============================================================================
# HELPER FUNCTIONS & MATH OPERATORS
# ==============================================================================

def input_features_original(cfg: AgentConfigOriginal) -> list:
    return sorted(set(cfg.features) | ({STATE} if cfg.n_states else set()))


def input_features_ntbn(cfg: AgentConfigNTBN) -> list:
    """Excludes delta and action from neural network inputs (action-independence)."""
    exogenous = {f for f in cfg.features if f not in ("delta", "action")}
    if cfg.n_states:
        exogenous.add(STATE)
    return sorted(exogenous)


def _glorot(key, shape):
    fan_in, fan_out = (shape[0], shape[0]) if len(shape) == 1 else shape
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, jnp.float32, -lim, lim)


def _unit(x):
    return jnp.where(x > 0.5, 1.0, 0.0)


def _split(x, sizes):
    out, start = [], 0
    for s in sizes:
        out.append(x[:, start:start + s])
        start += s
    return out


def softclip(a, lb, ub):
    """Buehler's DHSoftClip smooth saturation function."""
    sp = jax.nn.softplus
    a = jnp.minimum(a, ub * 10.0)
    a = jnp.maximum(a, lb * 10.0)
    d = ub - lb
    r = (a - lb) / d
    r = 1.0 - sp(1.0 - sp(r)) / sp(1.0)
    return jnp.where(d > 0.0, r * d + lb, 0.0)


# ==============================================================================
# BRANCH A: CANONICAL JAX INITIALIZATION & STEP
# ==============================================================================

def init_params_original(key, cfg: GymConfigOriginal, feature_dims: dict, n_inst: int) -> dict:
    a = cfg.agent
    dims = dict(feature_dims, action=n_inst, delta=n_inst, **({STATE: a.n_states} if a.n_states else {}))
    sizes = [sum(dims[f] for f in input_features_original(a))] + [a.width] * a.depth + [n_inst + a.n_states + a.aggregate + a.past_repr + a.event]
    keys = jax.random.split(key, len(sizes) + 1)
    params = dict(
        dense=[dict(W=_glorot(k, (i, o)), b=jnp.zeros((o,), jnp.float32)) for k, i, o in zip(keys[:-2], sizes[:-1], sizes[1:])],
        y=jnp.zeros((1,), jnp.float32),
        y0=jnp.zeros((1,), jnp.float32)
    )
    if a.n_states:
        params["state0"] = _glorot(keys[-2], (a.n_states,))
    if a.init_delta:
        params["idelta"] = _glorot(keys[-1], (n_inst,))
    return params


def agent_step_original(p: dict, cfg: AgentConfigOriginal, act, feats: dict, state):
    feats = dict(feats)
    if cfg.n_states:
        classic, aggr, past, event = _split(state, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event))
        classic, event = jnp.tanh(classic), _unit(event)
        aggr = jnp.tanh(aggr) if cfg.bound_aggr_states else aggr
        feats[STATE] = jnp.concatenate([classic, aggr, past, event], axis=1)
    z = jnp.concatenate([feats[f] for f in input_features_original(cfg)], axis=1)
    for layer in p["dense"][:-1]:
        z = act(z @ layer["W"] + layer["b"])
    out = z @ p["dense"][-1]["W"] + p["dense"][-1]["b"]
    n_inst = out.shape[1] - cfg.n_states - cfg.aggregate - cfg.past_repr - cfg.event
    if not cfg.n_states:
        return out, state
    rec = out[:, n_inst:]
    c_new, a_cand, p_cand, e_cand, a_upd, p_upd, e_upd = _split(
        rec, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event, cfg.aggregate, cfg.past_repr, cfg.event))
    a_cand = jnp.tanh(a_cand) if cfg.bound_aggr_states else a_cand
    a_upd, p_upd, e_upd = jax.nn.sigmoid(a_upd), _unit(p_upd), _unit(e_upd)
    new_state = jnp.concatenate([
        jnp.tanh(c_new),
        (1.0 - a_upd) * aggr + a_upd * a_cand,
        (1.0 - p_upd) * past + p_upd * p_cand,
        (1.0 - e_upd) * event + e_upd * _unit(e_cand),
    ], axis=1)
    return out[:, :n_inst], new_state


def rollout_original(p: dict, cfg: GymConfigOriginal, batch: dict) -> dict:
    a_cfg, act = cfg.agent, ACTIVATIONS[cfg.agent.activation]
    m = batch["market"]
    payoff, hedges = m["payoff"], m["hedges"]
    n, _, n_inst = hedges.shape
    zeros_i = jnp.zeros((n, n_inst), jnp.float32)
    state0 = jnp.broadcast_to(p["state0"], (n, a_cfg.n_states)) if a_cfg.n_states else jnp.zeros((n, 0), jnp.float32)
    idelta = p["idelta"][None, :] if a_cfg.init_delta else zeros_i
    tm = lambda x: jnp.swapaxes(x, 0, 1)
    xs = (tm(hedges), tm(m["cost"]), tm(m["lbnd_a"]), tm(m["ubnd_a"]), {k: tm(v) for k, v in batch["features"].items()})

    def body(carry, x):
        delta, action, cost, pnl, state, first = carry
        h, c, lb, ub, f = x
        feats = dict(action=action, delta=delta, **f)
        a, state = agent_step_original(p, a_cfg, act, feats, state)
        a = softclip(a + idelta * first, lb, ub)
        delta = delta + a
        cost = cost + jnp.sum(jnp.abs(a) * c, axis=1)
        pnl = pnl + jnp.sum(a * h, axis=1)
        return (delta, a, cost, pnl, state, 0.0), jax.lax.stop_gradient(a)

    z = jnp.zeros_like(payoff)
    (_, _, cost, pnl, _, _), actions = jax.lax.scan(body, (zeros_i, zeros_i, z, z, state0, 1.0), xs)
    y = p["y"][0] if cfg.utility != "mean" else 0.0
    y0 = p["y0"][0] if cfg.utility != "mean" else 0.0
    u = utility(cfg.utility, cfg.lmbda, payoff + pnl - cost, y)
    u0 = utility(cfg.utility, cfg.lmbda, payoff, y0)
    sg = jax.lax.stop_gradient
    return dict(
        loss=-u - u0, utility=sg(u), utility0=sg(u0), gains=sg(payoff + pnl - cost),
        payoff=payoff, pnl=sg(pnl), cost=sg(cost), actions=jnp.swapaxes(actions, 0, 1)
    )


def loss_original(p: dict, cfg: GymConfigOriginal, batch: dict):
    return jnp.mean(rollout_original(p, cfg, batch)["loss"])


# ==============================================================================
# BRANCH B: JAX + NTBN INITIALIZATION & STEP
# ==============================================================================

def init_params_ntbn(key, cfg: GymConfigNTBN, feature_dims: dict, n_inst: int) -> dict:
    a = cfg.agent
    dims = dict(feature_dims, **({STATE: a.n_states} if a.n_states else {}))
    in_dim = sum(dims[f] for f in input_features_ntbn(a))
    out_dim = 2 * n_inst + a.n_states + a.aggregate + a.past_repr + a.event
    sizes = [in_dim] + [a.width] * a.depth + [out_dim]
    keys = jax.random.split(key, len(sizes) + 1)
    params = dict(
        dense=[dict(W=_glorot(k, (i, o)), b=jnp.zeros((o,), jnp.float32)) for k, i, o in zip(keys[:-2], sizes[:-1], sizes[1:])],
        y=jnp.zeros((1,), jnp.float32),
        y0=jnp.zeros((1,), jnp.float32)
    )
    if a.n_states:
        params["state0"] = _glorot(keys[-2], (a.n_states,))
    if a.init_delta:
        params["idelta"] = _glorot(keys[-1], (n_inst,))
    return params


def agent_step_ntbn(p: dict, cfg: AgentConfigNTBN, act, feats: dict, state, n_inst: int):
    feats = dict(feats)
    if cfg.n_states:
        classic, aggr, past, event = _split(state, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event))
        classic, event = jnp.tanh(classic), _unit(event)
        aggr = jnp.tanh(aggr) if cfg.bound_aggr_states else aggr
        feats[STATE] = jnp.concatenate([classic, aggr, past, event], axis=1)
    z = jnp.concatenate([feats[f] for f in input_features_ntbn(cfg)], axis=1)
    for layer in p["dense"][:-1]:
        z = act(z @ layer["W"] + layer["b"])
    out = z @ p["dense"][-1]["W"] + p["dense"][-1]["b"]
    raw_bounds = out[:, :2 * n_inst]
    if cfg.band_mode == "center_width":
        bl, bu = ntbn_bounds_center_width(raw_bounds, n_inst)
    else:
        bl, bu = raw_bounds[:, :n_inst], raw_bounds[:, n_inst:2 * n_inst]

    if not cfg.n_states:
        return (bl, bu), state

    rec = out[:, 2 * n_inst:]
    c_new, a_cand, p_cand, e_cand, a_upd, p_upd, e_upd = _split(
        rec, (cfg.classic, cfg.aggregate, cfg.past_repr, cfg.event, cfg.aggregate, cfg.past_repr, cfg.event)
    )
    a_cand = jnp.tanh(a_cand) if cfg.bound_aggr_states else a_cand
    a_upd, p_upd, e_upd = jax.nn.sigmoid(a_upd), _unit(p_upd), _unit(e_upd)
    new_state = jnp.concatenate([
        jnp.tanh(c_new),
        (1.0 - a_upd) * aggr + a_upd * a_cand,
        (1.0 - p_upd) * past + p_upd * p_cand,
        (1.0 - e_upd) * event + e_upd * _unit(e_cand),
    ], axis=1)
    return (bl, bu), new_state


def rollout_ntbn(p: dict, cfg: GymConfigNTBN, batch: dict) -> dict:
    a_cfg, act = cfg.agent, ACTIVATIONS[cfg.agent.activation]
    m = batch["market"]
    payoff, hedges = m["payoff"], m["hedges"]
    n, _, n_inst = hedges.shape
    zeros_i = jnp.zeros((n, n_inst), jnp.float32)
    state0 = jnp.broadcast_to(p["state0"], (n, a_cfg.n_states)) if a_cfg.n_states else jnp.zeros((n, 0), jnp.float32)
    idelta = p["idelta"][None, :] if a_cfg.init_delta else zeros_i
    tm = lambda x: jnp.swapaxes(x, 0, 1)
    xs = (tm(hedges), tm(m["cost"]), tm(m["lbnd_a"]), tm(m["ubnd_a"]), {k: tm(v) for k, v in batch["features"].items()})

    def body(carry, x):
        delta, action, cost, pnl, state, first = carry
        h, c, lb, ub, f = x
        (bl, bu), state = agent_step_ntbn(p, a_cfg, act, f, state, n_inst)
        
        # Policy projection
        delta_target, a_target = ntbn_action(delta, bl, bu)
        
        # Initial trade adjustment if init_delta
        a_target = a_target + idelta * first
        
        # Action clipping according to market limits
        if a_cfg.action_clipping == "none":
            a = a_target
        elif a_cfg.action_clipping == "hard":
            a = jnp.clip(a_target, lb, ub)
        elif a_cfg.action_clipping == "soft":
            a = softclip(a_target, lb, ub)
        else:
            raise ValueError(f"Unknown action_clipping mode: {a_cfg.action_clipping}")
            
        delta = delta + a
        cost = cost + jnp.sum(jnp.abs(a) * c, axis=1)
        pnl = pnl + jnp.sum(a * h, axis=1)
        
        step_bands = jnp.stack([bl, bu], axis=-1)
        return (delta, a, cost, pnl, state, 0.0), (jax.lax.stop_gradient(a), jax.lax.stop_gradient(step_bands))

    z = jnp.zeros_like(payoff)
    (_, _, cost, pnl, _, _), (actions, bands) = jax.lax.scan(body, (zeros_i, zeros_i, z, z, state0, 1.0), xs)
    y = p["y"][0] if cfg.utility != "mean" else 0.0
    y0 = p["y0"][0] if cfg.utility != "mean" else 0.0
    u = utility(cfg.utility, cfg.lmbda, payoff + pnl - cost, y)
    u0 = utility(cfg.utility, cfg.lmbda, payoff, y0)
    sg = jax.lax.stop_gradient
    return dict(
        loss=-u - u0, utility=sg(u), utility0=sg(u0), gains=sg(payoff + pnl - cost),
        payoff=payoff, pnl=sg(pnl), cost=sg(cost),
        actions=jnp.swapaxes(actions, 0, 1),
        bands=jnp.swapaxes(bands, 0, 1)  # (nSamples, nSteps, n_inst, 2)
    )


def loss_ntbn(p: dict, cfg: GymConfigNTBN, batch: dict):
    return jnp.mean(rollout_ntbn(p, cfg, batch)["loss"])


def count_parameters(params: dict) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
