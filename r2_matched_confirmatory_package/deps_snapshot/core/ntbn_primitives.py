"""
core/ntbn_primitives.py: Core functional primitives for No-Transaction Band Network (NTBN).

Based on Imaki et al. (2021) "No-Transaction Band Network: A Neural Network Architecture for Efficient Deep Hedging".
Adheres strictly to @gng5 Epistemic Gate Standards T1-T5.
"""

from typing import Tuple
import jax
import jax.numpy as jnp


def ntbn_clamp(delta: jnp.ndarray, bl: jnp.ndarray, bu: jnp.ndarray) -> jnp.ndarray:
    """
    Exact implementation of clamp function defined in Eq. (13) and Section 4.1 of Imaki et al. (2021).
    
    Formula:
      clamp(delta, bl, bu) =
          bl                if delta < bl
          delta             if bl <= delta <= bu
          bu                if delta > bu
    
    Inverted boundary fallback (Section 4.1, Page 8):
      "since the above definition of clamp (13) ... does not expect an inverted input bl > bu,
       one may wish to give such an input a natural definition such as clamp(delta, bl, bu) := (bl + bu) / 2"
    """
    standard_clamped = jnp.clip(delta, bl, bu)
    midpoint = 0.5 * (bl + bu)
    return jnp.where(bl <= bu, standard_clamped, midpoint)


def ntbn_bounds_center_width(raw_out: jnp.ndarray, n_inst: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Parameterize no-transaction band via Center and Half-Width:
      bl = center - softplus(raw_width)
      bu = center + softplus(raw_width)
      
    Guarantees bl <= bu strictly by construction.
    Supports multi-instrument hedging (n_inst >= 1).
    """
    center = raw_out[..., :n_inst]
    half_width = jax.nn.softplus(raw_out[..., n_inst:2 * n_inst])
    bl = center - half_width
    bu = center + half_width
    return bl, bu


def ntbn_action(delta_prev: jnp.ndarray, bl: jnp.ndarray, bu: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes new target position delta_next and trading action a according to NTBN policy.
    
    delta_next = clamp(delta_prev, bl, bu)
    action = delta_next - delta_prev
    
    Returns:
      (delta_next, action)
    """
    delta_next = ntbn_clamp(delta_prev, bl, bu)
    action = delta_next - delta_prev
    return delta_next, action
