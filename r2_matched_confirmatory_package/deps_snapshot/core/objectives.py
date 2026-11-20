"""
core/objectives.py: Optimized Certainty Equivalent (OCE) utilities and risk measures.

Ported from Buehler et al. (2019) / NdQuan (2024), augmented with Rockafellar-Uryasev (2000, 2002)
variational CVaR/VaR evaluation with discrete-tie inverted CDF handling.
"""

from typing import Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp

UTILITIES = ("mean", "exp", "exp2", "vicky", "cvar", "quad")


def _neg(g):
    return jnp.where(g < 0.0, g, 0.0)


def utility(name: str, lmbda: float, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Computes OCE utility u(X + y) - y according to Buehler et al.
    """
    if lmbda < 1e-12:
        name, lmbda = "mean", 1.0
    gains = X + y
    if name == "mean":
        return gains
    if name == "cvar":
        return (1.0 + lmbda) * _neg(gains) - y
    if name == "quad":
        x0 = 1.0 / lmbda
        xx = _neg(gains - x0)
        return -0.5 * lmbda * xx ** 2 + 0.5 * lmbda * x0 ** 2 - y
    if name == "exp":
        inf = jax.lax.stop_gradient(jnp.min(X))
        return (1.0 - jnp.exp(-lmbda * (gains - inf))) / lmbda - y + inf
    if name == "exp2":
        g1 = jnp.where(gains >= 0.0, gains, 0.0)
        g2 = jnp.where(gains <= 0.0, gains, 0.0)
        u1 = (1.0 - jnp.exp(-lmbda * g1)) / lmbda - y
        u2 = g2 - 0.5 * lmbda * g2 * g2 - y
        return jnp.where(gains > 0.0, u1, u2)
    if name == "vicky":
        return (1.0 + lmbda * gains - jnp.sqrt(1.0 + (lmbda * gains) ** 2)) / lmbda - y
    raise ValueError(f"Unknown utility '{name}'. Supported: {UTILITIES}")


def compute_var_cvar(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Exact Rockafellar-Uryasev (2000) variational calculation of VaR_alpha and CVaR_alpha.
    Uses method='inverted_cdf' to avoid discrete ties / atomic probability distortion.
    
    losses: Array of net losses L = - Gains.
    alpha: Confidence level (e.g. 0.95).
    
    Returns:
      (VaR_alpha, CVaR_alpha)
    """
    losses = np.asarray(losses, dtype=np.float64)
    n = len(losses)
    if n == 0:
        return 0.0, 0.0
    
    # Inverted CDF percentile is the true infimum quantile
    var_alpha = float(np.percentile(losses, alpha * 100.0, method="inverted_cdf"))
    excess = np.maximum(losses - var_alpha, 0.0)
    cvar_alpha = float(var_alpha + np.sum(excess) / ((1.0 - alpha) * n))
    return var_alpha, cvar_alpha
