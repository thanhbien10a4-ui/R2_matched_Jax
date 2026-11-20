"""
robust_world.py: Multi-Scenario and Robust Market Simulator for Deep Hedging.

Supports:
1. Canonical Black-Scholes (sigma = constant, e.g. 0.20)
2. Robust Black-Scholes with Volatility Uncertainty (sigma_t ~ U[sigma_min, sigma_max])
   Inspired by Algorithm 1 of Lütkebohmert, Schmidt, Sester (2021)
3. Multi-Scenario Evaluation Generators:
   - Fixed sigma in {0.15, 0.20, 0.25, 0.30}
   - Regime-switching volatility (e.g. low 0.15 for t < 10, high 0.28 for t >= 10)

Maintains strict parity with deephedging.world.simulate when sigma_mode='fixed' and rvol=0.20.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np

@dataclass(frozen=True)
class RobustWorldConfig:
    steps: int = 20
    samples: int = 5000
    seed: int = 42
    dt: float = 1.0 / 50.0  # 0.02
    drift: float = 0.0
    cost_s: float = 0.0
    ubnd_as: float = 5.0
    lbnd_as: float = -5.0
    strike: float = 1.0
    
    # Uncertainty configurations
    sigma_mode: str = "fixed"  # "fixed", "uniform", "regime_switch"
    rvol: float = 0.20
    sigma_min: float = 0.15
    sigma_max: float = 0.25
    vol_seed_offset: int = 54321
    
    # Regime switch parameters
    regime_split_step: int = 10
    regime_sigma_low: float = 0.15
    regime_sigma_high: float = 0.28
    
    # Normalization convention (matches world.py line 115)
    normalize_batch_mean: bool = True


@dataclass
class RobustWorld:
    config: RobustWorldConfig
    market: dict
    features: dict
    spot_all: np.ndarray
    details: dict
    nSamples: int
    nSteps: int
    nInst: int
    dt: float

    @property
    def inst_names(self) -> list:
        return ["spot"]

    @property
    def timeline(self) -> np.ndarray:
        return np.arange(self.nSteps + 1) * self.dt


def simulate_robust_world(c: RobustWorldConfig) -> RobustWorld:
    dtype = np.float32
    nSteps, nSamples, dt = c.steps, c.samples, c.dt
    sqrtDt = math.sqrt(dt)
    
    time_left = np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=dtype) * dt
    sqrt_time_left = np.sqrt(time_left)
    
    # Reproduce identical Brownian motion draws as deephedging.world.simulate
    np.random.seed(c.seed)
    nAll = nSteps + 1  # 21 steps for spot: S_0 ... S_20
    dW = np.random.normal(size=(nSamples, nAll - 1, 4)).astype(dtype) * sqrtDt
    dW_s = dW[:, :, 0]
    
    # Volatility realization
    if c.sigma_mode == "fixed":
        rvol = np.full((nSamples, nAll), c.rvol, dtype=dtype)
    elif c.sigma_mode == "uniform":
        rng_vol = np.random.RandomState(c.seed + c.vol_seed_offset)
        rvol = rng_vol.uniform(low=c.sigma_min, high=c.sigma_max, size=(nSamples, nAll)).astype(dtype)
    elif c.sigma_mode == "static_path":
        # R2: sample once per path, held constant along entire trajectory
        rng_vol = np.random.RandomState(c.seed + c.vol_seed_offset)
        sigma_path = rng_vol.uniform(low=c.sigma_min, high=c.sigma_max, size=(nSamples, 1)).astype(dtype)
        rvol = np.broadcast_to(sigma_path, (nSamples, nAll)).copy()
    elif c.sigma_mode == "regime_switch":
        rvol = np.empty((nSamples, nAll), dtype=dtype)
        # Low volatility for steps before split, High volatility after split
        split = c.regime_split_step
        rvol[:, :split] = c.regime_sigma_low
        rvol[:, split:] = c.regime_sigma_high
    else:
        raise ValueError(f"Unknown sigma_mode: {c.sigma_mode}")

    spot = np.zeros((nSamples, nAll), dtype=dtype)
    rdrift = np.full((nSamples, nAll), c.drift, dtype=dtype)
    spot[:, 0] = 1.0
    expdriftdt = np.exp(c.drift * dt)
    
    for j in range(1, nAll):
        # Euler-Maruyama / Geometric Brownian Motion step
        spot[:, j] = spot[:, j - 1] * np.exp(
            rdrift[:, j - 1] * dt
            + rvol[:, j - 1] * dW_s[:, j - 1]
            - 0.5 * (rvol[:, j - 1] ** 2) * dt
        )
        if c.normalize_batch_mean:
            # Buehler / JAX forward-mean calibration across batch
            spot[:, j] *= expdriftdt / np.mean(spot[:, j])
            
    # Sort paths by terminal spot (matches world.py)
    ixs = np.argsort(spot[:, nSteps])
    spot = spot[ixs, :]
    rdrift = rdrift[ixs, :]
    rvol = rvol[ixs, :]
    
    # Market & Hedges
    dS = spot[:, nSteps][:, np.newaxis] - spot[:, :nSteps]
    cost_dS = spot[:, :nSteps] * c.cost_s
    hedges = dS[:, :, np.newaxis]
    cost = cost_dS[:, :, np.newaxis]
    price = spot[:, :nSteps, np.newaxis]
    
    ubnd_a = np.full((nSamples, nSteps, 1), c.ubnd_as, dtype=dtype)
    lbnd_a = np.full((nSamples, nSteps, 1), c.lbnd_as, dtype=dtype)
    
    # Payoff for ATM Call (written option: short position => negative payoff)
    payoff = -np.maximum(spot[:, -1] - c.strike, 0.0)
    
    per_step = dict(
        cost=cost,
        price=price,
        ubnd_a=ubnd_a,
        lbnd_a=lbnd_a,
        time_left=np.full((nSamples, nSteps, 1), time_left[np.newaxis, :, np.newaxis], dtype=dtype),
        sqrt_time_left=np.full((nSamples, nSteps, 1), sqrt_time_left[np.newaxis, :, np.newaxis], dtype=dtype),
        spot=spot[:, :nSteps, np.newaxis],
        ivol=rvol[:, :nSteps, np.newaxis]
    )
    
    market = dict(
        hedges=hedges.astype(dtype),
        cost=cost.astype(dtype),
        ubnd_a=ubnd_a.astype(dtype),
        lbnd_a=lbnd_a.astype(dtype),
        payoff=payoff.astype(dtype)
    )
    
    return RobustWorld(
        config=c,
        market=market,
        features=per_step,
        spot_all=spot[:, :nSteps + 1],
        details=dict(drift=rdrift, rvol=rvol),
        nSamples=nSamples,
        nSteps=nSteps,
        nInst=1,
        dt=dt
    )
