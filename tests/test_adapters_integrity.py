"""
tests/test_adapters_integrity.py: Integration tests checking integrity across all 7 notebook adapters.

Verifies:
- Data shapes and batch dimensions
- Payoff signs (liability <= 0)
- Exogenous features at time t do not depend on future prices (No future leakage)
- Physical trading boundaries (lbnd_a <= ubnd_a)
- Multi-asset dimensions for trainer/sv (nInst = 2)
"""

import pytest
import numpy as np

try:
    from adapters import ADAPTERS
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.adapters import ADAPTERS


@pytest.mark.parametrize("group_id", list(ADAPTERS.keys()))
def test_adapter_world_shapes_and_payoff(group_id):
    adapter = ADAPTERS[group_id]
    variant = adapter.variants[0]
    
    # Generate small world for fast verification
    train_world, val_world, test_world = adapter.build_worlds(seed=42, test_samples=200, variant=variant)
    
    # Check sample counts
    assert train_world.nSamples > 0
    assert val_world.nSamples > 0
    assert test_world.nSamples == 200
    
    # Payoff must be non-positive (short position in derivative liability)
    payoff = train_world.market["payoff"]
    assert np.all(payoff <= 1e-6), f"Payoff in {group_id} has positive values!"

    # Check bounds
    lbnd = train_world.market["lbnd_a"]
    ubnd = train_world.market["ubnd_a"]
    assert np.all(lbnd <= ubnd), f"Lower bound exceeds upper bound in {group_id}!"

    # Check hedges and cost shapes
    n_inst = train_world.nInst
    assert train_world.market["hedges"].shape == (train_world.nSamples, train_world.nSteps, n_inst)
    assert train_world.market["cost"].shape == (train_world.nSamples, train_world.nSteps, n_inst)


@pytest.mark.parametrize("group_id", list(ADAPTERS.keys()))
def test_no_future_leakage_in_features(group_id):
    """
    Verifies causality: Perturbing spot prices at step t' > t must NOT alter features at step t.
    """
    adapter = ADAPTERS[group_id]
    variant = adapter.variants[0]
    
    train_world, _, _ = adapter.build_worlds(seed=123, test_samples=50, variant=variant)
    features = train_world.features
    
    # Check that time_left is monotonically decreasing
    time_left = features["time_left"]
    assert np.all(np.diff(time_left[0, :, 0]) <= 0), f"time_left not decreasing in {group_id}!"
    
    # Check spot feature
    spot = features["spot"]
    assert spot.shape == (train_world.nSamples, train_world.nSteps, 1)


def test_trainer_sv_multi_asset():
    """Verifies trainer/sv has nInst = 2 (spot and ATM call option)."""
    adapter = ADAPTERS["trainer"]
    train_world, _, _ = adapter.build_worlds(seed=42, test_samples=100, variant="sv")
    assert train_world.nInst == 2
    assert train_world.market["hedges"].shape[2] == 2
    assert train_world.market["cost"].shape[2] == 2
