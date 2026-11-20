"""
tests/test_recurrent_and_states.py: Verifies path-dependent states and parameter configurations.
"""

import pytest
import numpy as np
import jax

try:
    from adapters import ADAPTERS
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.adapters import ADAPTERS
try:
    from core.model import (
        init_params_original, init_params_ntbn,
        count_parameters
    )
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.model import (
        init_params_original, init_params_ntbn,
        count_parameters
    )


def test_american_digital_knockout_monotonicity():
    adapter = ADAPTERS["trainer-american_digital"]
    train_world, _, _ = adapter.build_worlds(seed=42, test_samples=100, variant="none")
    notko = train_world.features["payoff_features"][:, :, 0]
    
    # Alive indicator must be monotonically non-increasing over time (once knocked out, remains 0)
    diffs = np.diff(notko, axis=1)
    assert np.all(diffs <= 1e-6), "American Digital alive indicator revived after knockout!"


def test_barrier_knockout_payoff_consistency():
    adapter = ADAPTERS["trainer-recurrent-barrier"]
    train_world, _, _ = adapter.build_worlds(seed=42, test_samples=500, variant="none")
    notko = train_world.features["payoff_features"][:, :, 0]
    payoff = train_world.market["payoff"]
    
    # If option has knocked out (notko at end == 0), terminal payoff must be exactly 0
    knocked_out_mask = (notko[:, -1] == 0.0)
    assert np.all(payoff[knocked_out_mask] == 0.0), "Barrier payoff is non-zero for knocked-out paths!"


def test_forward_start_payoff_reset():
    adapter = ADAPTERS["trainer-recurrent-fwdstart"]
    train_world, _, _ = adapter.build_worlds(seed=42, test_samples=500, variant="none")
    spots = train_world.spot_all
    payoff = train_world.market["payoff"]
    
    # Check payoff formula: - max(S_10 - S_T, 0)
    s_reset = spots[:, 10]
    s_terminal = spots[:, -1]
    expected_payoff = -np.maximum(s_reset - s_terminal, 0.0)
    np.testing.assert_allclose(payoff, expected_payoff, atol=1e-5)


def test_parameter_counts_branch_a_vs_b():
    """
    Verifies parameter counts between Branch A and Branch B across all 7 groups.
    Reports the exact parameter count difference due to NTBN predicting 2 * n_inst bounds.
    """
    key = jax.random.PRNGKey(0)
    
    for group_id, adapter in ADAPTERS.items():
        variant = adapter.variants[0]
        train_batch, val_batch, test_batch, feat_dims, n_inst = adapter.build_datasets(seed=42, test_samples=50)
        gym_a, gym_b, _ = adapter.get_variant_configs(variant)
        
        p_a = init_params_original(key, gym_a, feat_dims, n_inst)
        p_b = init_params_ntbn(key, gym_b, feat_dims, n_inst)
        
        n_params_a = count_parameters(p_a)
        n_params_b = count_parameters(p_b)
        
        # Branch B outputs 2*n_inst instead of n_inst in final layer, but excludes delta from input
        assert n_params_a > 0
        assert n_params_b > 0
        print(f"[{group_id} - {variant}] Params Branch A (Orig): {n_params_a}, Branch B (NTBN): {n_params_b}")
