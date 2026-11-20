"""
tests/test_finite_grads_and_smoke.py: Finite gradient checks and end-to-end smoke tests for all 7 groups.
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
        loss_original, loss_ntbn,
        rollout_original, rollout_ntbn
    )
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.model import (
        init_params_original, init_params_ntbn,
        loss_original, loss_ntbn,
        rollout_original, rollout_ntbn
    )
try:
    from core.train import TrainConfig, train_paired
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.train import TrainConfig, train_paired
try:
    from core.metrics import evaluate_rollout_metrics, compute_paired_comparison
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.metrics import evaluate_rollout_metrics, compute_paired_comparison


@pytest.mark.parametrize("group_id", list(ADAPTERS.keys()))
def test_finite_loss_and_gradients(group_id):
    """
    Verifies that initial loss and gradients are finite (no NaN or Inf)
    for both Branch A and Branch B across all 7 notebook groups.
    """
    adapter = ADAPTERS[group_id]
    variant = adapter.variants[0]
    train_batch, val_batch, _, feat_dims, n_inst = adapter.build_datasets(
        seed=42, train_samples=128, val_samples=64, test_samples=50, variant=variant
    )
    gym_a, gym_b, _ = adapter.get_variant_configs(variant)
    
    key = jax.random.PRNGKey(0)
    p_a = init_params_original(key, gym_a, feat_dims, n_inst)
    p_b = init_params_ntbn(key, gym_b, feat_dims, n_inst)
    
    # 1. Branch A
    l_a = loss_original(p_a, gym_a, train_batch)
    g_a = jax.grad(loss_original)(p_a, gym_a, train_batch)
    assert np.isfinite(float(l_a)), f"Branch A loss is non-finite in {group_id}"
    for g in jax.tree_util.tree_leaves(g_a):
        assert np.all(np.isfinite(g)), f"Branch A grad is non-finite in {group_id}"

    # 2. Branch B (NTBN)
    l_b = loss_ntbn(p_b, gym_b, train_batch)
    g_b = jax.grad(loss_ntbn)(p_b, gym_b, train_batch)
    assert np.isfinite(float(l_b)), f"Branch B loss is non-finite in {group_id}"
    for g in jax.tree_util.tree_leaves(g_b):
        assert np.all(np.isfinite(g)), f"Branch B grad is non-finite in {group_id}"


@pytest.mark.parametrize("group_id", list(ADAPTERS.keys()))
def test_smoke_end_to_end_both_branches(group_id):
    """
    Runs a 2-epoch smoke test on both Branch A and Branch B for each of the 7 groups.
    Locks checkpoint on validation loss, then evaluates metrics on test set.
    """
    adapter = ADAPTERS[group_id]
    variant = adapter.variants[0]
    
    # Fast smoke dataset (train=128, val=64, test=100)
    train_batch, val_batch, test_batch, feat_dims, n_inst = adapter.build_datasets(
        seed=101, train_samples=128, val_samples=64, test_samples=100, variant=variant
    )
    gym_a, gym_b, _ = adapter.get_variant_configs(variant)
    
    smoke_cfg = TrainConfig(epochs=2, batch_size=32, learning_rate=0.001)
    
    # Run Branch A
    res_a = train_paired("A", gym_a, smoke_cfg, train_batch, val_batch, key=0, feature_dims=feat_dims, n_inst=n_inst)
    roll_a = rollout_original(res_a.params, gym_a, test_batch)
    met_a = evaluate_rollout_metrics(roll_a)
    
    # Run Branch B (NTBN)
    res_b = train_paired("B", gym_b, smoke_cfg, train_batch, val_batch, key=0, feature_dims=feat_dims, n_inst=n_inst)
    roll_b = rollout_ntbn(res_b.params, gym_b, test_batch)
    met_b = evaluate_rollout_metrics(roll_b)
    
    # Compare
    paired = compute_paired_comparison(met_a, met_b, roll_a, roll_b)
    
    # Verify all output metrics are non-null and finite
    for k, v in met_a.items():
        assert np.isfinite(v), f"Branch A metric '{k}' non-finite in {group_id}"
    for k, v in met_b.items():
        assert np.isfinite(v), f"Branch B metric '{k}' non-finite in {group_id}"
    for k, v in paired.items():
        assert np.isfinite(v), f"Paired metric '{k}' non-finite in {group_id}"
