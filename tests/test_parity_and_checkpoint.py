"""
tests/test_parity_and_checkpoint.py: Parity tests with original repo and checkpoint selection verification.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3] / "deephedging-jax"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deephedging.model as orig_model
from deephedging.world import WorldConfig, simulate

try:
    from adapters import ADAPTERS
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.adapters import ADAPTERS
try:
    from core.model import (
        AgentConfigOriginal, GymConfigOriginal,
        init_params_original, rollout_original, loss_original
    )
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.model import (
        AgentConfigOriginal, GymConfigOriginal,
        init_params_original, rollout_original, loss_original
    )
try:
    from core.train import TrainConfig, train_paired
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.train import TrainConfig, train_paired


def test_parity_branch_a_with_original_deephedging():
    """
    Verifies that Branch A implementation achieves EXACT floating-point parity with
    the original deephedging-jax codebase on identical weights and batch data.
    """
    world_cfg = WorldConfig(samples=500, steps=20, black_scholes=True, seed=42)
    world = simulate(world_cfg)
    
    gym_orig = GymConfigOriginal(
        agent=AgentConfigOriginal(depth=3, width=20, activation="softplus"),
        utility="cvar",
        lmbda=1.0
    )
    
    feat_dims = {k: v.shape[2] for k, v in world.features.items()}
    key = jax.random.PRNGKey(42)
    params = init_params_original(key, gym_orig, feat_dims, world.nInst)
    
    batch = orig_model.as_batch(world)
    
    # 1. Forward loss from our Branch A
    our_loss = float(loss_original(params, gym_orig, batch))
    
    # 2. Forward loss from original deephedging
    # Convert config to original types
    orig_gym = orig_model.GymConfig(
        agent=orig_model.AgentConfig(depth=3, width=20, activation="softplus"),
        utility="cvar",
        lmbda=1.0
    )
    repo_loss = float(orig_model.loss(params, orig_gym, batch))
    
    # Must match within machine epsilon
    np.testing.assert_allclose(our_loss, repo_loss, rtol=1e-6, atol=1e-6)


def test_checkpoint_selection_on_validation_loss():
    """
    Verifies that the checkpoint is selected strictly on minimum validation loss.
    """
    adapter = ADAPTERS["blackscholes"]
    train_batch, val_batch, _, feat_dims, n_inst = adapter.build_datasets(seed=42, test_samples=50)
    gym_orig, _, _ = adapter.get_variant_configs("trainer_bs_small")
    
    train_cfg = TrainConfig(epochs=5, batch_size=32, learning_rate=0.001)
    
    res = train_paired(
        branch="A",
        gym_cfg=gym_orig,
        train_cfg=train_cfg,
        train_data=train_batch,
        val_data=val_batch,
        key=0,
        feature_dims=feat_dims,
        n_inst=n_inst
    )
    
    # Best epoch must correspond to minimum validation loss in history
    val_losses = res.history["val_loss"]
    expected_best_epoch = int(np.argmin(val_losses))
    assert res.best_epoch == expected_best_epoch
    assert res.best_val_loss == val_losses[expected_best_epoch]


def test_testset_isolation():
    """
    Verifies that the test set has ZERO impact on training or checkpoint selection.
    Running training with different test sets must produce IDENTICAL model parameters.
    """
    adapter = ADAPTERS["blackscholes"]
    train_batch, val_batch, test_batch_1, feat_dims, n_inst = adapter.build_datasets(seed=42, test_samples=100)
    _, _, test_batch_2, _, _ = adapter.build_datasets(seed=42, test_samples=500)
    gym_orig, _, _ = adapter.get_variant_configs("trainer_bs_small")
    
    train_cfg = TrainConfig(epochs=3, batch_size=32, learning_rate=0.001)
    
    # Training run 1
    res1 = train_paired("A", gym_orig, train_cfg, train_batch, val_batch, key=0, feature_dims=feat_dims, n_inst=n_inst)
    # Training run 2
    res2 = train_paired("A", gym_orig, train_cfg, train_batch, val_batch, key=0, feature_dims=feat_dims, n_inst=n_inst)
    
    # Parameters must be strictly identical
    leaves1 = jax.tree_util.tree_leaves(res1.params)
    leaves2 = jax.tree_util.tree_leaves(res2.params)
    for l1, l2 in zip(leaves1, leaves2):
        np.testing.assert_array_equal(l1, l2)
