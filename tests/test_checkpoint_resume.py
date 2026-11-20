"""
tests/test_checkpoint_resume.py: Unit tests for atomic checkpointing and auto-resume.
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import jax
import jax.numpy as jnp

try:
    from core.train import TrainConfig, train_paired
    from configs.catalog import get_config_spec
    from adapters import get_adapter
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.train import TrainConfig, train_paired
    from experiments.jax_ntbn_7notebooks_original_scale.configs.catalog import get_config_spec
    from experiments.jax_ntbn_7notebooks_original_scale.adapters import get_adapter


def test_checkpoint_saving_and_resumption():
    temp_dir = tempfile.mkdtemp(prefix="test_chk_")
    try:
        spec = get_config_spec("trainer__bs")
        adapter = get_adapter(spec.group_id)
        gym_a, gym_b, train_cfg_base = adapter.get_variant_configs(spec.variant)

        # Build small dataset for fast test
        train_batch, val_batch, test_batch, feat_dims, n_inst = adapter.build_datasets(
            seed=42, train_samples=128, val_samples=64, test_samples=64
        )

        train_cfg_full = TrainConfig(
            epochs=6,
            batch_size=32,
            learning_rate=0.001,
            seed=42
        )

        # 1. Run uninterrupted for 6 epochs
        res_full = train_paired(
            "B", gym_b, train_cfg_full, train_batch, val_batch,
            key=42, feature_dims=feat_dims, n_inst=n_inst
        )
        assert res_full.completed is True
        assert len(res_full.history["loss"]) == 6

        # 2. Run first session: target 6 epochs, but set max_session_seconds very small or simulate session stop at 3 epochs
        train_cfg_part = TrainConfig(
            epochs=6,
            batch_size=32,
            learning_rate=0.001,
            seed=42
        )
        chk_prefix = "test_run_resume"

        # Simulate first session stopping at epoch 3 by running 3 epochs with checkpoint
        train_cfg_3ep = TrainConfig(
            epochs=3,
            batch_size=32,
            learning_rate=0.001,
            seed=42
        )
        res_part = train_paired(
            "B", gym_b, train_cfg_3ep, train_batch, val_batch,
            key=42, feature_dims=feat_dims, n_inst=n_inst,
            checkpoint_dir=temp_dir,
            checkpoint_prefix=chk_prefix,
            checkpoint_every=1,
            resume=True
        )
        assert res_part.completed is True
        assert len(res_part.history["loss"]) == 3

        # Verify checkpoint file exists
        latest_file = os.path.join(temp_dir, f"{chk_prefix}_latest.pkl")
        assert os.path.exists(latest_file), "Latest checkpoint must be written to disk"

        # 3. Resume second session: target 6 epochs, loading checkpoint from epoch 3
        res_resumed = train_paired(
            "B", gym_b, train_cfg_full, train_batch, val_batch,
            key=42, feature_dims=feat_dims, n_inst=n_inst,
            checkpoint_dir=temp_dir,
            checkpoint_prefix=chk_prefix,
            checkpoint_every=1,
            resume=True
        )
        assert res_resumed.completed is True
        # Total history must now contain 6 epochs
        assert len(res_resumed.history["loss"]) == 6, f"Expected 6 total epochs in resumed history, got {len(res_resumed.history['loss'])}"
        
        # Verify loss values from resumed run closely match uninterrupted run
        # Since permutations and Adam state are saved and restored identically,
        # final val loss must match within numerical tolerance
        np.testing.assert_allclose(
            res_resumed.history["val_loss"][-1],
            res_full.history["val_loss"][-1],
            atol=1e-4,
            err_msg="Resumed training must reach identical loss to uninterrupted run"
        )
        print("Checkpoint resume test PASSED with identical loss convergence!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
