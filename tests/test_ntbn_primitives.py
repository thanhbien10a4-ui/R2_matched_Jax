"""
tests/test_ntbn_primitives.py: Unit tests for core NTBN mathematical primitives.
"""

import pytest
import numpy as np
import jax.numpy as jnp

try:
    from core.ntbn_primitives import (
        ntbn_clamp, ntbn_bounds_center_width, ntbn_action
    )
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.ntbn_primitives import (
        ntbn_clamp, ntbn_bounds_center_width, ntbn_action
    )


def test_ntbn_clamp_standard():
    bl = jnp.array([0.2, -1.0])
    bu = jnp.array([0.8, 1.0])

    # Case 1: Below lower band -> clamped to bl
    d_below = jnp.array([0.0, -2.0])
    clamped_below = ntbn_clamp(d_below, bl, bu)
    np.testing.assert_allclose(clamped_below, [0.2, -1.0], atol=1e-6)

    # Case 2: Inside band -> remains unchanged
    d_inside = jnp.array([0.5, 0.0])
    clamped_inside = ntbn_clamp(d_inside, bl, bu)
    np.testing.assert_allclose(clamped_inside, [0.5, 0.0], atol=1e-6)

    # Case 3: Above upper band -> clamped to bu
    d_above = jnp.array([1.2, 3.0])
    clamped_above = ntbn_clamp(d_above, bl, bu)
    np.testing.assert_allclose(clamped_above, [0.8, 1.0], atol=1e-6)


def test_ntbn_clamp_inverted_fallback():
    # Inverted band: bl > bu
    bl = jnp.array([1.0])
    bu = jnp.array([0.0])
    d = jnp.array([0.7])

    # Imaki et al. Section 4.1 fallback: (bl + bu) / 2 = 0.5
    clamped = ntbn_clamp(d, bl, bu)
    np.testing.assert_allclose(clamped, [0.5], atol=1e-6)


def test_ntbn_bounds_center_width():
    raw_out = jnp.array([[0.5, -2.0, 0.0, 1.0]])  # center: (0.5, -2.0), width: (0.0, 1.0)
    bl, bu = ntbn_bounds_center_width(raw_out, n_inst=2)
    
    # Check shape
    assert bl.shape == (1, 2)
    assert bu.shape == (1, 2)
    
    # Strictly bl <= bu by construction (softplus is strictly positive)
    assert np.all(bl <= bu)
    np.testing.assert_allclose(0.5 * (bl + bu), raw_out[..., :2], atol=1e-6)


def test_ntbn_action_in_and_out_of_band():
    bl = jnp.array([[0.3]])
    bu = jnp.array([[0.7]])
    
    # Position inside band: delta = 0.5 -> action = 0.0
    delta_in = jnp.array([[0.5]])
    d_next, act = ntbn_action(delta_in, bl, bu)
    np.testing.assert_allclose(d_next, [[0.5]], atol=1e-6)
    np.testing.assert_allclose(act, [[0.0]], atol=1e-6)

    # Position below band: delta = 0.1 -> action = +0.2 to reach 0.3
    delta_below = jnp.array([[0.1]])
    d_next, act = ntbn_action(delta_below, bl, bu)
    np.testing.assert_allclose(d_next, [[0.3]], atol=1e-6)
    np.testing.assert_allclose(act, [[0.2]], atol=1e-6)

    # Position above band: delta = 0.9 -> action = -0.2 to reach 0.7
    delta_above = jnp.array([[0.9]])
    d_next, act = ntbn_action(delta_above, bl, bu)
    np.testing.assert_allclose(d_next, [[0.7]], atol=1e-6)
    np.testing.assert_allclose(act, [[-0.2]], atol=1e-6)
