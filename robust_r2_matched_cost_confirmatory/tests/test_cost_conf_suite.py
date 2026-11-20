"""
tests/test_cost_conf_suite.py:
Comprehensive pre-run verification suite for robust_r2_matched_cost_confirmatory (c_s = 0.0010).
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

TEST_DIR = Path(__file__).resolve().parent
COST_CONF_DIR = TEST_DIR.parent
EXP_DIR = COST_CONF_DIR.parent
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"

for _p in (str(COST_CONF_DIR), str(BC_DIR), str(R3_DIR), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bc_models import (
    aref_128_step_loss,
    count_param_breakdown,
    init_full_params,
    r2_matched_step_loss,
    rollout_policy_paths,
)
from core.model import softclip


def _inv_softclip(target: float, lb: float = -5.0, ub: float = 5.0) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(softclip(jnp.asarray(mid, dtype=jnp.float32), lb, ub)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
from cost_design import (
    COST_S_LOCKED,
    EPS_NONINFERIORITY_OCE020,
    STEPS_PER_EPOCH_LOCKED,
    get_locked_branch_configs,
    get_locked_train_config,
)
from cost_keys import (
    COST_LOCKED_COHORT_SEEDS,
    COST_SMOKE_SEED,
    derive_cost_seed_bundle,
    verify_cost_key_isolation,
)
from cost_run import (
    build_cost_train_and_val_for_seed,
    classify_cost_run_exception,
    resolve_locked_cost_seeds,
)
from cost_stats import analyze_cost_confirmatory_pairs
from cost_train import train_cost_config
from cost_world_and_metrics import build_cost_scenario_world, evaluate_cost_rollout_metrics


def _exact_numpy_rollout_reference(actions: np.ndarray, S_path: np.ndarray, cost_s: float) -> dict:
    """Independent scalar-by-step reference accounting check (K=1.0, short ATM call)."""
    # S_path shape: (B, 21), actions shape: (B, 20)
    dS = S_path[:, 20:21] - S_path[:, :20]
    c_arr = S_path[:, :20] * cost_s
    payoff = -np.maximum(S_path[:, 20] - 1.0, 0.0)
    cost = np.sum(np.abs(actions) * c_arr, axis=1)
    pnl = np.sum(actions * dS, axis=1)
    gross = payoff + pnl
    gains = gross - cost
    return {"payoff": payoff, "pnl": pnl, "cost": cost, "gross": gross, "gains": gains}


def test_1_cost_accounting_short_trajectories_and_metrics() -> None:
    cfgs = get_locked_branch_configs()
    gym_cfg = cfgs["gym_cfg"]
    feat_dims = cfgs["feature_dims"]

    b_10bps = build_cost_scenario_world("fixed_020", 123456, samples=64, cost_s=0.0010)
    b_0bps = build_cost_scenario_world("fixed_020", 123456, samples=64, cost_s=0.0)
    assert float(np.max(np.asarray(b_0bps["market"]["cost"]))) == 0.0
    assert float(np.min(np.asarray(b_10bps["market"]["cost"]))) > 0.0

    # Verify cost array in market exactly equals S[:, :20] * 0.0010 (shape (B, 20, 1))
    S_rec = np.asarray(b_10bps["market"]["cost"])[:, :, 0] / 0.0010
    assert np.allclose(S_rec[:, 0], 1.0, atol=1e-6), "Initial spot S_0 must equal 1.0"

    # Case A: No-trade exact zero-action policy (out = inv_softclip(0.0) -> softclip(out) == 0.0)
    b_zero = _inv_softclip(0.0, -5.0, 5.0)
    b_open = _inv_softclip(0.6, -5.0, 5.0)
    params0 = init_full_params(jax.random.PRNGKey(1), gym_cfg, feat_dims, k_oce=1)
    params_zero = jax.tree_util.tree_map(jnp.zeros_like, params0)
    params_zero["policy"]["dense"][-1]["b"] = jnp.full_like(params_zero["policy"]["dense"][-1]["b"], b_zero)
    pr_zero = rollout_policy_paths(params_zero["policy"], gym_cfg, b_10bps)
    assert np.allclose(np.asarray(pr_zero["actions"]), 0.0, atol=1e-6)
    assert np.allclose(np.asarray(pr_zero["cost"]), 0.0, atol=1e-7)
    assert np.allclose(np.asarray(pr_zero["gains"]), np.asarray(pr_zero["payoff"]), atol=1e-6)

    # Case B: Open a_0 = +0.6 at j=0 and hold through j=19 -> cost = 0.6 * 1.0 * 0.0010 = 0.0006, NO settlement cost
    params_hold = jax.tree_util.tree_map(jnp.zeros_like, params0)
    params_hold["policy"]["dense"][-1]["b"] = jnp.full_like(params_hold["policy"]["dense"][-1]["b"], b_zero)
    params_hold["policy"]["idelta"] = jnp.full_like(params_hold["policy"]["idelta"], b_open - b_zero)
    pr_hold = rollout_policy_paths(params_hold["policy"], gym_cfg, b_10bps)
    assert np.allclose(np.asarray(pr_hold["actions"])[:, 0, 0], 0.6, atol=1e-5)
    assert np.allclose(np.asarray(pr_hold["actions"])[:, 1:, 0], 0.0, atol=1e-6)
    assert np.allclose(np.asarray(pr_hold["cost"]), 0.0006, atol=1e-6)

    # Case C: Time-varying trajectory (both positive/negative trades & delta reversal) vs numpy reference
    params_rev = jax.tree_util.tree_map(jnp.zeros_like, params0)
    params_rev["policy"]["dense"][0]["W"] = params_rev["policy"]["dense"][0]["W"].at[2, 0].set(20.0)
    params_rev["policy"]["dense"][0]["b"] = params_rev["policy"]["dense"][0]["b"].at[0].set(-4.5)
    for l_idx in (1, 2, 3, 4):
        params_rev["policy"]["dense"][l_idx]["W"] = params_rev["policy"]["dense"][l_idx]["W"].at[0, 0].set(1.0)
    params_rev["policy"]["dense"][5]["W"] = params_rev["policy"]["dense"][5]["W"].at[0, 0].set(1.0)
    params_rev["policy"]["dense"][5]["b"] = jnp.full_like(
        params_rev["policy"]["dense"][5]["b"], _inv_softclip(-0.8, -5.0, 5.0)
    )
    pr_rand_10 = rollout_policy_paths(params_rev["policy"], gym_cfg, b_10bps)
    pr_rand_0 = rollout_policy_paths(params_rev["policy"], gym_cfg, b_0bps)
    # Construct S[:, :21] directly from b_10bps market fields
    # Note market['hedges'][:, j, 0] = S_20 - S_j; S_0 = 1.0 => S_20 = 1.0 + hedges[:, 0, 0], S_j = S_20 - hedges[:, j, 0]
    dS = np.asarray(b_10bps["market"]["hedges"])[:, :, 0]
    S20 = 1.0 + dS[:, 0:1]
    S_head = S20 - dS
    S_full = np.concatenate([S_head, S20], axis=1)
    ref_10 = _exact_numpy_rollout_reference(np.asarray(pr_rand_10["actions"])[:, :, 0], S_full, 0.0010)
    assert np.allclose(np.asarray(pr_rand_10["cost"]), ref_10["cost"], atol=1e-6)
    assert np.allclose(np.asarray(pr_rand_10["pnl"]), ref_10["pnl"], atol=1e-5)
    assert np.allclose(np.asarray(pr_rand_10["gains"]), ref_10["gains"], atol=1e-5)
    assert np.allclose(
        np.asarray(pr_rand_0["gains"]) - np.asarray(pr_rand_10["gains"]),
        np.asarray(pr_rand_10["cost"]),
        atol=1e-6,
    ), "Cost subtracted twice or inconsistently!"
    acts_np = np.asarray(pr_rand_10["actions"])
    dels_np = np.asarray(pr_rand_10["deltas"])
    assert np.min(acts_np) < -0.01 and np.max(acts_np) > 0.01, "Expected sign-reversing trades in Case C"
    assert np.min(dels_np) < -0.01 and np.max(dels_np) > 0.01, "Expected both long and short deltas in Case C"

    # Case D: Hand-computed Section 8 metrics on exact synthetic rollout dict
    # Path 0: active actions [0.4, -0.2, 0.0 ...] -> active_steps=2, turnover=0.6, avg_active=0.3
    # Path 1: all inactive [0.0 ...] -> active_steps=0, zero_active_path_frac = 0.5
    act_syn = np.zeros((2, 20, 1), dtype=np.float32)
    act_syn[0, 0, 0] = 0.4
    act_syn[0, 1, 0] = -0.2
    delta_syn = np.cumsum(act_syn, axis=1)
    cost_syn = np.array([0.0006, 0.0], dtype=np.float32)
    payoff_syn = np.array([-0.05, -0.02], dtype=np.float32)
    pnl_syn = np.array([0.03, 0.0], dtype=np.float32)
    gains_syn = payoff_syn + pnl_syn - cost_syn
    synth_roll = {
        "payoff": jnp.asarray(payoff_syn),
        "pnl": jnp.asarray(pnl_syn),
        "cost": jnp.asarray(cost_syn),
        "gains": jnp.asarray(gains_syn),
        "actions": jnp.asarray(act_syn),
        "deltas": jnp.asarray(delta_syn),
        "hedged_loss": jnp.asarray(-gains_syn),
        "unhedged_loss": jnp.asarray(-payoff_syn),
        "loss": jnp.asarray(-gains_syn),
        "utility": jnp.asarray(gains_syn * 0.0),
        "utility0": jnp.asarray(payoff_syn * 0.0),
    }
    m = evaluate_cost_rollout_metrics(synth_roll)
    assert abs(m["active_steps_gt1e5"] - 1.0) < 1e-12
    assert abs(m["immobility_rate"] - (38.0 / 40.0)) < 1e-12
    assert abs(m["mean_trade_size_active_pooled"] - 0.3) < 1e-7
    assert abs(m["mean_trade_size_active_pathwise"] - 0.3) < 1e-7
    assert abs(m["zero_active_path_frac"] - 0.5) < 1e-12
    assert m["cost_accounting_residual"] < 1e-7


def test_2_prng_isolation_global_state_and_b3_flaw_elimination() -> None:
    iso = verify_cost_key_isolation()
    assert iso["isolation_passed"] is True, f"Isolation audit failed: {iso}"
    assert iso["internal_collision_count"] == 0
    assert iso["historical_collisions"] == []
    assert iso["b3_adjacent_val_train_collisions"] == []
    assert iso["vol_offset_cross_collisions"] == []
    assert COST_SMOKE_SEED == 5999
    assert COST_LOCKED_COHORT_SEEDS == list(range(5001, 5025))

    # Global np.random state must NOT be polluted by build_cost_scenario_world
    np.random.seed(987654)
    expected_draw = float(np.random.normal())
    np.random.seed(987654)
    b1 = build_cost_scenario_world("fixed_020", iso["bundles"][5001]["val_dW"]["fixed_020"], samples=5000, cost_s=0.0010)
    actual_draw = float(np.random.normal())
    assert expected_draw == actual_draw, "build_cost_scenario_world leaked/mutated global np.random!"

    # Order invariance: calling in reverse order produces bit-identical batch
    b2 = build_cost_scenario_world("fixed_020", iso["bundles"][5001]["val_dW"]["fixed_020"], samples=5000, cost_s=0.0010)
    assert np.array_equal(np.asarray(b1["market"]["payoff"]), np.asarray(b2["market"]["payoff"]))
    assert np.array_equal(np.asarray(b1["market"]["cost"]), np.asarray(b2["market"]["cost"]))

    # Direct B3 regression test: Val(5001) vs Train(5002)
    # Note simulate_robust_world sorts paths by terminal S_T, so raw payoffs correlate as order statistics;
    # the Brownian-bridge log-spot residual orthogonal to S_T must have near-zero correlation (~0, not 1.0 like B3)!
    val_5001 = b1
    train_5002 = build_cost_scenario_world("fixed_020", iso["bundles"][5002]["train_dW"]["fixed_020"], samples=5000, cost_s=0.0010)
    def _bridge_flat(batch: dict) -> np.ndarray:
        c = np.asarray(batch["market"]["cost"])[:, :, 0] / 0.0010  # S_0 .. S_19
        dS = np.asarray(batch["market"]["hedges"])[:, 0:1, 0]      # S_20 - S_0
        logS = np.log(c[:, 1:])                                    # steps 1..19
        logS20 = np.log(1.0 + dS)
        frac = np.arange(1, 20, dtype=np.float64)[None, :] / 20.0
        return (logS - frac * logS20).ravel()

    br_v = _bridge_flat(val_5001)
    br_t = _bridge_flat(train_5002)
    corr_bridge = float(np.corrcoef(br_v, br_t)[0, 1])
    assert abs(corr_bridge) < 0.015, f"Suspicious Brownian-bridge correlation Val(5001) vs Train(5002): {corr_bridge}"
    assert np.max(np.abs(br_v - br_t)) > 0.1, "Val(5001) and Train(5002) generated near-identical paths!"

    # Also verify Val(5999) vs Train(5999) and Val(5999) vs ConfTest(5999) on SMOKE seed 5999 ONLY
    # (confirmatory [5001..5040] conf_test_shared_dW is NEVER simulated during unit tests)
    b_smoke = iso["bundles"][COST_SMOKE_SEED]
    br_vs = _bridge_flat(build_cost_scenario_world("fixed_020", b_smoke["val_dW"]["fixed_020"], samples=5000, cost_s=0.0010))
    br_ts = _bridge_flat(build_cost_scenario_world("fixed_020", b_smoke["train_dW"]["fixed_020"], samples=5000, cost_s=0.0010))
    br_cs = _bridge_flat(build_cost_scenario_world("fixed_020", b_smoke["conf_test_shared_dW"], samples=5000, cost_s=0.0010))
    assert abs(float(np.corrcoef(br_vs, br_ts)[0, 1])) < 0.015
    assert abs(float(np.corrcoef(br_vs, br_cs)[0, 1])) < 0.015


def test_3_cost_in_training_val_test_and_2ep_smoke_parity() -> None:
    cfgs = get_locked_branch_configs()
    gym_cfg = cfgs["gym_cfg"]
    feat_dims = cfgs["feature_dims"]
    tcfg = replace(get_locked_train_config(), epochs=2)

    bundle = derive_cost_seed_bundle(COST_SMOKE_SEED)
    train_10, val_10 = build_cost_train_and_val_for_seed(bundle, cost_s=0.0010)
    train_0, _ = build_cost_train_and_val_for_seed(bundle, cost_s=0.0)

    p_init_a = init_full_params(jax.random.PRNGKey(bundle["init_policy"]), gym_cfg, feat_dims, k_oce=1)
    p_init_r2 = init_full_params(jax.random.PRNGKey(bundle["init_policy"]), gym_cfg, feat_dims, k_oce=4)
    # Bit-identical policy initialization
    max_diff = max(
        float(jnp.max(jnp.abs(l1 - l2)))
        for l1, l2 in zip(jax.tree_util.tree_leaves(p_init_a["policy"]), jax.tree_util.tree_leaves(p_init_r2["policy"]))
    )
    assert max_diff == 0.0
    assert count_param_breakdown(p_init_a)["grand_total_params"] == 6764
    assert count_param_breakdown(p_init_r2)["grand_total_params"] == 6770

    # Verify cost=0.0010 strictly affects gradient wrt policy vs cost=0.0
    sub_10 = [train_10["fixed_015"], train_10["fixed_020"], train_10["fixed_025"], train_10["rs_015_025"]]
    sub_0 = [train_0["fixed_015"], train_0["fixed_020"], train_0["fixed_025"], train_0["rs_015_025"]]
    l10, g10 = jax.value_and_grad(r2_matched_step_loss)(p_init_r2, gym_cfg, sub_10)
    l0, g0 = jax.value_and_grad(r2_matched_step_loss)(p_init_r2, gym_cfg, sub_0)
    assert abs(float(l10 - l0)) > 1e-5
    grad_diff = max(
        float(jnp.max(jnp.abs(x1 - x0)))
        for x1, x0 in zip(jax.tree_util.tree_leaves(g10["policy"]), jax.tree_util.tree_leaves(g0["policy"]))
    )
    assert grad_diff > 1e-5, f"Cost did not change policy gradients! grad_diff={grad_diff}"

    # Verify 2-epoch training executes exact steps & paths under positive cost
    with tempfile.TemporaryDirectory() as tmpd:
        rec = train_cost_config(
            "A_DECAY", bundle, train_10, val_10, gym_cfg, tcfg, feat_dims, Path(tmpd) / "A"
        )
        assert rec["actual_update_steps"] == 2 * STEPS_PER_EPOCH_LOCKED
        assert rec["actual_paths_processed"] == 2 * STEPS_PER_EPOCH_LOCKED * 128
        assert rec["val_cost_mean_020_at_best"] > 0.0
        assert all(v > 0.0 for v in rec["train_cost_means"].values())


def test_4_iut_stats_and_seed_offset_guards() -> None:
    planned = list(range(5001, 5025))
    # Case 1: Clean synthetic pass
    pass_rows = []
    for s in planned:
        pass_rows.append(
            {
                "seed": s,
                "ckpt_kind": "best_val",
                "scenario": "fixed_030",
                "delta_he_R2_minus_A": -0.0035 + ((s % 5) - 2) * 0.0002,
                "delta_oce_R2_minus_A": -0.0008,
            }
        )
        pass_rows.append(
            {
                "seed": s,
                "ckpt_kind": "best_val",
                "scenario": "fixed_020",
                "delta_he_R2_minus_A": 0.00001,
                "delta_oce_R2_minus_A": 0.00002 + ((s % 5) - 2) * 0.00001,
            }
        )
    res_pass = analyze_cost_confirmatory_pairs(pass_rows, planned_seeds=planned, eps_margin=EPS_NONINFERIORITY_OCE020)
    assert res_pass["verdict"] == "PASS_CONFIRMATORY_IUT"
    assert res_pass["confirmatory_pass"] is True

    # Case 2: Fail when H1b breaches eps (+0.00010) even if H1a strongly passes
    fail_rows = [dict(r) for r in pass_rows]
    for r in fail_rows:
        if r["scenario"] == "fixed_020":
            r["delta_oce_R2_minus_A"] = 0.00012 + (r["seed"] % 3) * 0.00001
    res_fail = analyze_cost_confirmatory_pairs(fail_rows, planned_seeds=planned, eps_margin=EPS_NONINFERIORITY_OCE020)
    assert res_fail["verdict"] == "FAIL_CONFIRMATORY_IUT"
    assert res_fail["confirmatory_pass"] is False

    # Case 3: Gate blocks PASS if any planned seed is NaN (ALGORITHM_FAILURE) or missing
    nan_rows = [dict(r) for r in pass_rows]
    nan_rows[0]["delta_he_R2_minus_A"] = float("nan")
    res_nan = analyze_cost_confirmatory_pairs(nan_rows, planned_seeds=planned, eps_margin=EPS_NONINFERIORITY_OCE020)
    assert res_nan["verdict"] == "INCOMPLETE_OR_FAILED_AUDIT"
    assert res_nan["confirmatory_pass"] is False

    # Case 4: --seed-offset lock & resume preservation
    try:
        resolve_locked_cost_seeds(n_seeds=24, seed_offset=1, locked_n_total=24)
        raise AssertionError("Should have blocked slice > 5024!")
    except ValueError:
        pass

    existing = [
        {"seed": 5001, "status": "COMPLETED"},
        {"seed": 5002, "status": "FAILED", "error_category": "ALGORITHM_FAILURE"},
        {"seed": 5003, "status": "FAILED", "error_category": "INFRA_ERROR"},
    ]
    to_run, info = resolve_locked_cost_seeds(
        n_seeds=24, seed_offset=0, locked_n_total=24, existing_seed_records=existing
    )
    assert 5001 not in to_run and 5001 in info["skipped_completed"]
    assert 5002 not in to_run and 5002 in info["preserved_algo_failures"]
    assert 5003 in to_run and 5003 in info["retried_infra"]
    assert 5024 in to_run and len(to_run) == 22

    # Case 5: Exception classification safety ("meminfo"/"infiniband" MUST stay INFRA_ERROR)
    assert classify_cost_run_exception(RuntimeError("Could not read /proc/meminfo or infiniband")) == "INFRA_ERROR"
    assert classify_cost_run_exception(FloatingPointError("NaN detected")) == "ALGORITHM_FAILURE"
    assert classify_cost_run_exception(AssertionError("cost_accounting_residual failed")) == "ALGORITHM_FAILURE"
    assert classify_cost_run_exception(ValueError("Unexpected inf value")) == "ALGORITHM_FAILURE"


if __name__ == "__main__":
    test_1_cost_accounting_short_trajectories_and_metrics()
    print("[PASS] test_1_cost_accounting_short_trajectories_and_metrics")
    test_2_prng_isolation_global_state_and_b3_flaw_elimination()
    print("[PASS] test_2_prng_isolation_global_state_and_b3_flaw_elimination")
    test_3_cost_in_training_val_test_and_2ep_smoke_parity()
    print("[PASS] test_3_cost_in_training_val_test_and_2ep_smoke_parity")
    test_4_iut_stats_and_seed_offset_guards()
    print("[PASS] test_4_iut_stats_and_seed_offset_guards")
    print("ALL 4 COST CONFIRMATORY UNIT TEST SUITES PASSED!")
