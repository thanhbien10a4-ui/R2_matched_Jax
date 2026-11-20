"""
tests/test_conf_suite.py: Comprehensive local & server verification suite for robust_r2_matched_confirmatory.
Tests ONLY PRNG isolation, locked config/budget/schedule parity, metric definitions on synthetic arrays,
statistical analyzer correctness on synthetic datasets (Berger IUT + ledger), and safety guards.
NEVER simulates confirmatory [4001..4040] datasets or opens HOLD_EVAL.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

TEST_DIR = Path(__file__).resolve().parent
CONF_DIR = TEST_DIR.parent
if str(CONF_DIR) not in sys.path:
    sys.path.insert(0, str(CONF_DIR))

from conf_keys import (
    TECH_SMOKE_SEED,
    CONF_RESERVED_SEEDS,
    verify_conf_key_isolation,
)
from conf_design import (
    get_locked_branch_configs,
    get_locked_train_config,
    summarize_locked_spec,
)
from conf_stats import (
    exact_binom_lower_p,
    wilcoxon_lower_tail_p,
    paired_t_summary,
    analyze_confirmatory_pairs,
)
import conf_run
from r3_metrics import evaluate_rollout_metrics_r3
from bc_models import init_full_params, count_param_breakdown, aref_128_step_loss, r2_matched_step_loss
from lr_schedule import compute_lr_float
import jax
import jax.numpy as jnp


def test_prng_key_isolation_and_no_test_data() -> None:
    iso = verify_conf_key_isolation()
    assert iso["isolation_passed"] is True
    assert iso["historical_keys_checked"] == 2167
    assert iso["keys_per_seed"] == 21
    assert iso["new_keys_checked"] == 861  # 41 seeds * 21 keys/seed
    assert iso["internal_collision_count"] == 0
    assert iso["historical_collisions"] == []
    assert iso["overflow_errors"] == []
    assert iso["hold_eval_status"] == "SEALED_AND_NEVER_OPENED"
    assert CONF_RESERVED_SEEDS == list(range(4001, 4041))
    assert TECH_SMOKE_SEED == 3999

    # Ensure zero confirmatory results/checkpoints were pre-generated
    for dname in ("results", "checkpoints"):
        ddir = CONF_DIR / dname
        if ddir.exists():
            for p in ddir.iterdir():
                assert not p.name.startswith("confirmatory_"), f"Unexpected confirmatory output found: {p}"


def test_locked_design_budget_and_schedule() -> None:
    spec = summarize_locked_spec()
    tcfg = get_locked_train_config()
    cfgs = get_locked_branch_configs()

    assert spec["budget"]["epochs"] == 200
    assert spec["budget"]["steps_per_epoch"] == 156
    assert spec["budget"]["total_update_steps"] == 31200
    assert spec["budget"]["paths_per_step"] == 128
    assert spec["budget"]["total_processed_paths"] == 3993600

    assert abs(tcfg.epsilon - 1e-7) < 1e-15
    assert float(cfgs["cost_s"]) == 0.0

    # Cosine LR schedule exact landmarks
    assert abs(compute_lr_float(0, "DECAY") - 1e-3) < 1e-12
    assert abs(compute_lr_float(15600, "DECAY") - 1e-3) < 1e-12
    assert abs(compute_lr_float(23400, "DECAY") - 5.5e-4) < 1e-12
    assert abs(compute_lr_float(31200, "DECAY") - 1e-4) < 1e-12

    # Shared policy initialization parity on synthetic dummy PRNGKey(0)
    k = jax.random.PRNGKey(0)
    pa = init_full_params(k, cfgs["gym_cfg"], cfgs["feature_dims"], k_oce=1)
    pr = init_full_params(k, cfgs["gym_cfg"], cfgs["feature_dims"], k_oce=4)
    ca = count_param_breakdown(pa)
    cr = count_param_breakdown(pr)
    assert ca["policy_total_params"] == cr["policy_total_params"] == 6762
    assert ca["oce_y_params"] == 1 and ca["oce_total_params"] == 2 and ca["grand_total_params"] == 6764
    assert cr["oce_y_params"] == 4 and cr["oce_total_params"] == 8 and cr["grand_total_params"] == 6770
    diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree_util.tree_leaves(pa["policy"]), jax.tree_util.tree_leaves(pr["policy"]))
    )
    assert diff == 0.0


def test_metric_exact_definitions_synthetic() -> None:
    # Construct synthetic gains with non-zero mean so population std != RMSE vs 0
    rng = np.random.default_rng(12345)
    n_paths = 500
    gains = -0.05 + 0.03 * rng.standard_normal(n_paths)
    payoff = gains.copy()
    pnl = np.zeros_like(gains)
    cost = np.zeros_like(gains)
    # Create dummy actions & deltas (n_paths, 20, 1)
    actions = 0.01 * rng.standard_normal((n_paths, 20, 1))
    deltas = np.cumsum(actions, axis=1)

    roll_syn = {
        "gains": jnp.asarray(gains),
        "payoff": jnp.asarray(payoff),
        "pnl": jnp.asarray(pnl),
        "cost": jnp.asarray(cost),
        "hedged_loss": jnp.asarray(-gains + 0.01),
        "unhedged_loss": jnp.asarray(-payoff + 0.02),
        "loss": jnp.asarray(-gains + 0.01),
        "actions": jnp.asarray(actions),
        "deltas": jnp.asarray(deltas),
    }
    m = evaluate_rollout_metrics_r3(roll_syn, eps=0.00010)
    expected_he_ddof0 = float(np.std(np.asarray(roll_syn["gains"], dtype=np.float64), ddof=0))
    expected_rmse_zero = float(np.sqrt(np.mean(gains ** 2)))
    assert abs(m["he_net_ddof0"] - expected_he_ddof0) < 1e-12
    assert abs(m["he_net_ddof0"] - expected_rmse_zero) > 0.01


def _make_synthetic_paired_rows(seeds, d_he_030, d_oce_020):
    rows = []
    for i, s in enumerate(seeds):
        for sc in ("fixed_015", "fixed_020", "fixed_025", "rs_015_025", "regime_switch_015_028", "fixed_030"):
            he_val = float(d_he_030[i]) if sc == "fixed_030" else -0.0005
            oce_val = float(d_oce_020[i]) if sc == "fixed_020" else 0.0
            a_he = 0.040 if sc == "fixed_030" else 0.016
            a_oce = 0.0512
            rows.append({
                "seed": s,
                "ckpt_kind": "best_val",
                "scenario": sc,
                "A_he_net_ddof0": a_he,
                "R2_he_net_ddof0": a_he + he_val,
                "delta_he_R2_minus_A": he_val,
                "A_hedged_oce_y_nom": a_oce,
                "R2_hedged_oce_y_nom": a_oce + oce_val,
                "delta_oce_R2_minus_A": oce_val,
                "delta_cvar95_R2_minus_A": -0.001,
                "A_cvar95": 0.080,
                "R2_cvar95": 0.079,
                "A_turnover": 1.80,
                "R2_turnover": 1.83,
                "delta_turnover_R2_minus_A": 0.03,
            })
    return rows


def test_synthetic_confirmatory_iut_pass_fail_and_ledger() -> None:
    seeds = list(range(4001, 4025))
    rng = np.random.default_rng(777)
    noise_a = rng.normal(0.0, 0.0008, len(seeds))
    noise_b = rng.normal(0.0, 0.00004, len(seeds))

    # Case A: Clear PASS on both H1a and H1b
    d_he_pass = -0.0045 + noise_a - np.mean(noise_a)
    d_oce_pass = 0.00000 + noise_b - np.mean(noise_b)
    rows_a = _make_synthetic_paired_rows(seeds, d_he_pass, d_oce_pass)
    res_a = analyze_confirmatory_pairs(rows_a, planned_seeds=seeds, eps_margin=0.00010, alpha_one_sided=0.025)
    assert res_a["verdict"] == "PASS_CONFIRMATORY_IUT"
    assert res_a["confirmatory_pass"] is True
    assert res_a["complete_and_clean_gate"] is True
    assert res_a["co_primary"]["H1a_HE_030_best_val"]["ci95_two_sided_high"] < 0.0
    assert res_a["co_primary"]["H1b_OCE_020_best_val"]["ci95_two_sided_high"] < 0.00010
    assert res_a["co_primary"]["H1a_HE_030_best_val"]["loo_decision_flips"] == 0
    assert res_a["co_primary"]["H1b_OCE_020_best_val"]["loo_decision_flips"] == 0
    # Verify exact Wilcoxon enumeration is used for N=24 (m_nonzero=24 <= 50)
    assert res_a["co_primary"]["H1a_HE_030_best_val"]["supportive_wilcoxon"]["method"] == "exact_enumeration"
    # Verify absolute branch metrics appear in secondary_summaries
    sec_metrics = {item["metric"] for item in res_a["secondary_summaries"]}
    assert "A_he_net_ddof0" in sec_metrics and "R2_he_net_ddof0" in sec_metrics
    assert "A_hedged_oce_y_nom" in sec_metrics and "R2_hedged_oce_y_nom" in sec_metrics

    # Case B: H1a passes, H1b fails (upper 95% CI exceeds +0.00010)
    d_oce_fail = 0.000095 + noise_b - np.mean(noise_b)
    rows_b = _make_synthetic_paired_rows(seeds, d_he_pass, d_oce_fail)
    res_b = analyze_confirmatory_pairs(rows_b, planned_seeds=seeds, eps_margin=0.00010, alpha_one_sided=0.025)
    assert res_b["co_primary"]["H1a_HE_030_best_val"]["pass_bound"] is True
    assert res_b["co_primary"]["H1b_OCE_020_best_val"]["pass_bound"] is False
    assert res_b["verdict"] == "FAIL_CONFIRMATORY_IUT"
    assert res_b["confirmatory_pass"] is False

    # Case C: H1b passes, H1a fails (HE mean near 0 crossing bound)
    d_he_fail = 0.0002 + noise_a - np.mean(noise_a)
    rows_c = _make_synthetic_paired_rows(seeds, d_he_fail, d_oce_pass)
    res_c = analyze_confirmatory_pairs(rows_c, planned_seeds=seeds, eps_margin=0.00010, alpha_one_sided=0.025)
    assert res_c["co_primary"]["H1a_HE_030_best_val"]["pass_bound"] is False
    assert res_c["co_primary"]["H1b_OCE_020_best_val"]["pass_bound"] is True
    assert res_c["verdict"] == "FAIL_CONFIRMATORY_IUT"

    # Case D: Completeness & NaN ledger gate (1 NaN + 1 missing seed)
    rows_d = [r.copy() for r in rows_a if r["seed"] != 4024]
    for r in rows_d:
        if r["seed"] == 4023 and r["scenario"] == "fixed_020":
            r["delta_oce_R2_minus_A"] = float("nan")
    res_d = analyze_confirmatory_pairs(rows_d, planned_seeds=seeds, eps_margin=0.00010, alpha_one_sided=0.025)
    assert res_d["complete_and_clean_gate"] is False
    assert res_d["confirmatory_pass"] is False
    assert res_d["verdict"] == "INCOMPLETE_OR_FAILED_AUDIT"
    assert res_d["ledger"]["H1a_HE_030"]["missing_seeds"] == [4024]
    assert res_d["ledger"]["H1b_OCE_020"]["missing_seeds"] == [4024]
    assert res_d["ledger"]["H1b_OCE_020"]["non_finite_seeds"] == [4023]


def test_nonparametric_exact_sign_and_wilcoxon() -> None:
    # Exact binomial: 8/8 below bound under p0=0.5 -> (1/2)^8 = 1/256 = 0.00390625
    p_sign = exact_binom_lower_p(8, 8, 0.5)
    assert abs(p_sign - (1.0 / 256.0)) < 1e-12

    # Exact Wilcoxon signed-rank with all 8 negative values: W+ = 0 -> p = 1/256
    w = wilcoxon_lower_tail_p([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])
    assert w["method"] == "exact_enumeration"
    assert w["m_nonzero"] == 8
    assert w["w_plus"] == 0.0
    assert abs(w["p_one_sided"] - (1.0 / 256.0)) < 1e-12

    # Exact Wilcoxon at N=24 (no ties) uses exact DP and gives 1 / 2^24
    w24 = wilcoxon_lower_tail_p([float(-i - 1) for i in range(24)])
    assert w24["method"] == "exact_enumeration"
    assert abs(w24["p_one_sided"] - (1.0 / float(1 << 24))) < 1e-12

    # Two-sided 95% CI alignment check: p_one_sided < 0.025 iff ci95_two_sided_high < bound
    sample = [-0.001, -0.002, -0.0015, -0.0005, -0.0012, -0.0018, -0.0009, -0.0011]
    st = paired_t_summary(sample, bound=0.0, alpha_one_sided=0.025)
    assert (st["p_one_sided"] < 0.025) == st["pass_bound"]
    assert abs(st["ci95_two_sided_high"] - st["one_sided_ucl"]) < 1e-15


def test_dry_run_and_confirm_guard() -> None:
    # Refuses confirmatory run without --confirm-locked-run
    raised = False
    try:
        conf_run.run_experiment(mode="execute_confirmatory", n_seeds=24, confirm_locked_run=False)
    except RuntimeError:
        raised = True
    assert raised, "execute_confirmatory must refuse without confirm_locked_run=True!"

    rep = conf_run.run_dry_run_check()
    assert rep["test_sets_4001_4040_status"] == "NOT_GENERATED_AND_NEVER_OPENED"
    assert rep["hold_eval_status"] == "SEALED_AND_NEVER_OPENED"
    assert rep["isolation"]["isolation_passed"] is True
    assert rep["isolation"]["keys_per_seed"] == 21
    assert "robust_r2_matched_confirmatory/R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md" in rep["source_sha256"]
    assert "robust_r2_matched_confirmatory/SAMPLE_SIZE_SENSITIVITY.csv" in rep["source_sha256"]
    assert "robust_r3_lr_stability/lr_train.py" in rep["source_sha256"]
    assert "robust_r3_dev/r3_metrics.py" in rep["source_sha256"]


def test_seed_offset_guard_and_failure_classification() -> None:
    # 1. Refuses any slice spilling past locked N=24 cohort [4001..4024] into [4025..4040]
    spilled = False
    try:
        conf_run.resolve_locked_confirmatory_seeds(n_seeds=5, seed_offset=20, locked_n_total=24)
    except ValueError:
        spilled = True
    assert spilled, "--seed-offset must refuse exceeding locked [4001..4024]!"

    # 2. Valid slice returns exact missing seeds in [4001..4024]
    s_ok, _ = conf_run.resolve_locked_confirmatory_seeds(n_seeds=6, seed_offset=18, locked_n_total=24)
    assert s_ok == [4019, 4020, 4021, 4022, 4023, 4024]

    # 3. Distinguishes ALGORITHM_FAILURE vs INFRA_ERROR & never discards/replaces adverse seed
    assert conf_run.classify_run_exception(RuntimeError("Non-finite weights detected in ep 42")) == "ALGORITHM_FAILURE"
    assert conf_run.classify_run_exception(OSError("Temporary cluster filesystem timeout")) == "INFRA_ERROR"

    existing = [
        {"seed": 4001, "status": "COMPLETED"},
        {"seed": 4002, "status": "COMPLETED"},
        {"seed": 4003, "status": "FAILED", "error_category": "ALGORITHM_FAILURE"},
        {"seed": 4004, "status": "FAILED", "error_category": "INFRA_ERROR"},
    ]
    to_run, info = conf_run.resolve_locked_confirmatory_seeds(
        n_seeds=24, seed_offset=0, locked_n_total=24, existing_seed_records=existing
    )
    assert info["skipped_completed"] == [4001, 4002]
    assert info["preserved_algo_failures"] == [4003]
    assert info["retried_infra"] == [4004]
    assert to_run == list(range(4004, 4025))
    assert 4003 not in to_run and 4025 not in to_run


def test_step_loss_and_policy_grad_parity_synthetic() -> None:
    """When all 4 scenario sub-batches & oce vectors are identical, A_ref_128 and R2_matched_128 have identical loss & policy grad."""
    cfgs = get_locked_branch_configs()
    k = jax.random.PRNGKey(0)
    pa = init_full_params(k, cfgs["gym_cfg"], cfgs["feature_dims"], k_oce=1)
    pr = init_full_params(k, cfgs["gym_cfg"], cfgs["feature_dims"], k_oce=4)

    # Synthetic mini world (32 paths, 20 steps) using dummy seed=0 (zero test data touched)
    b = conf_run.build_scenario_world("fixed_020", 0, samples=32)
    sub_batches = [b, b, b, b]

    la, ga = jax.value_and_grad(lambda p: aref_128_step_loss(p, cfgs["gym_cfg"], sub_batches))(pa)
    lr_, gr = jax.value_and_grad(lambda p: r2_matched_step_loss(p, cfgs["gym_cfg"], sub_batches))(pr)
    assert abs(float(la) - float(lr_)) < 1e-6
    pol_grad_diff = max(
        float(jnp.max(jnp.abs(g1 - g2)))
        for g1, g2 in zip(jax.tree_util.tree_leaves(ga["policy"]), jax.tree_util.tree_leaves(gr["policy"]))
    )
    assert pol_grad_diff < 1e-6


def main() -> None:
    tests = [
        test_prng_key_isolation_and_no_test_data,
        test_locked_design_budget_and_schedule,
        test_metric_exact_definitions_synthetic,
        test_synthetic_confirmatory_iut_pass_fail_and_ledger,
        test_nonparametric_exact_sign_and_wilcoxon,
        test_dry_run_and_confirm_guard,
        test_seed_offset_guard_and_failure_classification,
        test_step_loss_and_policy_grad_parity_synthetic,
    ]
    for fn in tests:
        fn()
        print(f"PASS: {fn.__name__}", flush=True)
    print(f"ALL {len(tests)}/{len(tests)} CONFIRMATORY UNIT TESTS PASSED", flush=True)


if __name__ == "__main__":
    main()
