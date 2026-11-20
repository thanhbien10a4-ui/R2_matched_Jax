"""
conf_design.py: Locked specifications for the independent R2_matched confirmatory experiment.
Reuses canonical constants/configs from upstream r3_train, lr_schedule, and bc_models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = CONF_DIR.parent
LR_DIR = EXP_DIR / "robust_r3_lr_stability"
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"
P2_DIR = EXP_DIR / "scripts" / "robust_deep_hedging_phase2"

for _p in (str(CONF_DIR), str(LR_DIR), str(BC_DIR), str(R3_DIR), str(EXP_DIR), str(P2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from r3_keys import TRAINING_SCENARIOS, TEST_SCENARIOS
from r3_train import R3TrainConfig
from lr_schedule import STEPS_PER_EPOCH, LR_MAX, LR_MIN, T_CONST, T_TOTAL, compute_lr_float
from bc_models import make_canonical_configs
from run_r3_pilot import SCENARIO_SPECS

# Locked design specification constants
PRIMARY_BRANCHES: Tuple[str, str] = ("A_DECAY", "R2_DECAY")
BRANCH_NAMES: Dict[str, str] = {
    "A_DECAY": "A_ref_128",
    "R2_DECAY": "R2_matched_128",
}

TRAIN_PATHS_PER_SCENARIO: int = 5000
VAL_PATHS_PER_SCENARIO: int = 5000
CONF_TEST_PATHS_PER_SCENARIO: int = 20000

EPOCHS_LOCKED: int = 200
BATCH_SIZE_LOCKED: int = 32
PATHS_PER_STEP_A: int = 128          # 4 x 32 nominal fixed_020 sub-batches
PATHS_PER_STEP_R2: int = 128         # 4 x 32 across 4 training scenarios
STEPS_PER_EPOCH_LOCKED: int = 156    # floor(20000 / 128) = 156 (matches STEPS_PER_EPOCH)
TOTAL_UPDATE_STEPS_LOCKED: int = EPOCHS_LOCKED * STEPS_PER_EPOCH_LOCKED  # 31,200
TOTAL_PROCESSED_PATHS_LOCKED: int = TOTAL_UPDATE_STEPS_LOCKED * 128      # 3,993,600

# Statistical rules (Berger IUT)
ALPHA_ONE_SIDED: float = 0.025
EPS_NONINFERIORITY_OCE020: float = 0.00010
PRIMARY_N_SEEDS: int = 24
FALLBACK_N_SEEDS: int = 16

PRIMARY_CKPT_KIND: str = "best_val"
SECONDARY_CKPT_KIND: str = "ep200"

PRIMARY_SCENARIOS: Tuple[str, str] = ("fixed_030", "fixed_020")
SECONDARY_SCENARIOS: Tuple[str, ...] = tuple(
    sc for sc in TEST_SCENARIOS if sc not in PRIMARY_SCENARIOS
)


def get_locked_train_config() -> R3TrainConfig:
    """Return the locked training configuration identical to LR_STABILITY DECAY runs."""
    cfg = R3TrainConfig(
        epochs=EPOCHS_LOCKED,
        batch_size=BATCH_SIZE_LOCKED,
        learning_rate=float(LR_MAX),
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )
    assert cfg.epochs == EPOCHS_LOCKED
    assert cfg.batch_size == BATCH_SIZE_LOCKED
    assert abs(cfg.learning_rate - 1e-3) < 1e-12
    assert abs(cfg.epsilon - 1e-7) < 1e-15
    assert STEPS_PER_EPOCH == STEPS_PER_EPOCH_LOCKED
    assert T_CONST == 100 * STEPS_PER_EPOCH_LOCKED
    assert T_TOTAL == TOTAL_UPDATE_STEPS_LOCKED
    assert abs(LR_MIN - 1e-4) < 1e-12
    return cfg


def get_locked_branch_configs() -> Dict[str, Any]:
    """Return canonical agent_cfg, gym_cfg, feature_dims and k_oce map for A_DECAY & R2_DECAY."""
    agent_cfg, gym_cfg, feature_dims = make_canonical_configs()
    assert tuple(agent_cfg.features) == ("price", "delta", "time_left")
    assert int(agent_cfg.width) == 40 and int(agent_cfg.depth) == 5
    assert str(agent_cfg.activation) == "softplus"
    assert bool(agent_cfg.init_delta) is True
    assert str(gym_cfg.utility) == "exp2" and float(gym_cfg.lmbda) == 10.0
    return {
        "agent_cfg": agent_cfg,
        "gym_cfg": gym_cfg,
        "feature_dims": feature_dims,
        "k_oce": {"A_ref_128": 1, "R2_matched_128": 4},
        "cost_s": 0.0,
        "normalize_batch_mean": True,
        "steps": 20,
        "scenario_specs": {sc: SCENARIO_SPECS[sc] for sc in TEST_SCENARIOS},
    }


def summarize_locked_spec() -> Dict[str, Any]:
    """Structured design summary for manifests and dry-run checks."""
    tcfg = get_locked_train_config()
    cfgs = get_locked_branch_configs()
    return {
        "branches": PRIMARY_BRANCHES,
        "branch_to_canonical": BRANCH_NAMES,
        "training_scenarios": list(TRAINING_SCENARIOS),
        "conf_eval_scenarios": list(TEST_SCENARIOS),
        "primary_scenarios": list(PRIMARY_SCENARIOS),
        "secondary_scenarios": list(SECONDARY_SCENARIOS),
        "train_paths_per_scenario": TRAIN_PATHS_PER_SCENARIO,
        "val_paths_per_scenario": VAL_PATHS_PER_SCENARIO,
        "conf_test_paths_per_scenario": CONF_TEST_PATHS_PER_SCENARIO,
        "budget": {
            "epochs": tcfg.epochs,
            "steps_per_epoch": STEPS_PER_EPOCH_LOCKED,
            "total_update_steps": TOTAL_UPDATE_STEPS_LOCKED,
            "paths_per_step": PATHS_PER_STEP_A,
            "total_processed_paths": TOTAL_PROCESSED_PATHS_LOCKED,
        },
        "optimizer": {
            "name": "Adam",
            "beta_1": tcfg.beta_1,
            "beta_2": tcfg.beta_2,
            "epsilon": tcfg.epsilon,
            "schedule": "constant_then_cosine_decay",
            "lr_max": float(LR_MAX),
            "lr_min": float(LR_MIN),
            "t_const_steps": T_CONST,
            "t_total_steps": T_TOTAL,
            "lr_step_0": compute_lr_float(0, "DECAY"),
            "lr_step_15600": compute_lr_float(15600, "DECAY"),
            "lr_step_31200": compute_lr_float(31200, "DECAY"),
        },
        "environment": {
            "cost_s_A": float(cfgs["cost_s"]),
            "cost_s_R2": float(cfgs["cost_s"]),
            "steps": int(cfgs["steps"]),
            "normalize_batch_mean": bool(cfgs["normalize_batch_mean"]),
        },
        "statistical_protocol": {
            "design": "paired_by_seed_label",
            "alpha_one_sided": ALPHA_ONE_SIDED,
            "ci_two_sided_level": 1.0 - 2.0 * ALPHA_ONE_SIDED,
            "eps_noninferiority_oce020": EPS_NONINFERIORITY_OCE020,
            "primary_n_seeds": PRIMARY_N_SEEDS,
            "fallback_n_seeds": FALLBACK_N_SEEDS,
            "joint_rule": "Berger_IUT (reject H0a at alpha=0.025 AND reject H0b at alpha=0.025)",
        },
    }
