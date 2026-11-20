"""
robust_r2_matched_confirmatory/conf_keys.py:
Cryptographically isolated 31-bit PRNG key plan for the R2_matched_128 vs A_ref_128 confirmatory study.

Guarantees:
- Imports ALL 2,167 historical keys:
  * 2,092 historical keys through Selection Diagnostics (incl. sealed HOLD_EVAL & unused [3301..3303])
  * 75 new development keys [3401, 3402, 3403] from `robust_r3_lr_stability/lr_keys.py`
- Derives brand-new isolated confirmatory seed keys under namespace `R2_MATCHED_CONF_V1`
  for reserved seed pool `[4001..4040]` plus an isolated technical smoke seed `3999` (never mixed with [4001..4040]).
- Strictly separates `init_policy`, `train`, `val`, and `conf_test` (`n_test_paths = 20,000`).
- NEVER opens, reads, or simulates `HOLD_EVAL` or any confirmatory test dataset.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = CONF_DIR.parent
LR_DIR = EXP_DIR / "robust_r3_lr_stability"
DIAG_DIR = EXP_DIR / "robust_r3_selection_diagnostics"
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"
for p in (str(LR_DIR), str(DIAG_DIR), str(BC_DIR), str(R3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from r3_keys import TRAINING_SCENARIOS, TEST_SCENARIOS, VOL_SEED_OFFSET
from lr_keys import LR_DEV_SEEDS, collect_all_2092_historical_keys, derive_lr_seed_bundle

NS = "R2_MATCHED_CONF_V1"
TECH_SMOKE_SEED = 3999
CONF_RESERVED_SEEDS = list(range(4001, 4041))  # Supports N up to 40 (e.g., 16 low-budget, 24 primary, or 30-36)
CONF_EVAL_SCENARIOS = tuple(TEST_SCENARIOS)   # ("fixed_015", "fixed_020", "fixed_025", "rs_015_025", "regime_switch_015_028", "fixed_030")


def sha256_seed31_conf(stage: str, seed_label: int, stream: str, purpose: str, scenario: str) -> int:
    payload = f"{NS}|{stage}|{int(seed_label)}|{stream}|{purpose}|{scenario}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    val = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return 1 if val == 0 else val


def collect_all_2167_historical_keys() -> Set[int]:
    """Collects exact 2,092 prior + 75 LR stability keys = 2,167 unique historical integers."""
    hist: Set[int] = set(collect_all_2092_historical_keys())
    for s in LR_DEV_SEEDS:
        hist.add(int(s))
        b = derive_lr_seed_bundle(s)
        hist.add(int(b["init_policy"]))
        hist.add(int(b["dev_eval_shared_dW"]))
        hist.add(int(b["diag1_jk_perm"]))
        hist.add(int(b["diag2_jk_perm"]))
        for sc, v in b["train_dW"].items():
            hist.add(int(v))
        for sc, v in b["train_perm"].items():
            hist.add(int(v))
        for sc, v in b["val_dW"].items():
            hist.add(int(v))
        for sc, v in b["diag1_20k_dW"].items():
            hist.add(int(v))
        for sc, v in b["diag2_20k_dW"].items():
            hist.add(int(v))
    return hist


def derive_conf_seed_bundle(seed_label: int) -> Dict[str, Any]:
    st = "R2_CONF_TECH_SMOKE" if int(seed_label) == TECH_SMOKE_SEED else "R2_CONF_RUN"
    train_dW = {sc: sha256_seed31_conf(st, seed_label, "train", "dW", sc) for sc in TRAINING_SCENARIOS}
    train_perm = {sc: sha256_seed31_conf(st, seed_label, "train", "perm", sc) for sc in TRAINING_SCENARIOS}
    for sub in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        train_dW[sub] = sha256_seed31_conf(st, seed_label, "train", "dW", sub)
        train_perm[sub] = sha256_seed31_conf(st, seed_label, "train", "perm", sub)

    val_dW = {sc: sha256_seed31_conf(st, seed_label, "val", "dW", sc) for sc in TRAINING_SCENARIOS}
    conf_test_shared_dW = sha256_seed31_conf(st, seed_label, "conf_test", "dW", "shared_pair_6scen_20k")

    return {
        "seed_label": int(seed_label),
        "role": "tech_smoke_only" if int(seed_label) == TECH_SMOKE_SEED else "reserved_confirmatory_pool",
        "init_policy": sha256_seed31_conf(st, seed_label, "init", "policy", "shared_pair"),
        "train_dW": train_dW,
        "train_perm": train_perm,
        "val_dW": val_dW,
        "conf_test_shared_dW": conf_test_shared_dW,
    }


def verify_conf_key_isolation() -> Dict[str, Any]:
    hist = collect_all_2167_historical_keys()
    all_target_seeds = [TECH_SMOKE_SEED] + CONF_RESERVED_SEEDS
    bundles = {s: derive_conf_seed_bundle(s) for s in all_target_seeds}

    new_keys: List[int] = []
    overflow_errors: List[str] = []
    for s, b in bundles.items():
        new_keys.append(int(s))
        flat = [int(b["init_policy"]), int(b["conf_test_shared_dW"])]
        flat.extend(int(v) for v in b["train_dW"].values())
        flat.extend(int(v) for v in b["train_perm"].values())
        flat.extend(int(v) for v in b["val_dW"].values())
        for k in flat:
            if k <= 0 or (k + VOL_SEED_OFFSET) >= 2**32:
                overflow_errors.append(f"seed={s}, key={k}")
        new_keys.extend(flat)

    new_set = set(new_keys)
    internal_collisions = len(new_keys) - len(new_set)
    hist_collisions = sorted(list(new_set.intersection(hist)))
    first_b = bundles[CONF_RESERVED_SEEDS[0]]
    k_per_seed = 1 + 2 + len(first_b["train_dW"]) + len(first_b["train_perm"]) + len(first_b["val_dW"])
    return {
        "namespace": NS,
        "isolation_passed": (len(overflow_errors) == 0 and internal_collisions == 0 and len(hist_collisions) == 0),
        "historical_keys_checked": len(hist),
        "new_keys_checked": len(new_keys),
        "reserved_conf_seeds": CONF_RESERVED_SEEDS,
        "tech_smoke_seed": TECH_SMOKE_SEED,
        "keys_per_seed": k_per_seed,
        "overflow_errors": overflow_errors,
        "internal_collision_count": internal_collisions,
        "historical_collisions": hist_collisions,
        "hold_eval_status": "SEALED_AND_NEVER_OPENED",
        "bundles": bundles,
    }


def write_data_and_seed_plan() -> Path:
    audit = verify_conf_key_isolation()
    assert audit["isolation_passed"], f"PRNG isolation failed: {audit}"
    # Recompute exact keys_per_seed dynamically rather than hardcoding
    first_b = audit["bundles"][CONF_RESERVED_SEEDS[0]]
    k_per_seed = 1 + 2 + len(first_b["train_dW"]) + len(first_b["train_perm"]) + len(first_b["val_dW"])
    audit["keys_per_seed"] = k_per_seed

    plan = {
        "status": "LOCKED_N24_SEEDS_4001_4024_PRE_EXECUTION",
        "namespace": NS,
        "hold_eval_status": "SEALED_AND_NEVER_OPENED",
        "conf_test_status": "KEYS_DERIVED_ONLY_NOT_SIMULATED_OR_OPENED",
        "eval_scenarios_locked": list(CONF_EVAL_SCENARIOS),
        "primary_condition_A": {"scenario": "fixed_030", "metric": "he_net_ddof0", "hypothesis": "H0: E[DeltaHE] >= 0 vs H1: E[DeltaHE] < 0"},
        "primary_condition_B": {"scenario": "fixed_020", "metric": "hedged_oce_y_nom", "eps": 0.00010, "hypothesis": "H0: E[DeltaOCE] >= 0.00010 vs H1: E[DeltaOCE] < 0.00010"},
        "locked_primary_N": 24,
        "locked_seeds": list(range(4001, 4025)),
        "unused_pool_reserved_untouched": list(range(4025, 4041)),
        "tech_smoke_seed_excluded_from_analysis": TECH_SMOKE_SEED,
        "data_dimensions": {
            "A_ref_128_train": "4 x 5,000 distinct fixed_020 paths (32/sub x 4 = 128/step, 156 steps/ep, 200 ep = 31,200 steps, 3,993,600 path-rollouts)",
            "R2_matched_128_train": "4 x 5,000 paths across ('fixed_015','fixed_020','fixed_025','rs_015_025') (32/scen x 4 = 128/step, 31,200 steps, 3,993,600 path-rollouts)",
            "A_ref_128_val_selection": "argmin over 200 epochs on 1 x 5,000 fixed_020 paths (oce_idx=0)",
            "R2_matched_128_val_selection": "argmin over 200 epochs of uniform mean across 4 x 5,000 training-scenario Val sets (oce_idx=0..3)",
            "conf_test_shared_pair": "20,000 shared-dW paths per scenario across 6 scenarios (created ONLY after selection closes; identical dW across scenarios introduces cross-scenario dependence noted in plan)",
            "normalize_batch_mean": True,
        },
        "prng_isolation_audit": {k: v for k, v in audit.items() if k != "bundles"},
        "seed_bundles": {str(s): b for s, b in audit["bundles"].items()},
    }
    out = CONF_DIR / "DATA_AND_SEED_PLAN.json"
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = write_data_and_seed_plan()
    d = json.loads(p.read_text(encoding="utf-8"))
    print("PRNG AUDIT:", json.dumps(d["prng_isolation_audit"], indent=2))
    print("Saved key plan to", p)
