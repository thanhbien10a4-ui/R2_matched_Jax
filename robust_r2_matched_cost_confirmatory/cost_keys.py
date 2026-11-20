"""
robust_r2_matched_cost_confirmatory/cost_keys.py:
Cryptographically isolated 31-bit PRNG key plan for the R2_matched_128 vs A_ref_128
confirmatory study at proportional transaction cost c_s = 0.0010 (10 bps).

Guarantees & Scope:
- Imports ALL historical keys across the repository:
  * All 2,167 prior R3 / BC / DIAG / LR keys (including HOLD_EVAL & PHASE2_CANDIDATE_SEEDS)
  * All 492 prior R2_MATCHED_CONF_V1 keys from robust_r2_matched_confirmatory (3999 & [4001..4040])
  * Additional B3 cost & rough-memory integer seeds/offsets ([701..710], [801..810], [1700..1710], etc.)
- Derives fresh 31-bit keys under distinct namespace `R2_COST10_CONF_V1` for:
  * Isolated technical smoke seed `5999` (excluded from confirmatory cohort)
  * Reserved confirmatory pool `[5001..5040]` where `[5001..5024]` (N=24) is locked
    and `[5025..5040]` stays unused/sealed outside the locked experiment.
- Explicitly eliminates the B3 Val(s)/Train(s+1) shared-noise defect: every
  (seed_label, purpose, scenario, component) stream receives a distinct SHA-256 31-bit key
  and preserves zero cross-seed or train/val/test key sharing.
- Note: Hash disjointness guarantees zero integer seed collisions across all tracked pools;
  statistical stream near-orthogonality relies on MT19937/Philox seeding properties and is
  also checked empirically via pairwise dW correlation tests.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

COST_CONF_DIR = Path(__file__).resolve().parent
EXP_DIR = COST_CONF_DIR.parent
R2_CONF_DIR = EXP_DIR / "robust_r2_matched_confirmatory"
LR_DIR = EXP_DIR / "robust_r3_lr_stability"
DIAG_DIR = EXP_DIR / "robust_r3_selection_diagnostics"
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"

for _p in (str(R2_CONF_DIR), str(LR_DIR), str(DIAG_DIR), str(BC_DIR), str(R3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from r3_keys import TRAINING_SCENARIOS, TEST_SCENARIOS, VOL_SEED_OFFSET
from conf_keys import (
    TECH_SMOKE_SEED as R2_CONF_SMOKE_SEED,
    CONF_RESERVED_SEEDS as R2_CONF_RESERVED_SEEDS,
    collect_all_2167_historical_keys,
    derive_conf_seed_bundle,
)

NS = "R2_COST10_CONF_V1"
COST_SMOKE_SEED = 5999
COST_RESERVED_SEEDS = list(range(5001, 5041))
COST_LOCKED_COHORT_SEEDS = list(range(5001, 5025))  # Locked N=24
COST_SEALED_UNUSED_SEEDS = list(range(5025, 5041))  # Strictly untouched
COST_EVAL_SCENARIOS = tuple(TEST_SCENARIOS)


def sha256_seed31_cost(
    stage: str,
    seed_label: int,
    purpose: str,
    scenario: str,
    component: str,
) -> int:
    payload = f"{NS}|{stage}|{int(seed_label)}|{purpose}|{scenario}|{component}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    val = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return 1 if val == 0 else val


def collect_all_historical_keys() -> Set[int]:
    """
    Collects ALL historical RNG integers and seed labels across the entire repo:
    - 2,167 keys through LR Stability (incl. sealed HOLD_EVAL & PHASE2_CANDIDATE_SEEDS)
    - All 41 seeds x 12 ints from robust_r2_matched_confirmatory (3999 & [4001..4040])
    - Any extra B3 / rough-memory seed & test offset integers ([1700..1710], etc.)
    """
    hist: Set[int] = set(collect_all_2167_historical_keys())

    # All R2_MATCHED_CONF_V1 (c_s=0.0) confirmatory & reserved seeds
    for s in [R2_CONF_SMOKE_SEED] + list(R2_CONF_RESERVED_SEEDS):
        hist.add(int(s))
        b = derive_conf_seed_bundle(s)
        hist.add(int(b["init_policy"]))
        hist.add(int(b["conf_test_shared_dW"]))
        for v in b["train_dW"].values():
            hist.add(int(v))
        for v in b["train_perm"].values():
            hist.add(int(v))
        for v in b["val_dW"].values():
            hist.add(int(v))

    # Explicit extra B3 / rough / legacy script integers just in case
    for b3_test in range(1700, 1720):
        hist.add(b3_test)
        hist.add(b3_test + VOL_SEED_OFFSET)
    for base in (42, 43, 44, 45, 46, 500, 501, 502, 503, 504, 505, 601, 602, 603):
        hist.add(base)
        hist.add(base + 1)
        hist.add(base + 999)
        hist.add(base + VOL_SEED_OFFSET)
    for b3s in range(701, 711):
        for off in (0, 1, 2, 10, 999, 1000, 10000, 50000, 99999, VOL_SEED_OFFSET):
            hist.add(b3s + off)
    return hist


def derive_cost_seed_bundle(seed_label: int) -> Dict[str, Any]:
    s_int = int(seed_label)
    stage = "R2_COST10_TECH_SMOKE" if s_int == COST_SMOKE_SEED else "R2_COST10_CONF_RUN"
    if s_int == COST_SMOKE_SEED:
        role = "tech_smoke_only"
    elif s_int in COST_LOCKED_COHORT_SEEDS:
        role = "locked_confirmatory_n24"
    elif s_int in COST_SEALED_UNUSED_SEEDS:
        role = "reserved_unused_quarantined"
    else:
        role = "custom_out_of_pool"

    train_dW = {sc: sha256_seed31_cost(stage, s_int, "train", sc, "dW") for sc in TRAINING_SCENARIOS}
    train_perm = {sc: sha256_seed31_cost(stage, s_int, "train", sc, "perm") for sc in TRAINING_SCENARIOS}
    for sub in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        train_dW[sub] = sha256_seed31_cost(stage, s_int, "train", sub, "dW")
        train_perm[sub] = sha256_seed31_cost(stage, s_int, "train", sub, "perm")

    val_dW = {sc: sha256_seed31_cost(stage, s_int, "val", sc, "dW") for sc in TRAINING_SCENARIOS}
    conf_test_shared_dW = sha256_seed31_cost(stage, s_int, "conf_test", "shared_pair_6scen_20k", "dW")
    init_policy = sha256_seed31_cost(stage, s_int, "init", "shared_pair", "policy")

    return {
        "seed_label": s_int,
        "stage": stage,
        "role": role,
        "init_policy": init_policy,
        "train_dW": train_dW,
        "train_perm": train_perm,
        "val_dW": val_dW,
        "conf_test_shared_dW": conf_test_shared_dW,
    }


def _flatten_bundle_keys(b: Dict[str, Any]) -> List[int]:
    keys = [int(b["init_policy"]), int(b["conf_test_shared_dW"])]
    keys.extend(int(v) for v in b["train_dW"].values())
    keys.extend(int(v) for v in b["train_perm"].values())
    keys.extend(int(v) for v in b["val_dW"].values())
    return keys


def verify_cost_key_isolation() -> Dict[str, Any]:
    hist = collect_all_historical_keys()
    all_target_seeds = [COST_SMOKE_SEED] + COST_RESERVED_SEEDS
    bundles = {s: derive_cost_seed_bundle(s) for s in all_target_seeds}

    all_ints_with_labels: List[int] = []
    dW_only_keys: List[int] = []
    overflow_errors: List[str] = []

    for s, b in bundles.items():
        all_ints_with_labels.append(int(s))
        flat = _flatten_bundle_keys(b)
        for k in flat:
            if k <= 0 or (k + VOL_SEED_OFFSET) >= 2**32:
                overflow_errors.append(f"seed={s}, key={k}")
        all_ints_with_labels.extend(flat)
        dW_only_keys.extend(int(v) for v in b["train_dW"].values())
        dW_only_keys.extend(int(v) for v in b["val_dW"].values())
        dW_only_keys.append(int(b["conf_test_shared_dW"]))

    unique_all = set(all_ints_with_labels)
    internal_collisions = len(all_ints_with_labels) - len(unique_all)
    hist_collisions = sorted(unique_all.intersection(hist))

    # Explicit cross-seed Val(s) vs Train(s+1) disjointness audit (direct elimination of B3 flaw)
    b3_style_adjacent_collisions: List[str] = []
    for i in range(len(COST_RESERVED_SEEDS) - 1):
        s_curr = COST_RESERVED_SEEDS[i]
        s_next = COST_RESERVED_SEEDS[i + 1]
        vals_curr = set(int(v) for v in bundles[s_curr]["val_dW"].values())
        trains_next = set(int(v) for v in bundles[s_next]["train_dW"].values())
        inter = vals_curr.intersection(trains_next)
        if inter:
            b3_style_adjacent_collisions.append(f"Val({s_curr}) & Train({s_next}) = {sorted(inter)}")

    # Also check secondary vol_seed_offset space has no internal or historical collisions
    vol_offset_ints = {k + VOL_SEED_OFFSET for k in dW_only_keys}
    vol_cross_collisions = sorted(vol_offset_ints.intersection(unique_all | hist))

    sample_b = bundles[COST_LOCKED_COHORT_SEEDS[0]]
    keys_per_seed = len(_flatten_bundle_keys(sample_b))
    passed = (
        len(overflow_errors) == 0
        and internal_collisions == 0
        and len(hist_collisions) == 0
        and len(b3_style_adjacent_collisions) == 0
        and len(vol_cross_collisions) == 0
    )

    return {
        "namespace": NS,
        "isolation_passed": bool(passed),
        "historical_keys_checked": len(hist),
        "new_total_ints_checked": len(all_ints_with_labels),
        "new_unique_ints_checked": len(unique_all),
        "keys_per_seed_excluding_label": keys_per_seed,
        "smoke_seed": COST_SMOKE_SEED,
        "locked_cohort_seeds": COST_LOCKED_COHORT_SEEDS,
        "sealed_unused_seeds": COST_SEALED_UNUSED_SEEDS,
        "overflow_errors": overflow_errors,
        "internal_collision_count": internal_collisions,
        "historical_collisions": hist_collisions,
        "b3_adjacent_val_train_collisions": b3_style_adjacent_collisions,
        "vol_offset_cross_collisions": vol_cross_collisions,
        "hold_eval_status": "SEALED_AND_NEVER_OPENED",
        "prior_4001_4040_status": "UNTOUCHED_AND_EXCLUDED",
        "bundles": bundles,
    }


def write_data_and_seed_manifest() -> Path:
    audit = verify_cost_key_isolation()
    assert audit["isolation_passed"], f"Cost PRNG isolation audit failed: {audit}"

    manifest = {
        "status": "PRE_EXECUTION_KEY_PLAN_NO_TEST_DATA_SIMULATED",
        "namespace": NS,
        "cost_s": 0.0010,
        "cost_bps": 10.0,
        "hold_eval_status": "SEALED_AND_NEVER_OPENED",
        "prior_r2_conf_4001_4040_status": "UNTOUCHED_AND_EXCLUDED",
        "conf_test_status": "KEYS_DERIVED_ONLY_NOT_SIMULATED_OR_OPENED",
        "locked_primary_N": 24,
        "locked_seeds": COST_LOCKED_COHORT_SEEDS,
        "unused_quarantined_pool": COST_SEALED_UNUSED_SEEDS,
        "tech_smoke_seed_excluded_from_analysis": COST_SMOKE_SEED,
        "eval_scenarios_locked": list(COST_EVAL_SCENARIOS),
        "pairing_structure": {
            "within_seed_shared_streams": [
                "init_policy (exact bit-identical policy MLP + idelta initialization)",
                "train_dW['fixed_020'] & train_perm['fixed_020'] (nominal 1/4 sub-stream of both branches)",
                "val_dW['fixed_020'] (nominal validation stream; A selects min val_020, R2 selects min mean val over 4 training scenarios)",
                "conf_test_shared_dW (20,000 shared-dW test paths across 6 scenarios within seed, created ONLY after best_val & ep200 close)",
            ],
            "across_seed_and_split_disjointness": (
                "Zero shared keys across distinct seed_labels, zero shared keys between train / val / conf_test, "
                "and direct resolution of B3's Val(s)/Train(s+1) 5,000-row dW duplication defect."
            ),
        },
        "prng_isolation_audit": {k: v for k, v in audit.items() if k != "bundles"},
        "seed_bundles": {str(s): b for s, b in audit["bundles"].items()},
    }
    out = COST_CONF_DIR / "DATA_AND_SEED_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = write_data_and_seed_manifest()
    d = json.loads(p.read_text(encoding="utf-8"))
    print("COST PRNG AUDIT:", json.dumps(d["prng_isolation_audit"], indent=2))
    print("Saved manifest to", p)
