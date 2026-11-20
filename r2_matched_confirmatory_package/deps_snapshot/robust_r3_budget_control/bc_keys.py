"""
bc_keys.py: Cryptographically isolated 31-bit PRNG keys for robust_r3_budget_control.

Mandatory @gng5 Fix #1 & #2 & #3:
- Imports / reproduces ALL historical RNG integers PLUS ALL 51 SHA-256 derived keys
  from `robust_r3_dev` (seeds [3101, 3102, 3103]) into `hist_all` for zero-collision checking.
- On BOTH old [3101..3103] and new [3201..3203], `A_ref_128`'s sub-pool k=1 uses the EXACT
  same `train_dW['fixed_020']` and `train_perm['fixed_020']` as `A_ref_32` and `R2_matched_128` k=1.
- Provides a fixed deterministic `sub1250_perm` key per scenario for zero-confound `R2_matched_32`
  subsampling from the M=5,000 normalized pool before epoch 0.
"""

import hashlib
from typing import Dict, List, Set, Tuple, Any
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
R3_DIR = EXP_DIR / "robust_r3_dev"
if str(R3_DIR) not in sys.path:
    sys.path.insert(0, str(R3_DIR))

from r3_keys import (
    TRAINING_SCENARIOS,
    TEST_SCENARIOS,
    VOL_SEED_OFFSET,
    collect_historical_rng_integers,
    derive_seed_bundle as derive_r3_dev_bundle,
)

OLD_PILOT_SEEDS = [3101, 3102, 3103]
NEW_DEV_SEEDS = [3201, 3202, 3203]


def sha256_seed31_ns(namespace: str, stage: str, seed_label: int, dataset: str, purpose: str, scenario: str) -> int:
    payload = f"{namespace}|{stage}|{int(seed_label)}|{dataset}|{purpose}|{scenario}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    val = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return 1 if val == 0 else val


def collect_all_prior_rng_integers_including_r3dev() -> Set[int]:
    hist = collect_historical_rng_integers()
    prior: Set[int] = set(hist["base_seed_labels"]) | set(hist["world_and_vol_rng_ints"])
    for s in OLD_PILOT_SEEDS:
        prior.add(s)
        b = derive_r3_dev_bundle("R3_DEV", s)
        prior.add(b["init_policy"])
        prior.add(b["test_dW_shared_pair"])
        for v in b["train_dW"].values():
            prior.add(v)
        for v in b["train_vol_effective"].values():
            prior.add(v)
        for v in b["train_perm"].values():
            prior.add(v)
        for v in b["val_dW"].values():
            prior.add(v)
    return prior


def derive_old_seed_extension_bundle(seed_label: int) -> Dict[str, Any]:
    """
    For old pilot seeds [3101, 3102, 3103]:
    - Preserves exact R3_DEV keys for init_policy, fixed_020 (sub1), all val_dW, and test_dW_shared_pair.
    - Adds strictly isolated new keys under R3_BUDGET_CTRL_POSTHOC for:
      * A_ref_128 extra 3 fixed_020 sub-pools (sub0, sub2, sub3) + their permutations
      * R2_matched_32 fixed permutation keys to draw 1,250 paths from M=5,000 pools + batch perm keys.
    """
    base = derive_r3_dev_bundle("R3_DEV", seed_label)
    ns = "R3_BC_V1"
    st = "R3_BUDGET_CTRL_POSTHOC"

    extra_train_dW = {}
    extra_train_perm = {}
    for sub_name in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        extra_train_dW[sub_name] = sha256_seed31_ns(ns, st, seed_label, "train", "dW", sub_name)
        extra_train_perm[sub_name] = sha256_seed31_ns(ns, st, seed_label, "train", "perm", sub_name)

    sub1250_select_perm = {
        sc: sha256_seed31_ns(ns, st, seed_label, "train_32", "sub1250_select", sc)
        for sc in TRAINING_SCENARIOS
    }
    sub1250_batch_perm = {
        sc: sha256_seed31_ns(ns, st, seed_label, "train_32", "perm_1250", sc)
        for sc in TRAINING_SCENARIOS
    }

    return {
        "seed_group": "posthoc_old_pilot",
        "seed_label": int(seed_label),
        "init_policy": base["init_policy"],
        "train_dW": {**base["train_dW"], **extra_train_dW},
        "train_perm": {**base["train_perm"], **extra_train_perm},
        "sub1250_select_perm": sub1250_select_perm,
        "sub1250_batch_perm": sub1250_batch_perm,
        "val_dW": base["val_dW"],
        "test_dW_shared_pair": base["test_dW_shared_pair"],
        "extra_only_keys": {
            "extra_train_dW": extra_train_dW,
            "extra_train_perm": extra_train_perm,
            "sub1250_select_perm": sub1250_select_perm,
            "sub1250_batch_perm": sub1250_batch_perm,
        },
    }


def derive_new_seed_bundle(seed_label: int) -> Dict[str, Any]:
    """
    For pristine new dev seeds [3201, 3202, 3203]:
    All keys freshly derived under namespace R3_BC_V1 | R3_BUDGET_DEV.
    """
    ns = "R3_BC_V1"
    st = "R3_BUDGET_DEV"

    train_dW = {
        sc: sha256_seed31_ns(ns, st, seed_label, "train", "dW", sc)
        for sc in TRAINING_SCENARIOS
    }
    train_perm = {
        sc: sha256_seed31_ns(ns, st, seed_label, "train", "perm", sc)
        for sc in TRAINING_SCENARIOS
    }
    for sub_name in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        train_dW[sub_name] = sha256_seed31_ns(ns, st, seed_label, "train", "dW", sub_name)
        train_perm[sub_name] = sha256_seed31_ns(ns, st, seed_label, "train", "perm", sub_name)

    sub1250_select_perm = {
        sc: sha256_seed31_ns(ns, st, seed_label, "train_32", "sub1250_select", sc)
        for sc in TRAINING_SCENARIOS
    }
    sub1250_batch_perm = {
        sc: sha256_seed31_ns(ns, st, seed_label, "train_32", "perm_1250", sc)
        for sc in TRAINING_SCENARIOS
    }
    val_dW = {
        sc: sha256_seed31_ns(ns, st, seed_label, "val", "dW", sc)
        for sc in TRAINING_SCENARIOS
    }

    return {
        "seed_group": "new_dev",
        "seed_label": int(seed_label),
        "init_policy": sha256_seed31_ns(ns, st, seed_label, "model", "init_policy", "shared"),
        "train_dW": train_dW,
        "train_perm": train_perm,
        "sub1250_select_perm": sub1250_select_perm,
        "sub1250_batch_perm": sub1250_batch_perm,
        "val_dW": val_dW,
        "test_dW_shared_pair": sha256_seed31_ns(ns, st, seed_label, "test", "dW", "shared_pair"),
    }


def verify_bc_key_isolation() -> Dict[str, Any]:
    """
    Verifies zero internal collisions, zero uint32 overflow, and zero collision
    with ALL prior historical seeds/offsets + ALL 51 R3_DEV keys!
    """
    prior_all = collect_all_prior_rng_integers_including_r3dev()
    old_bundles = [derive_old_seed_extension_bundle(s) for s in OLD_PILOT_SEEDS]
    new_bundles = [derive_new_seed_bundle(s) for s in NEW_DEV_SEEDS]

    new_records: List[Tuple[int, str]] = []
    label_collisions: List[int] = []

    for b in old_bundles:
        s = b["seed_label"]
        ek = b["extra_only_keys"]
        for cat, d in ek.items():
            for sc, val in d.items():
                new_records.append((val, f"old{s}.{cat}.{sc}"))

    for b in new_bundles:
        s = b["seed_label"]
        if s in prior_all:
            label_collisions.append(s)
        new_records.append((b["init_policy"], f"new{s}.init_policy"))
        new_records.append((b["test_dW_shared_pair"], f"new{s}.test_dW_shared_pair"))
        for dname in ("train_dW", "train_perm", "sub1250_select_perm", "sub1250_batch_perm", "val_dW"):
            for sc, val in b[dname].items():
                new_records.append((val, f"new{s}.{dname}.{sc}"))

    seen: Dict[int, str] = {}
    internal_collisions: List[Dict[str, Any]] = []
    historical_collisions: List[Dict[str, Any]] = []
    overflow_errors: List[Dict[str, Any]] = []

    for val, name in new_records:
        if val < 1 or val + VOL_SEED_OFFSET > 0xFFFFFFFF:
            overflow_errors.append({"key_name": name, "value": val})
        if val in seen:
            internal_collisions.append({"value": val, "first": seen[val], "second": name})
        else:
            seen[val] = name
        if val in prior_all:
            historical_collisions.append({"key_name": name, "value": val})

    ok = (
        len(label_collisions) == 0
        and len(internal_collisions) == 0
        and len(historical_collisions) == 0
        and len(overflow_errors) == 0
    )
    return {
        "isolation_passed": ok,
        "prior_integers_checked": len(prior_all),
        "new_keys_checked": len(new_records),
        "label_collisions": label_collisions,
        "internal_collisions": internal_collisions,
        "historical_collisions": historical_collisions,
        "overflow_errors": overflow_errors,
        "old_bundles": old_bundles,
        "new_bundles": new_bundles,
    }
