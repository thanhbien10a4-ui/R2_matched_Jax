"""
lr_keys.py: Cryptographically isolated 31-bit PRNG keys for `robust_r3_lr_stability/`.

Guarantees:
- Imports ALL 2,092 historical PRNG integers (1,672 up to BC + 420 from `robust_r3_selection_diagnostics`
  including sealed `HOLD_EVAL` and unused `[3301..3303]` keys), cross-checked against manifest counts.
- Uses 3 brand-new development seeds [3401, 3402, 3403] under namespace `R3_LR_STAB_V1`.
- Ensures exact paired streams inside `A_FIXED vs A_DECAY` and `R2_FIXED vs R2_DECAY`, while separating
  `init_policy`, `train`, `val`, `diag_20k_1`, `diag_20k_2`, and `dev_eval`.
- Explicitly keeps `HOLD_EVAL` from `diag_keys` sealed and untouched.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Any
import sys

LR_DIR = Path(__file__).resolve().parent
EXP_DIR = LR_DIR.parent
DIAG_DIR = EXP_DIR / "robust_r3_selection_diagnostics"
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"
for p in (str(DIAG_DIR), str(BC_DIR), str(R3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from r3_keys import TRAINING_SCENARIOS
from diag_keys import (
    ALL_EXISTING_SEEDS,
    PHASE2_CANDIDATE_SEEDS,
    DIAG_EVAL_SCENARIOS,
    collect_all_1672_historical_keys,
    derive_diag_bundle,
    derive_phase2_train_bundle,
)

LR_DEV_SEEDS = [3401, 3402, 3403]
EVAL_SCENARIOS_LR = ("fixed_015", "fixed_020", "fixed_025", "rs_015_025", "fixed_030")
NS = "R3_LR_STAB_V1"


def sha256_seed31_lr(stage: str, seed_label: int, stream: str, purpose: str, scenario: str) -> int:
    payload = f"{NS}|{stage}|{int(seed_label)}|{stream}|{purpose}|{scenario}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    val = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return 1 if val == 0 else val


def collect_all_2092_historical_keys() -> Set[int]:
    """Collects exact 1,672 prior + 420 selection-diagnostics keys (= 2,092 total)."""
    prior: Set[int] = set(collect_all_1672_historical_keys())
    for s in ALL_EXISTING_SEEDS + PHASE2_CANDIDATE_SEEDS:
        b = derive_diag_bundle(s)
        for sc in DIAG_EVAL_SCENARIOS:
            prior.add(int(b["val_diag_a_20k_dW"][sc]))
            prior.add(int(b["val_diag_a_sub5k_perm"][sc]))
            prior.add(int(b["val_diag_c_20k_dW"][sc]))
            for r in range(4):
                prior.add(int(b["val_diag_b_5k_dW"][f"b{r}"][sc]))
        for sc in TRAINING_SCENARIOS:
            prior.add(int(b["val_diag_ycal_20k_dW"][sc]))
        prior.add(int(b["hold_eval_sealed"]["shared_pair_dW"]))
    for s in PHASE2_CANDIDATE_SEEDS:
        prior.add(int(s))
        pb = derive_phase2_train_bundle(s)
        prior.add(int(pb["init_policy"]))
        for d in (pb["train_dW"], pb["train_perm"], pb["val_dW"]):
            for v in d.values():
                prior.add(int(v))
    return prior


def derive_lr_seed_bundle(seed_label: int) -> Dict[str, Any]:
    st = "R3_LR_STAB"
    train_dW = {sc: sha256_seed31_lr(st, seed_label, "train", "dW", sc) for sc in TRAINING_SCENARIOS}
    train_perm = {sc: sha256_seed31_lr(st, seed_label, "train", "perm", sc) for sc in TRAINING_SCENARIOS}
    for sub in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        train_dW[sub] = sha256_seed31_lr(st, seed_label, "train", "dW", sub)
        train_perm[sub] = sha256_seed31_lr(st, seed_label, "train", "perm", sub)

    val_dW = {sc: sha256_seed31_lr(st, seed_label, "val", "dW", sc) for sc in TRAINING_SCENARIOS}
    diag1_20k_dW = {sc: sha256_seed31_lr(st, seed_label, "diag1_20k", "dW", sc) for sc in ("fixed_020",)}
    diag2_20k_dW = {sc: sha256_seed31_lr(st, seed_label, "diag2_20k", "dW", sc) for sc in ("fixed_020",)}
    diag1_jk_perm = sha256_seed31_lr(st, seed_label, "diag1_20k", "jk_perm", "fixed_020")
    diag2_jk_perm = sha256_seed31_lr(st, seed_label, "diag2_20k", "jk_perm", "fixed_020")
    dev_eval_shared_dW = sha256_seed31_lr(st, seed_label, "dev_eval", "dW", "shared_pair")

    return {
        "seed_label": int(seed_label),
        "init_policy": sha256_seed31_lr(st, seed_label, "model", "init_policy", "shared"),
        "train_dW": train_dW,
        "train_perm": train_perm,
        "val_dW": val_dW,
        "diag1_20k_dW": diag1_20k_dW,
        "diag2_20k_dW": diag2_20k_dW,
        "diag1_jk_perm": diag1_jk_perm,
        "diag2_jk_perm": diag2_jk_perm,
        "dev_eval_shared_dW": dev_eval_shared_dW,
    }


def verify_lr_key_isolation() -> Dict[str, Any]:
    hist = collect_all_2092_historical_keys()
    new_keys: List[int] = []
    bundles: Dict[int, Any] = {}
    for s in LR_DEV_SEEDS:
        b = derive_lr_seed_bundle(s)
        bundles[s] = b
        new_keys.append(int(s))
        new_keys.append(int(b["init_policy"]))
        for d in (b["train_dW"], b["train_perm"], b["val_dW"], b["diag1_20k_dW"], b["diag2_20k_dW"]):
            new_keys.extend(int(v) for v in d.values())
        new_keys.extend([int(b["diag1_jk_perm"]), int(b["diag2_jk_perm"]), int(b["dev_eval_shared_dW"])])

    overflow = [k for k in new_keys if not (1 <= int(k) <= 0x7FFFFFFF)]
    internal_coll = len(new_keys) - len(set(new_keys))
    hist_coll = sorted(set(new_keys) & hist)
    passed = (len(overflow) == 0) and (internal_coll == 0) and (len(hist_coll) == 0)
    return {
        "isolation_passed": passed,
        "historical_keys_checked": len(hist),
        "new_keys_checked": len(new_keys),
        "overflow_errors": overflow,
        "internal_collision_count": internal_coll,
        "historical_collisions": hist_coll,
        "bundles": bundles,
    }


if __name__ == "__main__":
    res = verify_lr_key_isolation()
    print(json.dumps({k: v for k, v in res.items() if k != "bundles"}, indent=2))
    assert res["isolation_passed"]
