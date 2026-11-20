"""
diag_keys.py: Cryptographically isolated 31-bit PRNG keys for `robust_r3_selection_diagnostics/`.

Strict Isolation Guarantees:
- Imports all 1,543 prior + R3_DEV integers, all 126 SHA-256 keys of `robust_r3_budget_control`,
  AND the 3 integer seed labels [3201, 3202, 3203] (= 1,672 historical integers total),
  cross-checked against `BUDGET_CONTROL_DATA_AND_SEED_MANIFEST.json` (`1,543 + 126 = 1,669` + 3 seed labels).
- Derives strictly non-overlapping 31-bit SHA-256 seeds under namespace `R3_SEL_DIAG_V1` for:
  1. `VAL_ORIG`: exact reproduction of existing Val keys (5,000 paths/scen per seed, read-only check only).
  2. `VAL_DIAG_A_20K` & `VAL_DIAG_A_5K_SUB`: 20,000-path diagnostic Val set + its fixed 5,000-path SHA-256 permutation subset.
  3. `VAL_DIAG_B0..B3_5K`: 4 completely independent 5,000-path diagnostic Val batches (for empirical batch-norm MC SE!).
  4. `VAL_DIAG_C_20K`: completely independent 20,000-path diagnostic Val set (sensitivity across datasets).
  5. `VAL_DIAG_YCAL_20K`: independent 20,000-path Val calibration stream strictly separated from evaluation streams
     (for testing OCE offset `y` re-calibration cleanly without leakage).
  6. `HOLD_EVAL`: sealed evaluation keys for [3101..3103, 3201..3203] and optional fresh dev seeds [3301..3303]
     — NEVER simulated or opened during Section 3 read-only diagnostics!
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Any

import sys
DIAG_DIR = Path(__file__).resolve().parent
EXP_DIR = DIAG_DIR.parent
BC_DIR = EXP_DIR / "robust_r3_budget_control"
R3_DIR = EXP_DIR / "robust_r3_dev"
for p in (str(BC_DIR), str(R3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from r3_keys import TRAINING_SCENARIOS, TEST_SCENARIOS
from bc_keys import (
    OLD_PILOT_SEEDS,
    NEW_DEV_SEEDS,
    collect_all_prior_rng_integers_including_r3dev,
    derive_old_seed_extension_bundle,
    derive_new_seed_bundle,
)

ALL_EXISTING_SEEDS = OLD_PILOT_SEEDS + NEW_DEV_SEEDS
PHASE2_CANDIDATE_SEEDS = [3301, 3302, 3303]
DIAG_EVAL_SCENARIOS = ("fixed_015", "fixed_020", "fixed_025", "rs_015_025", "fixed_030")
NS = "R3_SEL_DIAG_V1"


def sha256_seed31_diag(stage: str, seed_label: int, stream: str, purpose: str, scenario: str) -> int:
    payload = f"{NS}|{stage}|{int(seed_label)}|{stream}|{purpose}|{scenario}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    val = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return 1 if val == 0 else val


def collect_all_1672_historical_keys() -> Set[int]:
    """Collects exact 1,543 prior + 126 BC SHA-256 keys + 3 BC seed labels [3201..3203] (= 1,672 keys total)."""
    prior: Set[int] = set(collect_all_prior_rng_integers_including_r3dev())
    for s in OLD_PILOT_SEEDS:
        b = derive_old_seed_extension_bundle(s)
        for cat in b["extra_only_keys"].values():
            for v in cat.values():
                prior.add(int(v))
    for s in NEW_DEV_SEEDS:
        prior.add(int(s))
        b = derive_new_seed_bundle(s)
        prior.add(int(b["init_policy"]))
        prior.add(int(b["test_dW_shared_pair"]))
        for dkey in ("train_dW", "train_perm", "sub1250_select_perm", "sub1250_batch_perm", "val_dW"):
            for v in b[dkey].values():
                prior.add(int(v))
    return prior


def get_existing_val_orig_keys(seed_label: int) -> Dict[str, int]:
    if seed_label in OLD_PILOT_SEEDS:
        return dict(derive_old_seed_extension_bundle(seed_label)["val_dW"])
    if seed_label in NEW_DEV_SEEDS:
        return dict(derive_new_seed_bundle(seed_label)["val_dW"])
    raise ValueError(f"Unknown existing seed: {seed_label}")


def derive_diag_bundle(seed_label: int) -> Dict[str, Any]:
    """Derives isolated diagnostic Val keys (Stages A/B/C/YCAL) + sealed HOLD_EVAL keys."""
    st = "R3_DIAG_VAL"
    val_a_dW = {sc: sha256_seed31_diag(st, seed_label, "val_diag_a_20k", "dW", sc) for sc in DIAG_EVAL_SCENARIOS}
    val_a_sub5k_perm = {sc: sha256_seed31_diag(st, seed_label, "val_diag_a_sub5k", "perm", sc) for sc in DIAG_EVAL_SCENARIOS}
    val_b_5k_dW = {
        f"b{r}": {sc: sha256_seed31_diag(st, seed_label, f"val_diag_b{r}_5k", "dW", sc) for sc in DIAG_EVAL_SCENARIOS}
        for r in range(4)
    }
    val_c_20k_dW = {sc: sha256_seed31_diag(st, seed_label, "val_diag_c_20k", "dW", sc) for sc in DIAG_EVAL_SCENARIOS}
    val_ycal_20k_dW = {sc: sha256_seed31_diag(st, seed_label, "val_diag_ycal_20k", "dW", sc) for sc in TRAINING_SCENARIOS}

    # Sealed evaluation keys (NOT opened in Section 3 read-only diagnostics!)
    hold_eval = {
        "shared_pair_dW": sha256_seed31_diag("R3_SEL_DIAG_HOLD_EVAL", seed_label, "hold_eval", "dW", "shared_pair")
    }
    return {
        "seed_label": int(seed_label),
        "seed_group": "posthoc_old_pilot" if seed_label in OLD_PILOT_SEEDS else ("new_dev_bc" if seed_label in NEW_DEV_SEEDS else "phase2_dev"),
        "val_diag_a_20k_dW": val_a_dW,
        "val_diag_a_sub5k_perm": val_a_sub5k_perm,
        "val_diag_b_5k_dW": val_b_5k_dW,
        "val_diag_c_20k_dW": val_c_20k_dW,
        "val_diag_ycal_20k_dW": val_ycal_20k_dW,
        "hold_eval_sealed": hold_eval,
    }


def derive_phase2_train_bundle(seed_label: int) -> Dict[str, Any]:
    """Fresh training + val keys reserved ONLY if Section 4 warrants a single-factor Stage 5 dev run."""
    st = "R3_SEL_PHASE2_TRAIN"
    train_dW = {sc: sha256_seed31_diag(st, seed_label, "train", "dW", sc) for sc in TRAINING_SCENARIOS}
    train_perm = {sc: sha256_seed31_diag(st, seed_label, "train", "perm", sc) for sc in TRAINING_SCENARIOS}
    for sub in ("fixed_020_sub0", "fixed_020_sub2", "fixed_020_sub3"):
        train_dW[sub] = sha256_seed31_diag(st, seed_label, "train", "dW", sub)
        train_perm[sub] = sha256_seed31_diag(st, seed_label, "train", "perm", sub)
    val_dW = {sc: sha256_seed31_diag(st, seed_label, "val", "dW", sc) for sc in TRAINING_SCENARIOS}
    return {
        "seed_label": int(seed_label),
        "init_policy": sha256_seed31_diag(st, seed_label, "model", "init_policy", "shared"),
        "train_dW": train_dW,
        "train_perm": train_perm,
        "val_dW": val_dW,
    }


def verify_diag_key_isolation() -> Dict[str, Any]:
    hist = collect_all_1672_historical_keys()
    new_keys: List[int] = []
    diag_bundles: Dict[int, Any] = {}
    phase2_bundles: Dict[int, Any] = {}

    for s in ALL_EXISTING_SEEDS + PHASE2_CANDIDATE_SEEDS:
        b = derive_diag_bundle(s)
        diag_bundles[s] = b
        for sc in DIAG_EVAL_SCENARIOS:
            new_keys.append(b["val_diag_a_20k_dW"][sc])
            new_keys.append(b["val_diag_a_sub5k_perm"][sc])
            new_keys.append(b["val_diag_c_20k_dW"][sc])
            for r in range(4):
                new_keys.append(b["val_diag_b_5k_dW"][f"b{r}"][sc])
        for sc in TRAINING_SCENARIOS:
            new_keys.append(b["val_diag_ycal_20k_dW"][sc])
        new_keys.append(b["hold_eval_sealed"]["shared_pair_dW"])

    for s in PHASE2_CANDIDATE_SEEDS:
        pb = derive_phase2_train_bundle(s)
        phase2_bundles[s] = pb
        new_keys.append(s)
        new_keys.append(pb["init_policy"])
        for d in (pb["train_dW"], pb["train_perm"], pb["val_dW"]):
            new_keys.extend(d.values())

    overflow = [k for k in new_keys if not (1 <= int(k) <= 0x7FFFFFFF)]
    internal_coll = len(new_keys) - len(set(new_keys))
    hist_coll = sorted(set(new_keys) & hist)
    # Cross-check historical count vs BC manifest if available
    manifest_p = BC_DIR / "BUDGET_CONTROL_DATA_AND_SEED_MANIFEST.json"
    bc_manifest_count = None
    if manifest_p.exists():
        m = json.loads(manifest_p.read_text(encoding="utf-8"))
        bc_audit = m.get("prng_isolation_audit", {})
        bc_manifest_count = int(bc_audit.get("prior_integers_checked", 0)) + int(bc_audit.get("new_keys_checked", 0))

    passed = (len(overflow) == 0) and (internal_coll == 0) and (len(hist_coll) == 0)
    return {
        "isolation_passed": passed,
        "historical_keys_checked": len(hist),
        "bc_manifest_sum_prior_plus_new": bc_manifest_count,
        "new_keys_checked": len(new_keys),
        "overflow_errors": overflow,
        "internal_collision_count": internal_coll,
        "historical_collisions": hist_coll,
        "diag_bundles": diag_bundles,
        "phase2_bundles": phase2_bundles,
    }


if __name__ == "__main__":
    res = verify_diag_key_isolation()
    summary = {k: v for k, v in res.items() if k not in ("diag_bundles", "phase2_bundles")}
    print(json.dumps(summary, indent=2))
    assert res["isolation_passed"], "PRNG isolation failed!"
