"""
r3_keys.py: Cryptographically isolated 31-bit PRNG key derivation & historical collision audit.

All R3 seeds are derived deterministically via SHA-256 over explicit namespace tuples:
  (stage, seed_label, dataset, purpose, scenario)
reduced to 31-bit non-negative integers (& 0x7FFFFFFF) so that adding any downstream
legacy offset (e.g., vol_seed_offset = 54321 in robust_world.py) never overflows uint32.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


TRAINING_SCENARIOS = ("fixed_015", "fixed_020", "fixed_025", "rs_015_025")
TEST_SCENARIOS = (
    "fixed_015",
    "fixed_020",
    "fixed_025",
    "rs_015_025",
    "regime_switch_015_028",
    "fixed_030",
)

VOL_SEED_OFFSET = 54321


def sha256_seed31(stage: str, seed_label: int, dataset: str, purpose: str, scenario: str) -> int:
    """Derives an isolated 31-bit unsigned integer seed from explicit hierarchical namespace."""
    payload = f"R3_PROJECT_V1|{stage}|{int(seed_label)}|{dataset}|{purpose}|{scenario}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    val = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    if val == 0:
        val = 1
    return val


def collect_historical_rng_integers() -> Dict[str, List[int]]:
    """
    Enumerates ALL integer RNG seeds and derived offset keys used across every prior
    Deep Hedging experiment on this repository to guarantee strict zero collision.
    """
    hist: Dict[str, Set[int]] = {
        "base_seed_labels": set(),
        "world_and_vol_rng_ints": set(),
    }

    # All prior experiment seed groups
    groups = {
        "v3_and_probes": [42, 43, 44, 45, 46],
        "b3_confirm_10": list(range(101, 111)),
        "robust_stage1_2": [201, 202, 203],
        "robust_stage3_10": list(range(301, 311)),
        "robust_stage4_20": list(range(401, 421)),
        "b3_cost_probe": [501, 502, 503, 504, 505],
        "rough_cpu_probe": [601, 602, 603],
        "b3_cost_confirm_10": list(range(701, 711)),
        "rough_m2p_cpu_10": list(range(801, 811)),
        "r2_proposed_future": list(range(901, 921)),
        "audit_norm_7000": list(range(7001, 7011)),
    }

    for gname, seeds in groups.items():
        for s in seeds:
            hist["base_seed_labels"].add(s)
            hist["world_and_vol_rng_ints"].add(s)
            hist["world_and_vol_rng_ints"].add(s + 1)
            hist["world_and_vol_rng_ints"].add(s + 54321)
            hist["world_and_vol_rng_ints"].add(s + 1 + 54321)
            hist["world_and_vol_rng_ints"].add(s + 999)
            hist["world_and_vol_rng_ints"].add(s + 8888)
            hist["world_and_vol_rng_ints"].add(s + 9999)
            for moff in (555, 655, 755):
                hist["world_and_vol_rng_ints"].add(s + moff)
            # B3 / NTBN & Rough offsets
            for boff in (10000, 20000, 30000, 40000, 50000, 99999, 100000):
                hist["world_and_vol_rng_ints"].add(s + boff)
            # Hash-based or raw offsets from robust_world tests
            for extra in (s + 100, s + 200, s + 300):
                hist["world_and_vol_rng_ints"].add(extra)

    # Also v3 default test scenarios (seed 42 offsets if any)
    for extra_v3 in range(40, 60):
        hist["world_and_vol_rng_ints"].add(extra_v3)
        hist["world_and_vol_rng_ints"].add(extra_v3 * 100)

    return {k: sorted(v) for k, v in hist.items()}


def derive_seed_bundle(stage: str, seed_label: int) -> Dict[str, Any]:
    """
    Produces the complete, disjoint dictionary of 31-bit RNG keys for one dev/conf seed.
    Includes BOTH primary keys and secondary vol_seed = key + 54321 for collision audit.
    """
    bundle: Dict[str, Any] = {
        "stage": stage,
        "seed_label": int(seed_label),
        "init_policy": sha256_seed31(stage, seed_label, "model", "init_policy", "shared"),
        "train_dW": {},
        "train_vol_effective": {},
        "train_perm": {},
        "val_dW": {},
        "test_dW_shared_pair": sha256_seed31(stage, seed_label, "test", "dW", "shared_pair"),
    }

    # Training streams: A_ref uses fixed_020; R2_matched & R3 use fixed_015, fixed_020, fixed_025, rs_015_025.
    # Importantly, A_ref and R3/R2_matched share the SAME fixed_020 train dW stream and batch permutation
    # key so L_nom(theta; B_b) and L_nom(A_ref; B_b) are strictly 1-to-1 paired on every mini-batch!
    for sc in TRAINING_SCENARIOS:
        k = sha256_seed31(stage, seed_label, "train", "dW", sc)
        bundle["train_dW"][sc] = k
        bundle["train_perm"][sc] = sha256_seed31(stage, seed_label, "train", "perm", sc)

    # R2 uses an independent train dW stream and independent static_path vol stream (+54321 inside robust_world)
    r2_k = sha256_seed31(stage, seed_label, "train", "dW", "r2_static_path")
    bundle["train_dW"]["r2_static_path"] = r2_k
    bundle["train_vol_effective"]["r2_static_path"] = r2_k + VOL_SEED_OFFSET
    bundle["train_perm"]["r2_static_path"] = sha256_seed31(stage, seed_label, "train", "perm", "r2_static_path")

    # Validation streams (independent per scenario)
    for sc in TRAINING_SCENARIOS:
        bundle["val_dW"][sc] = sha256_seed31(stage, seed_label, "val", "dW", sc)

    return bundle


def verify_key_isolation(stage: str, seed_labels: List[int]) -> Dict[str, Any]:
    """
    Audits that all derived 31-bit keys across `seed_labels`:
      1. Fit strictly inside [1, 2^31 - 1 + VOL_SEED_OFFSET] < 2^32 - 1 (no uint32 overflow).
      2. Have ZERO duplicates within the entire R3 study (every key used for only 1 purpose).
      3. Have ZERO overlap with any seed label or offset integer from all prior experiments.
    """
    hist = collect_historical_rng_integers()
    hist_all: Set[int] = set(hist["base_seed_labels"]) | set(hist["world_and_vol_rng_ints"])

    bundles = [derive_seed_bundle(stage, s) for s in seed_labels]
    used_records: List[Tuple[int, str]] = []
    label_collisions: List[int] = []

    for b in bundles:
        s = b["seed_label"]
        if s in hist_all:
            label_collisions.append(s)
        used_records.append((b["init_policy"], f"seed{s}.init_policy"))
        used_records.append((b["test_dW_shared_pair"], f"seed{s}.test_dW_shared_pair"))
        for sc, val in b["train_dW"].items():
            used_records.append((val, f"seed{s}.train_dW.{sc}"))
        for sc, val in b["train_vol_effective"].items():
            used_records.append((val, f"seed{s}.train_vol_eff.{sc}"))
        for sc, val in b["train_perm"].items():
            used_records.append((val, f"seed{s}.train_perm.{sc}"))
        for sc, val in b["val_dW"].items():
            used_records.append((val, f"seed{s}.val_dW.{sc}"))

    seen_vals: Dict[int, str] = {}
    internal_collisions: List[Dict[str, Any]] = []
    historical_collisions: List[Dict[str, Any]] = []
    overflow_errors: List[Dict[str, Any]] = []

    for val, name in used_records:
        if val < 1 or val > 0xFFFFFFFF or (".train_dW." in name and val + VOL_SEED_OFFSET > 0xFFFFFFFF):
            overflow_errors.append({"key_name": name, "value": val})
        if val in seen_vals:
            internal_collisions.append({"value": val, "first": seen_vals[val], "second": name})
        else:
            seen_vals[val] = name
        if val in hist_all:
            historical_collisions.append({"key_name": name, "value": val})

    ok = (
        len(label_collisions) == 0
        and len(internal_collisions) == 0
        and len(historical_collisions) == 0
        and len(overflow_errors) == 0
    )
    return {
        "isolation_passed": ok,
        "stage": stage,
        "seed_labels": list(seed_labels),
        "total_keys_checked": len(used_records),
        "label_collisions": label_collisions,
        "internal_collisions": internal_collisions,
        "historical_collisions": historical_collisions,
        "overflow_errors": overflow_errors,
        "bundles": bundles,
    }
