"""
scripts/run_single_config.py: CLI runner for a single configuration and branch.

Usage:
  python run_single_config.py --config_id blackscholes__trainer_bs_delta --branch both --epochs 10 --seed 42
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
import numpy as np

# Set python path
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
REPO_ROOT = WORKSPACE_ROOT / "deephedging-jax"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from configs.catalog import get_config_spec, CATALOG
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.configs.catalog import get_config_spec, CATALOG
try:
    from adapters import get_adapter
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.adapters import get_adapter
try:
    from core.train import TrainConfig, train_paired
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.train import TrainConfig, train_paired
try:
    from core.model import (
        rollout_original, rollout_ntbn, count_parameters
    )
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.model import (
        rollout_original, rollout_ntbn, count_parameters
    )
try:
    from core.metrics import (
        evaluate_rollout_metrics, compute_paired_comparison
    )
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.metrics import (
        evaluate_rollout_metrics, compute_paired_comparison
    )


def main():
    parser = argparse.ArgumentParser(description="Run single Deep Hedging / NTBN config.")
    parser.add_argument("--config_id", type=str, required=True, help="Config ID from catalog")
    parser.add_argument("--branch", type=str, default="both", choices=["A", "B", "both"], help="Branch to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save/load checkpoints")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--max_session_seconds", type=float, default=None, help="Maximum GPU session run duration in seconds")
    args = parser.parse_args()

    spec = get_config_spec(args.config_id)
    adapter = get_adapter(spec.group_id)
    gym_a, gym_b, train_cfg_base = adapter.get_variant_configs(spec.variant)

    epochs = args.epochs if args.epochs is not None else spec.epochs
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        clipvalue=train_cfg_base.clipvalue,
        global_clipnorm=train_cfg_base.global_clipnorm,
        clipnorm=train_cfg_base.clipnorm,
        seed=args.seed
    )

    print("=" * 80)
    print(f"RUNNING CONFIG: {spec.config_id}")
    print(f"Group: {spec.group_id} | Variant: {spec.variant}")
    print(f"Theoretical Status: {spec.theoretical_status}")
    print(f"Scale: {spec.train_samples} samples, {spec.steps} steps, dt={spec.dt}, {epochs} epochs, batch {spec.batch_size}")
    print(f"Branch(es): {args.branch} | Seed: {args.seed}")
    print("=" * 80)

    # 1. Build datasets (Frozen and shared)
    print("Building worlds and freezing datasets...")
    train_batch, val_batch, test_batch, feat_dims, n_inst = adapter.build_datasets(seed=args.seed)

    results = {
        "config_id": spec.config_id,
        "group_id": spec.group_id,
        "variant": spec.variant,
        "theoretical_status": spec.theoretical_status,
        "seed": args.seed,
        "epochs": epochs,
        "branches": {}
    }

    roll_a, roll_b = None, None
    met_a, met_b = None, None
    all_completed = True

    base_chk_dir = args.checkpoint_dir or str(Path(__file__).resolve().parent.parent / "checkpoints" / spec.config_id)

    # Run Branch A
    if args.branch in ("A", "both"):
        print("\n>>> Training Branch A (Original JAX)...")
        t0 = time.time()
        chk_prefix_a = f"{spec.config_id}_seed{args.seed}_branch_A"
        res_a = train_paired(
            "A", gym_a, train_cfg, train_batch, val_batch,
            key=args.seed, feature_dims=feat_dims, n_inst=n_inst,
            log_every=max(1, epochs // 5),
            checkpoint_dir=base_chk_dir,
            checkpoint_prefix=chk_prefix_a,
            resume=not args.no_resume,
            max_session_seconds=args.max_session_seconds
        )
        if not res_a.completed:
            all_completed = False
            print(f"[Branch A] Session reached limit at epoch {res_a.last_epoch + 1}/{epochs}. Checkpoint saved. Run again to continue.")
        else:
            roll_a = rollout_original(res_a.params, gym_a, test_batch)
            met_a = evaluate_rollout_metrics(roll_a)
            wall_time_a = time.time() - t0
            
            results["branches"]["A"] = {
                "best_epoch": res_a.best_epoch,
                "best_val_loss": float(res_a.best_val_loss),
                "train_time_sec": res_a.train_time_sec,
                "wall_time_sec": wall_time_a,
                "test_metrics": met_a,
                "param_count": count_parameters(res_a.params),
                "completed": True
            }
            print(f"Branch A Finished: Best Epoch {res_a.best_epoch + 1}, Test Loss: {met_a['loss']:.5f}, CVaR95: {met_a['cvar95']:.5f}, Time: {wall_time_a:.2f}s")

    # Run Branch B (NTBN)
    if args.branch in ("B", "both"):
        print("\n>>> Training Branch B (JAX + NTBN)...")
        t0 = time.time()
        chk_prefix_b = f"{spec.config_id}_seed{args.seed}_branch_B"
        res_b = train_paired(
            "B", gym_b, train_cfg, train_batch, val_batch,
            key=args.seed, feature_dims=feat_dims, n_inst=n_inst,
            log_every=max(1, epochs // 5),
            checkpoint_dir=base_chk_dir,
            checkpoint_prefix=chk_prefix_b,
            resume=not args.no_resume,
            max_session_seconds=args.max_session_seconds
        )
        if not res_b.completed:
            all_completed = False
            print(f"[Branch B] Session reached limit at epoch {res_b.last_epoch + 1}/{epochs}. Checkpoint saved. Run again to continue.")
        else:
            roll_b = rollout_ntbn(res_b.params, gym_b, test_batch)
            met_b = evaluate_rollout_metrics(roll_b)
            wall_time_b = time.time() - t0

            results["branches"]["B"] = {
                "best_epoch": res_b.best_epoch,
                "best_val_loss": float(res_b.best_val_loss),
                "train_time_sec": res_b.train_time_sec,
                "wall_time_sec": wall_time_b,
                "test_metrics": met_b,
                "param_count": count_parameters(res_b.params),
                "completed": True
            }
            print(f"Branch B Finished: Best Epoch {res_b.best_epoch + 1}, Test Loss: {met_b['loss']:.5f}, CVaR95: {met_b['cvar95']:.5f}, Time: {wall_time_b:.2f}s")

    if not all_completed:
        print("\n[PARTIAL RUN] Training paused due to session limits. Results not yet finalized. Checkpoints are ready for next session.")
        sys.exit(10) # Exit code 10 signals 'interrupted / checkpoint ready to continue'

    # Paired comparison if both ran
    if roll_a is not None and roll_b is not None:
        paired = compute_paired_comparison(met_a, met_b, roll_a, roll_b)
        results["paired_comparison"] = paired
        print("\n" + "=" * 50)
        print("PAIRED COMPARISON (Branch B [NTBN] - Branch A [Orig]):")
        print(f"  Delta Test Loss : {paired['delta_loss']:+.6f}")
        print(f"  Delta CVaR95    : {paired['delta_cvar95']:+.6f}")
        print(f"  Delta Turnover  : {paired['delta_turnover']:+.6f}")
        print(f"  Delta Mean Cost : {paired['delta_mean_cost']:+.6f}")
        print("=" * 50)

    # Save output
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"result_{spec.config_id}_seed{args.seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
