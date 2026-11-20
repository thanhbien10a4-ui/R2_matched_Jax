"""
scripts/run_benchmark.py: Master Benchmark Orchestrator across all 24 configurations.

Supports:
- Running all 7 groups or subset
- Full scale or smoke mode (--smoke flag)
- Compiling consolidated summary JSON and markdown reports
"""

import argparse
import json
import time
import sys
from pathlib import Path
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from configs.catalog import CATALOG, ConfigSpec
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.configs.catalog import CATALOG, ConfigSpec
try:
    from adapters import get_adapter
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.adapters import get_adapter
try:
    from core.train import TrainConfig, train_paired
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.train import TrainConfig, train_paired
try:
    from core.model import rollout_original, rollout_ntbn, count_parameters
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.model import rollout_original, rollout_ntbn, count_parameters
try:
    from core.metrics import evaluate_rollout_metrics, compute_paired_comparison
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.core.metrics import evaluate_rollout_metrics, compute_paired_comparison


def main():
    parser = argparse.ArgumentParser(description="Master Deep Hedging vs NTBN benchmark orchestrator.")
    parser.add_argument("--smoke", action="store_true", help="Run in quick smoke test mode (2 epochs, 100 samples)")
    parser.add_argument("--groups", nargs="+", default=None, help="Filter specific groups")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs_to_run = list(CATALOG.values())
    if args.groups:
        configs_to_run = [c for c in configs_to_run if c.group_id in args.groups]

    print("=" * 80)
    print(f"MASTER BENCHMARK ORCHESTRATOR: {len(configs_to_run)} CONFIGURATIONS")
    print(f"Mode: {'SMOKE TEST (2 epochs)' if args.smoke else 'FULL ORIGINAL SCALE'}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    summary_records = []
    t_start_all = time.time()

    for idx, spec in enumerate(configs_to_run):
        print(f"\n[{idx + 1}/{len(configs_to_run)}] Executing {spec.config_id}...")
        adapter = get_adapter(spec.group_id)
        gym_a, gym_b, train_cfg_base = adapter.get_variant_configs(spec.variant)

        epochs = 2 if args.smoke else spec.epochs
        test_samples = 100 if args.smoke else spec.test_samples
        
        train_cfg = TrainConfig(
            epochs=epochs,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            clipvalue=train_cfg_base.clipvalue,
            global_clipnorm=train_cfg_base.global_clipnorm,
            clipnorm=train_cfg_base.clipnorm,
            seed=args.seed
        )

        train_batch, val_batch, test_batch, feat_dims, n_inst = adapter.build_datasets(
            seed=args.seed,
            test_samples=test_samples
        )

        # Train Branch A
        t0 = time.time()
        res_a = train_paired("A", gym_a, train_cfg, train_batch, val_batch, key=args.seed, feature_dims=feat_dims, n_inst=n_inst)
        roll_a = rollout_original(res_a.params, gym_a, test_batch)
        met_a = evaluate_rollout_metrics(roll_a)
        time_a = time.time() - t0

        # Train Branch B (NTBN)
        t0 = time.time()
        res_b = train_paired("B", gym_b, train_cfg, train_batch, val_batch, key=args.seed, feature_dims=feat_dims, n_inst=n_inst)
        roll_b = rollout_ntbn(res_b.params, gym_b, test_batch)
        met_b = evaluate_rollout_metrics(roll_b)
        time_b = time.time() - t0

        paired = compute_paired_comparison(met_a, met_b, roll_a, roll_b)

        rec = {
            "config_id": spec.config_id,
            "group_id": spec.group_id,
            "variant": spec.variant,
            "theoretical_status": spec.theoretical_status,
            "epochs": epochs,
            "train_samples": spec.train_samples,
            "steps": spec.steps,
            "dt": spec.dt,
            "loss_a": met_a["loss"],
            "loss_b": met_b["loss"],
            "delta_loss": paired["delta_loss"],
            "cvar95_a": met_a["cvar95"],
            "cvar95_b": met_b["cvar95"],
            "delta_cvar95": paired["delta_cvar95"],
            "turnover_a": met_a["turnover"],
            "turnover_b": met_b["turnover"],
            "delta_turnover": paired["delta_turnover"],
            "cost_a": met_a["mean_cost"],
            "cost_b": met_b["mean_cost"],
            "delta_cost": paired["delta_mean_cost"],
            "time_a_sec": time_a,
            "time_b_sec": time_b,
            "params_a": count_parameters(res_a.params),
            "params_b": count_parameters(res_b.params),
        }
        summary_records.append(rec)
        print(f"  -> Delta Loss: {paired['delta_loss']:+.6f} | Delta CVaR95: {paired['delta_cvar95']:+.6f} | Delta Cost: {paired['delta_mean_cost']:+.6f}")

    total_time = time.time() - t_start_all
    df = pd.DataFrame(summary_records)
    csv_path = out_dir / f"benchmark_summary_seed{args.seed}.csv"
    json_path = out_dir / f"benchmark_summary_seed{args.seed}.json"
    
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED IN {total_time:.2f}s")
    print(f"Summary saved to: {csv_path} and {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
