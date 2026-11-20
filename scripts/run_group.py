"""
scripts/run_group.py: Runs all configurations in a given notebook group.

Usage:
  python run_group.py --group_id trainer --branch both --epochs 10 --seed 42
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from configs.catalog import list_configs_by_group, CATALOG
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.configs.catalog import list_configs_by_group, CATALOG


def main():
    parser = argparse.ArgumentParser(description="Run all configurations in a notebook group.")
    parser.add_argument("--group_id", type=str, required=True, help="Notebook group ID")
    parser.add_argument("--branch", type=str, default="both", choices=["A", "B", "both"], help="Branch to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    args = parser.parse_args()

    specs = list_configs_by_group(args.group_id)
    if not specs:
        raise ValueError(f"No configurations found for group_id '{args.group_id}'")

    print("=" * 80)
    print(f"RUNNING ENTIRE GROUP: {args.group_id} ({len(specs)} configurations)")
    print("=" * 80)

    runner_script = Path(__file__).resolve().parent / "run_single_config.py"

    for idx, spec in enumerate(specs):
        print(f"\n[{idx + 1}/{len(specs)}] Launching config: {spec.config_id}...")
        cmd = [
            sys.executable, str(runner_script),
            "--config_id", spec.config_id,
            "--branch", args.branch,
            "--seed", str(args.seed)
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.output_dir is not None:
            cmd.extend(["--output_dir", args.output_dir])

        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"ERROR: Config {spec.config_id} exited with code {res.returncode}")
        else:
            print(f"SUCCESS: Config {spec.config_id} completed.")


if __name__ == "__main__":
    main()
