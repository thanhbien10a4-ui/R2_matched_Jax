#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/gpu_coordinator.py: Autonomous GPU Session Coordinator with Auto-Resume from Checkpoint.

Features:
- Coordinates execution across all 24 configurations (or selected subset)
- Enforces session budgets / handles GPU time limits and preemption
- When GPU reaches session limits (exit code 10) or crashes, automatically creates a new session
  that seamlessly continues from the exact latest saved checkpoint
- Tracks progress in coordinator_state.json
- Compiles consolidated summary upon 100% completion
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent.parent

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from configs.catalog import CATALOG, ConfigSpec, get_config_spec
except ImportError:
    from experiments.jax_ntbn_7notebooks_original_scale.configs.catalog import CATALOG, ConfigSpec, get_config_spec


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class GPUCoordinator:
    def __init__(
        self,
        checkpoint_dir: Path,
        results_dir: Path,
        seed: int = 42,
        branch: str = "both",
        session_limit_seconds: float = None,
        max_restarts_per_config: int = 30,
        smoke: bool = False
    ):
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.seed = seed
        self.branch = branch
        self.session_limit_seconds = session_limit_seconds
        self.max_restarts = max_restarts_per_config
        self.smoke = smoke

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / f"coordinator_state_seed{self.seed}.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log(f"Warning: Failed to load state ({e}). Initializing fresh state.")
        return {
            "seed": self.seed,
            "branch": self.branch,
            "smoke": self.smoke,
            "configs": {}
        }

    def _save_state(self):
        tmp_file = self.state_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        tmp_file.replace(self.state_file)

    def is_completed(self, config_id: str) -> bool:
        res_file = self.results_dir / f"result_{config_id}_seed{self.seed}.json"
        if not res_file.exists():
            return False
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            branches = data.get("branches", {})
            if self.branch in ("A", "both") and "A" not in branches:
                return False
            if self.branch in ("B", "both") and "B" not in branches:
                return False
            return True
        except Exception:
            return False

    def run_config(self, spec: ConfigSpec) -> bool:
        cid = spec.config_id
        if self.is_completed(cid):
            log(f"Config '{cid}' is already completed. Skipping.")
            self.state["configs"][cid] = {
                "status": "COMPLETED",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self._save_state()
            return True

        config_chk_dir = self.checkpoint_dir / cid
        config_chk_dir.mkdir(parents=True, exist_ok=True)

        restart_count = self.state["configs"].get(cid, {}).get("restart_count", 0)
        self.state["configs"][cid] = {
            "status": "IN_PROGRESS",
            "restart_count": restart_count,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_state()

        runner_script = SCRIPT_DIR / "run_single_config.py"

        while restart_count < self.max_restarts:
            session_num = restart_count + 1
            log("=" * 70)
            log(f"🚀 [SESSION #{session_num}] CONFIG: {cid} (Seed {self.seed}, Branch: {self.branch})")
            if self.session_limit_seconds:
                log(f"⏱️ Session GPU Time Limit: {self.session_limit_seconds}s")
            log("=" * 70)

            cmd = [
                sys.executable,
                str(runner_script),
                "--config_id", cid,
                "--branch", self.branch,
                "--seed", str(self.seed),
                "--checkpoint_dir", str(config_chk_dir),
                "--output_dir", str(self.results_dir)
            ]
            if self.session_limit_seconds is not None:
                cmd.extend(["--max_session_seconds", str(self.session_limit_seconds)])
            if self.smoke:
                cmd.extend(["--epochs", "2"])

            t_session_start = time.time()
            proc = subprocess.run(cmd)
            session_duration = time.time() - t_session_start

            # Check completion
            if proc.returncode == 0 and self.is_completed(cid):
                log(f"✅ CONFIG {cid} COMPLETED SUCCESSFULLY in session #{session_num} ({session_duration:.1f}s)!")
                self.state["configs"][cid] = {
                    "status": "COMPLETED",
                    "restart_count": restart_count,
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self._save_state()
                return True
            elif proc.returncode == 10:
                # GPU limit reached cleanly, checkpoint saved
                restart_count += 1
                log(f"⏸️ [GPU LIMIT REACHED] Session #{session_num} saved checkpoint at limit. Spawning next session continuation #{restart_count + 1}...")
                self.state["configs"][cid]["restart_count"] = restart_count
                self.state["configs"][cid]["last_limit_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._save_state()
                time.sleep(2)
            else:
                # Unexpected crash or OOM
                restart_count += 1
                log(f"⚠️ [INTERRUPTED] Session #{session_num} exited with code {proc.returncode}. Re-spawning continuation from checkpoint in 5s...")
                self.state["configs"][cid]["restart_count"] = restart_count
                self._save_state()
                time.sleep(5)

        log(f"❌ CONFIG {cid} reached maximum retries ({self.max_restarts}). Marked as FAILED.")
        self.state["configs"][cid]["status"] = "FAILED"
        self._save_state()
        return False

    def run_all(self, target_configs: List[ConfigSpec]):
        total = len(target_configs)
        log("*" * 80)
        log(f"🌟 AUTONOMOUS GPU COORDINATOR STARTING: {total} CONFIGURATIONS")
        log(f"Target Seed: {self.seed} | Branch: {self.branch} | Mode: {'SMOKE' if self.smoke else 'FULL SCALE'}")
        log(f"Checkpoint Directory: {self.checkpoint_dir}")
        log(f"Results Directory: {self.results_dir}")
        log("*" * 80)

        success_count = 0
        for i, spec in enumerate(target_configs, 1):
            log(f"\n>>> Progress: [{i}/{total}] -> Processing {spec.config_id}...")
            ok = self.run_config(spec)
            if ok:
                success_count += 1

        log("\n" + "=" * 80)
        log(f"🏁 BENCHMARK COORDINATION FINISHED: {success_count}/{total} CONFIGURATIONS COMPLETED.")
        log("=" * 80)

        # Compile consolidated summary
        self.compile_summary(target_configs)

    def compile_summary(self, target_configs: List[ConfigSpec]):
        summary_records = []
        for spec in target_configs:
            res_file = self.results_dir / f"result_{spec.config_id}_seed{self.seed}.json"
            if res_file.exists():
                try:
                    with open(res_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    rec = {
                        "config_id": spec.config_id,
                        "group_id": spec.group_id,
                        "variant": spec.variant,
                        "theoretical_status": spec.theoretical_status,
                        "epochs": data.get("epochs", spec.epochs),
                    }
                    branches = data.get("branches", {})
                    if "A" in branches:
                        rec["loss_A"] = branches["A"]["test_metrics"]["loss"]
                        rec["cvar95_A"] = branches["A"]["test_metrics"]["cvar95"]
                        rec["turnover_A"] = branches["A"]["test_metrics"]["turnover"]
                        rec["time_A"] = branches["A"]["train_time_sec"]
                    if "B" in branches:
                        rec["loss_B"] = branches["B"]["test_metrics"]["loss"]
                        rec["cvar95_B"] = branches["B"]["test_metrics"]["cvar95"]
                        rec["turnover_B"] = branches["B"]["test_metrics"]["turnover"]
                        rec["time_B"] = branches["B"]["train_time_sec"]
                    if "paired_comparison" in data:
                        rec["delta_loss"] = data["paired_comparison"]["delta_loss"]
                        rec["delta_cvar95"] = data["paired_comparison"]["delta_cvar95"]
                    summary_records.append(rec)
                except Exception as e:
                    log(f"Error reading {res_file}: {e}")

        summary_path = self.results_dir / f"benchmark_summary_seed{self.seed}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=2)
        log(f"Saved consolidated summary ({len(summary_records)} configs) to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous GPU Session Coordinator with Auto-Resume.")
    parser.add_argument("--config_id", type=str, default=None, help="Run single config")
    parser.add_argument("--groups", nargs="+", default=None, help="Filter specific groups")
    parser.add_argument("--branch", type=str, default="both", choices=["A", "B", "both"], help="Branch to run")
    parser.add_argument("--seed", type=int, default=42, help="Seed to run")
    parser.add_argument("--smoke", action="store_true", help="Run 2 epochs smoke mode")
    parser.add_argument("--session_limit_seconds", type=float, default=None, help="Session limit in seconds (auto-respawn when reached)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint root directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Results root directory")
    parser.add_argument("--max_restarts", type=int, default=30, help="Max restarts per config")
    args = parser.parse_args()

    chk_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else PACKAGE_ROOT / "checkpoints"
    res_dir = Path(args.results_dir) if args.results_dir else PACKAGE_ROOT / "results"

    coordinator = GPUCoordinator(
        checkpoint_dir=chk_dir,
        results_dir=res_dir,
        seed=args.seed,
        branch=args.branch,
        session_limit_seconds=args.session_limit_seconds,
        max_restarts_per_config=args.max_restarts,
        smoke=args.smoke
    )

    if args.config_id:
        targets = [get_config_spec(args.config_id)]
    else:
        targets = list(CATALOG.values())
        if args.groups:
            targets = [c for c in targets if c.group_id in args.groups]

    coordinator.run_all(targets)


if __name__ == "__main__":
    main()
