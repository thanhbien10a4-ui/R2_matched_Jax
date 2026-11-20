#!/bin/bash
set -euo pipefail
cd /home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory
PROMPT_TEXT="$(cat logs/gng5_pre_prompt.txt)"
/home/user04/.local/bin/agy -p "$PROMPT_TEXT" --model gemini-4-argon-high --mode auto --dangerously-skip-permissions --output-format json --print-timeout 480s > logs/gng5_pre_audit_raw.json
ls -lh logs/gng5_pre_audit_raw.json
