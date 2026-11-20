#!/bin/bash
set -euo pipefail
cd /home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory
PROMPT="$(cat logs/gng5_postrun_prompt.txt)"
/home/user04/.local/bin/agy -p "$PROMPT" \
  --add-dir /home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale \
  --model gemini-4-argon-high \
  --output-format json \
  --dangerously-skip-permissions \
  --print-timeout 480s \
  > logs/gng5_postrun_audit_raw.json 2> logs/gng5_postrun_audit.err
wc -c logs/gng5_postrun_audit_raw.json
