#!/bin/bash
#SBATCH --job-name=jax_ntbn_coord
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/slurm_logs/slurm_coord_%j.log
#SBATCH --error=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/slurm_logs/slurm_coord_%j.log

set -euo pipefail

# ==============================================================================
# GNG5 AUTONOMOUS GPU COORDINATOR FOR JAX NTBN 7 NOTEBOOK GROUPS
# Features:
# - Runs 24 configurations across Branch A and Branch B
# - Enforces atomic checkpointing after each milestone/epoch
# - If session reaches GPU time limit (e.g. 2h sessions) or gets interrupted,
#   automatically resumes from the latest checkpoint without lost work
# ==============================================================================

export TMPDIR=/home/user04/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

BASE_DIR="/home/user04/deephedging_jax_training_optimization"
EXP_DIR="$BASE_DIR/experiments/jax_ntbn_7notebooks_original_scale"
PYTHON="$BASE_DIR/.venv/bin/python3"

export PYTHONPATH="$EXP_DIR:$BASE_DIR:${PYTHONPATH:-}"

mkdir -p "$EXP_DIR/slurm_logs"
mkdir -p "$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/results"
cd "$EXP_DIR"

echo "================================================================="
echo "GNG5 GPU COORDINATOR STARTED"
echo "SLURM JOB ID: ${SLURM_JOB_ID:-manual}"
echo "RUNNING ON NODE: $(hostname)"
echo "START TIME: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================================================="

echo "--- 1. HARDWARE VERIFICATION ---"
nvidia-smi

$PYTHON -c "
import sys, jax
print('Python:', sys.version.split()[0])
print('JAX version:', jax.__version__)
print('JAX backend:', jax.default_backend())
devices = jax.devices()
print('JAX devices:', devices)
if not devices or devices[0].platform != 'gpu':
    print('FATAL ERROR: JAX did not detect GPU!')
    sys.exit(1)
print('Device kind:', devices[0].device_kind)
"

echo "--- 2. LAUNCHING AUTONOMOUS GPU COORDINATOR ---"
# Each config gets up to 7200s (2 hours) per session budget before safe respawn
$PYTHON -u scripts/gpu_coordinator.py \
    --seed 42 \
    --branch both \
    --session_limit_seconds 7200 \
    --checkpoint_dir "$EXP_DIR/checkpoints" \
    --results_dir "$EXP_DIR/results"

echo "================================================================="
echo "COORDINATOR JOB FINISHED AT $(date)"
echo "================================================================="
