#!/bin/bash
#SBATCH --job-name=r2_cost10_n24
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/slurm_cost10_n24_%j.log
#SBATCH --error=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory/logs/slurm_cost10_n24_%j.log

# LOCKED CONFIRMATORY EXPERIMENT (10 bps COST):
#   N=24 seeds [5001..5024], cost_s=0.0010 in training/val/test, 48 runs (A_DECAY & R2_DECAY)
#   Strictly forbidden to alter N, stop early, or alter locked hyperparameters.

set -euo pipefail

N_SEEDS=${N_SEEDS:-24}
SEED_OFFSET=${SEED_OFFSET:-0}

REPO_ROOT="/home/user04/deephedging_jax_training_optimization"
COST_CONF_DIR="${REPO_ROOT}/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_cost_confirmatory"
VENV_PY="${REPO_ROOT}/.venv/bin/python3"

mkdir -p "${COST_CONF_DIR}/logs" "${COST_CONF_DIR}/results" "${COST_CONF_DIR}/checkpoints"

export TMPDIR=/home/user04/tmp
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${COST_CONF_DIR}"

echo "=== NODE & ENV CHECK AT $(date) ==="
hostname
ls -la /home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR
"${VENV_PY}" -c "import jax; print('JAX:', jax.__version__, 'Devices:', jax.devices())"

echo "=== LOCKED PRE-RUN SHA-256 VERIFICATION ==="
sha256sum -c SOURCE_SHA256SUMS.txt

echo "=== FINAL PRE-RUN UNIT TESTS & DRY RUN ==="
"${VENV_PY}" -u tests/test_cost_conf_suite.py
"${VENV_PY}" -u cost_run.py --mode dry_run

echo "=== EXECUTING LOCKED 10 BPS COST CONFIRMATORY RUN (N_SEEDS=${N_SEEDS}, SEED_OFFSET=${SEED_OFFSET}) AT $(date) ==="
"${VENV_PY}" -u cost_run.py --mode execute_confirmatory --confirm-locked-run --n-seeds "${N_SEEDS}" --seed-offset "${SEED_OFFSET}"
echo "=== COMPLETED LOCKED 10 BPS COST CONFIRMATORY RUN AT $(date) ==="
