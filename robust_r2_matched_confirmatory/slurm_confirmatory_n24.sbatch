#!/bin/bash
#SBATCH --job-name=r2_conf_n24
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_n24_%j.log
#SBATCH --error=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_n24_%j.log

# LOCKED CONFIRMATORY EXPERIMENT: N=24 seeds [4001..4024], cost_s=0.0, 48 runs (A_DECAY & R2_DECAY)
# Approved N=24 fixed; strictly forbidden to alter N, stop early, or alter locked hyperparameters.

set -euo pipefail

N_SEEDS=${N_SEEDS:-24}
SEED_OFFSET=${SEED_OFFSET:-0}

REPO_ROOT="/home/user04/deephedging_jax_training_optimization"
CONF_DIR="${REPO_ROOT}/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory"
VENV_PY="${REPO_ROOT}/.venv/bin/python3"

mkdir -p "${CONF_DIR}/logs" "${CONF_DIR}/results" "${CONF_DIR}/checkpoints"

export TMPDIR=/home/user04/tmp
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${CONF_DIR}"

echo "=== NODE & ENV CHECK AT $(date) ==="
hostname
ls -la /home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/PAUSE_COORDINATOR
"${VENV_PY}" -c "import jax; print('JAX:', jax.__version__, 'Devices:', jax.devices())"

echo "=== LOCKED PRE-RUN SHA-256 VERIFICATION ==="
sha256sum \
  R2_MATCHED_CONFIRMATORY_DESIGN_LOCK.md SAMPLE_SIZE_SENSITIVITY.csv DATA_AND_SEED_PLAN.json \
  conf_keys.py conf_design.py conf_stats.py conf_run.py \
  scripts/sample_size_and_power.py tests/test_conf_suite.py \
  ../robust_r3_lr_stability/lr_train.py ../robust_r3_lr_stability/lr_schedule.py ../robust_r3_lr_stability/lr_keys.py \
  ../robust_r3_budget_control/bc_models.py \
  ../robust_r3_dev/r3_models.py ../robust_r3_dev/r3_metrics.py ../robust_r3_dev/r3_train.py ../robust_r3_dev/r3_keys.py

echo "=== FINAL PRE-RUN UNIT TESTS & DRY RUN ==="
"${VENV_PY}" -u tests/test_conf_suite.py
"${VENV_PY}" -u conf_run.py --mode dry_run

echo "=== EXECUTING LOCKED CONFIRMATORY RUN (N_SEEDS=${N_SEEDS}, SEED_OFFSET=${SEED_OFFSET}) AT $(date) ==="
"${VENV_PY}" -u conf_run.py --mode execute_confirmatory --confirm-locked-run --n-seeds "${N_SEEDS}" --seed-offset "${SEED_OFFSET}"
echo "=== COMPLETED LOCKED CONFIRMATORY RUN AT $(date) ==="
