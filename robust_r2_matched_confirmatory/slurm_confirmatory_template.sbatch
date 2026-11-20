#!/bin/bash
#SBATCH --job-name=r2_conf_24
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_%j.out
#SBATCH --error=/home/user04/deephedging_jax_training_optimization/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory/logs/slurm_conf_%j.err

# NOTE: THIS TEMPLATE IS UNSUBMITTED BY DESIGN DURING PHASE-1 DESIGN PREPARATION.
# BEFORE SUBMITTING LATER:
#   1. Lock N_SEEDS (recommended: 24; fallback option: 16 with --time=03:30:00).
#   2. Record SHA-256 of frozen scripts BEFORE sbatch.
#   3. Verify PAUSE_COORDINATOR and cluster queue status.

set -euo pipefail

N_SEEDS=${N_SEEDS:-24}
SEED_OFFSET=${SEED_OFFSET:-0}

REPO_ROOT="/home/user04/deephedging_jax_training_optimization"
CONF_DIR="${REPO_ROOT}/experiments/jax_ntbn_7notebooks_original_scale/robust_r2_matched_confirmatory"
VENV_PY="${REPO_ROOT}/.venv/bin/python3"

if [[ ! -x "${VENV_PY}" ]]; then
    VENV_PY="$(command -v python3)"
fi

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

echo "=== PRE-RUN SOURCE INTEGRITY (SHA-256 LOCAL + UPSTREAM) ==="
sha256sum \
  conf_keys.py conf_design.py conf_stats.py conf_run.py \
  scripts/sample_size_and_power.py tests/test_conf_suite.py DATA_AND_SEED_PLAN.json R2_MATCHED_CONFIRMATORY_DESIGN_DRAFT.md \
  ../robust_r3_lr_stability/lr_train.py ../robust_r3_lr_stability/lr_schedule.py ../robust_r3_lr_stability/lr_keys.py \
  ../robust_r3_budget_control/bc_models.py \
  ../robust_r3_dev/r3_models.py ../robust_r3_dev/r3_metrics.py ../robust_r3_dev/r3_train.py ../robust_r3_dev/r3_keys.py

echo "=== RUNNING UNIT & DRY-RUN AUDIT ==="
"${VENV_PY}" -u tests/test_conf_suite.py
"${VENV_PY}" -u conf_run.py --mode dry_run

echo "=== LAUNCHING CONFIRMATORY RUN (N_SEEDS=${N_SEEDS}, SEED_OFFSET=${SEED_OFFSET}) ==="
"${VENV_PY}" -u conf_run.py --mode execute_confirmatory --confirm-locked-run --n-seeds "${N_SEEDS}" --seed-offset "${SEED_OFFSET}"
