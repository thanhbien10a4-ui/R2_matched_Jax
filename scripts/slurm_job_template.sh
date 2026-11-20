#!/bin/bash
#SBATCH --job-name=jax_ntbn_7groups
#SBATCH --output=slurm_jax_ntbn_7groups_%j.out
#SBATCH --error=slurm_jax_ntbn_7groups_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

set -euo pipefail

echo "================================================================================"
echo "SLURM JOB ${SLURM_JOB_ID}: JAX + NTBN 7 GROUPS BENCHMARK (ORIGINAL SCALE)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "================================================================================"

# 1. Environment Activation
VENV_PATH="/home/user04/deephedging_jax_training_optimization/.venv"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated virtual environment: ${VENV_PATH}"
else
    echo "Warning: Virtual environment not found at ${VENV_PATH}, using system python"
fi

# 2. Hardware Preflight Check
echo "=== NVIDIA GPU HARDWARE CHECK ==="
nvidia-smi

echo "=== JAX DEVICE CHECK ==="
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Default backend:', jax.default_backend())
print('Devices:', jax.devices())
assert jax.default_backend() == 'gpu', 'ERROR: JAX is not using GPU backend!'
"

# 3. Path Configuration
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/deephedging-jax:${PYTHONPATH:-}"

echo "PYTHONPATH: ${PYTHONPATH}"

# 4. Start VRAM Background Monitor
TRACE_FILE="gpu_memory_trace_${SLURM_JOB_ID}.csv"
echo "timestamp,utilization_gpu_pct,memory_used_mb,memory_total_mb" > "${TRACE_FILE}"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "${TRACE_FILE}" 2>/dev/null || true
        sleep 5
    done
) &
MONITOR_PID=$!
trap "kill -9 $MONITOR_PID 2>/dev/null || true" EXIT

# 5. Execute Master Benchmark
echo "=== EXECUTING BENCHMARK ACROSS 7 GROUPS ==="
python3 experiments/jax_ntbn_7notebooks_original_scale/scripts/run_benchmark.py \
    --seed 42 \
    --output_dir "experiments/jax_ntbn_7notebooks_original_scale/results_job_${SLURM_JOB_ID}"

echo "================================================================================"
echo "BENCHMARK COMPLETED SUCCESSFULLY: $(date)"
echo "================================================================================"
