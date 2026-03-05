#!/bin/bash
# =============================================================================
# T1: 2D Checkerboard (~5 min) then T2: CIFAR-10 FM w/ OT (~24-36 hrs)
# =============================================================================
#SBATCH -J fm_t1_t2
#SBATCH -A MLMI-SN691-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/t1_t2_%j.out
#SBATCH --error=logs/t1_t2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Environment Setup
# =============================================================================
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PROJECT_DIR="/home/sn691/rds/hpc-work/MLMI4/mlmi4"
cd "${PROJECT_DIR}"

# Install uv if not already present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# Sync dependencies (uv will fetch Python 3.13 if needed)
echo "Syncing dependencies..."
uv sync

echo "=============================================="
echo "Flow Matching Experiments: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(uv run python3 --version)"
echo "=============================================="

set -e
mkdir -p logs

# =============================================================================
# T1: 2D Checkerboard (~5 min)
# =============================================================================
echo ""
echo "========== T1: 2D Checkerboard =========="
echo "Start: $(date)"

uv run python3 checkerboard.py --device cuda --out-dir ./runs/checkerboard

echo "T1 done: $(date)"

# =============================================================================
# T2: CIFAR-10 FM w/ OT (~24-36 hrs)
# =============================================================================
echo ""
echo "========== T2: CIFAR-10 FM w/ OT =========="
echo "Start: $(date)"

uv run python3 training.py --config configs/cifar10_ot.yaml \
    --use-wandb \
    --wandb-project mlmi4-flow-matching \
    --wandb-name cifar10-ot \
    --out-dir ./runs/cifar10_ot

echo "T2 done: $(date)"
echo "=============================================="
echo "All jobs complete: $(date)"
