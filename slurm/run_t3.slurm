#!/bin/bash
# =============================================================================
# T3: CIFAR-10 FM w/ Diffusion
# =============================================================================
#SBATCH -J fm_t3_diff
#SBATCH -A MLMI-SN691-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/t3_%j.out
#SBATCH --error=logs/t3_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PROJECT_DIR="/home/sn691/rds/hpc-work/MLMI4/mlmi4"
cd "${PROJECT_DIR}"
export PATH="$HOME/.local/bin:$PATH"

echo "=============================================="
echo "T3: CIFAR-10 FM w/ Diffusion: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================================="

set -e
mkdir -p logs

uv run python3 training.py --config configs/cifar10_diffusion.yaml \
    --resume \
    --use-wandb \
    --wandb-project mlmi4-flow-matching \
    --wandb-name cifar10-diffusion \
    --out-dir ./runs/cifar10_diffusion

echo "T3 done: $(date)"
