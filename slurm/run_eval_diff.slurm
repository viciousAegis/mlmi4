#!/bin/bash
# =============================================================================
# Post-training evaluations for CIFAR-10 FM w/ Diffusion
# =============================================================================
#SBATCH -J fm_eval_diff
#SBATCH -A MLMI-SN691-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_diff_%j.out
#SBATCH --error=logs/eval_diff_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PROJECT_DIR="/home/$USER/rds/hpc-work/mlmi4"
cd "${PROJECT_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# !! UPDATE THIS to the best checkpoint from training (check WandB for best FID) !!
CKPT="runs/cifar10_diffusion/ckpt_step125000.pt"

echo "=============================================="
echo "Evaluating: ${CKPT}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================================="

set -e
mkdir -p runs/eval_diff

# ---- E1: FID with dopri5 (Table 1) ----
echo ""
echo "========== E1: FID (dopri5, 50k samples) =========="
echo "Start: $(date)"
uv run python3 evaluate_fid.py \
    --ckpt "${CKPT}" --dataset cifar10 --n 50000 \
    --solver dopri5 --batch-size 256 \
    --out-dir runs/eval_diff/fid_dopri5
echo "E1 done: $(date)"

# ---- E4: ODE error vs NFE (Figure 7 left) ----
echo ""
echo "========== E4: ODE error vs NFE =========="
echo "Start: $(date)"
uv run python3 evaluate_low_cost.py \
    --ckpt "${CKPT}" --mode error --dataset cifar10 \
    --n-seeds 256 --out-dir runs/eval_diff
echo "E4 done: $(date)"

# ---- E5: FID vs NFE (Figure 7 right) ----
echo ""
echo "========== E5: FID vs NFE =========="
echo "Start: $(date)"
uv run python3 evaluate_low_cost.py \
    --ckpt "${CKPT}" --mode fid --dataset cifar10 \
    --n-samples 50000 --batch-size 256 \
    --nfe-list 10 20 40 60 80 100 \
    --out-dir runs/eval_diff
echo "E5 done: $(date)"

# ---- E6: Sample grid ----
echo ""
echo "========== E6: Sample grid =========="
uv run python3 sample.py \
    --ckpt "${CKPT}" --n 64 --output runs/eval_diff/samples.png

# ---- E7: Trajectory visualisation ----
echo ""
echo "========== E7: Trajectory visualisation =========="
uv run python3 sample.py \
    --ckpt "${CKPT}" --n 8 --show-trajectory --num-steps 10 \
    --output runs/eval_diff/trajectory.png

echo ""
echo "=============================================="
echo "All evaluations complete: $(date)"
echo "=============================================="
