#!/bin/bash
# =============================================================================
# Post-training evaluations for CIFAR-10 FM w/ OT
# E1: FID (dopri5), E4: ODE error vs NFE, E5: FID vs NFE, E6/E7: samples
# =============================================================================
#SBATCH -J fm_eval_ot
#SBATCH -A MLMI-SN691-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_ot_%j.out
#SBATCH --error=logs/eval_ot_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

PROJECT_DIR="/home/sn691/rds/hpc-work/MLMI4/mlmi4"
cd "${PROJECT_DIR}"
export PATH="$HOME/.local/bin:$PATH"

# Best checkpoint (FID=5.95 at step 125k)
CKPT="runs/cifar10_ot/ckpt_step125000.pt"
echo "=============================================="
echo "Evaluating: ${CKPT}"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================================="

set -e
mkdir -p runs/eval_ot

# ---- E1: FID with dopri5 (Table 1) ----
echo ""
echo "========== E1: FID (dopri5, 50k samples) =========="
echo "Start: $(date)"
uv run python3 evaluate_fid.py \
    --ckpt "${CKPT}" --dataset cifar10 --n 50000 \
    --solver dopri5 --batch-size 256 \
    --out-dir runs/eval_ot/fid_dopri5
echo "E1 done: $(date)"

# ---- E4: ODE error vs NFE (Figure 7 left) ----
echo ""
echo "========== E4: ODE error vs NFE =========="
echo "Start: $(date)"
uv run python3 evaluate_low_cost.py \
    --ckpt "${CKPT}" --mode error --dataset cifar10 \
    --n-seeds 256 --out-dir runs/eval_ot
echo "E4 done: $(date)"

# ---- E5: FID vs NFE (Figure 7 right) ----
echo ""
echo "========== E5: FID vs NFE =========="
echo "Start: $(date)"
uv run python3 evaluate_low_cost.py \
    --ckpt "${CKPT}" --mode fid --dataset cifar10 \
    --n-samples 50000 --batch-size 256 \
    --nfe-list 10 20 40 60 80 100 \
    --out-dir runs/eval_ot
echo "E5 done: $(date)"

# ---- E6: Sample grid ----
echo ""
echo "========== E6: Sample grid =========="
uv run python3 sample.py \
    --ckpt "${CKPT}" --n 64 --output runs/eval_ot/samples.png

# ---- E7: Trajectory visualisation ----
echo ""
echo "========== E7: Trajectory visualisation =========="
uv run python3 sample.py \
    --ckpt "${CKPT}" --n 8 --show-trajectory --num-steps 10 \
    --output runs/eval_ot/trajectory.png

echo ""
echo "=============================================="
echo "All evaluations complete: $(date)"
echo "=============================================="
