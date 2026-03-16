#!/bin/bash
#SBATCH --job-name=text-dfm-small
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

set -euo pipefail

# Optional overrides at submit time:
# sbatch --export=ALL,CONFIG=config_text_example.yaml,OUT_DIR=/path/to/out,WANDB_PROJECT=mlmi4-text-dfm slurm/train_text_small_dfm.slurm

CONFIG="${CONFIG:-config_text_example.yaml}"
OUT_DIR="${OUT_DIR:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-text-dfm-small-${SLURM_JOB_ID}}"
WANDB_TAGS="${WANDB_TAGS:-}"

mkdir -p logs

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "Using WANDB_API_KEY from environment"
fi

CMD=(
  uv run python training_text.py
  --config "$CONFIG"
  --wandb-name "$WANDB_RUN_NAME"
)

if [[ -n "$OUT_DIR" ]]; then
  CMD+=(--out-dir "$OUT_DIR")
fi

if [[ -n "$WANDB_PROJECT" ]]; then
  CMD+=(--wandb-project "$WANDB_PROJECT")
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ -n "$WANDB_TAGS" ]]; then
  read -r -a TAG_ARRAY <<< "$WANDB_TAGS"
  CMD+=(--wandb-tags "${TAG_ARRAY[@]}")
fi

echo "Running command: ${CMD[*]}"
"${CMD[@]}"
