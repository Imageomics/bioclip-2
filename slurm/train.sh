#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=[partition]
#SBATCH --job-name=bioclip-train
#SBATCH --time=38:00:00
#SBATCH --mem=800GB

set -euo pipefail

module load miniconda3/24.1.2-py310
conda activate bioclip-train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_DATA='/path/to/train/shards/xxx.tar'
VAL_DATA='/path/to/val/shards/xxx.tar'
PRETRAINED='[checkpoint-path]'
LOG_DIR='./logs'

MODEL_TYPE='ViT-B-16'
BATCH_SIZE=4096
WORKERS=8
EPOCHS=80
LR=1e-4
WARMUP=300
SEED=42

USE_HYPERBOLIC=0
HYPERBOLIC_SIMILARITY='dist'
HYPERBOLIC_CURV_INIT=1
HYPERBOLIC_LEARN_CURV=0

CMD=(
  torchrun --nproc_per_node 4
  -m src.training.main
  --train-data "${TRAIN_DATA}"
  --val-data "${VAL_DATA}"
  --dataset-type webdataset
  --pretrained "${PRETRAINED}"
  --text_type 'taxon'
  --dataset-resampled
  --warmup "${WARMUP}"
  --batch-size "${BATCH_SIZE}"
  --accum-freq 1
  --epochs "${EPOCHS}"
  --workers "${WORKERS}"
  --save-frequency 10
  --model "${MODEL_TYPE}"
  --log-every-n-steps 1
  --lr "${LR}"
  --seed "${SEED}"
  --local-loss
  --gather-with-grad
  --grad-checkpointing
  --logs "${LOG_DIR}"
  --taxonomy-all-levels
  --val-frequency 0
  --no-weights-only
  --torchcompile
)

if [[ "${USE_HYPERBOLIC}" == "1" ]]; then
  CMD+=(
    --use-hyperbolic
    --hyperbolic-load-nonstrict
    --hyperbolic-curv-init "${HYPERBOLIC_CURV_INIT}"
    --hyperbolic-similarity "${HYPERBOLIC_SIMILARITY}"
  )
  if [[ "${HYPERBOLIC_LEARN_CURV}" == "0" ]]; then
    CMD+=(--no-hyperbolic-learn-curv)
  fi
fi

echo "[INFO] ${CMD[*]}"
"${CMD[@]}"
