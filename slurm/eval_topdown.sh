#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=[partition]
#SBATCH --job-name=bioclip-eval
#SBATCH --time=1:00:00
#SBATCH --mem=400GB

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

module load miniconda3/24.1.2-py310
conda activate bioclip-test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_TYPE="ViT-B-16"
PRETRAINED="[checkpoint-path]"
DATA_ROOT="[dataset-root]"
LABEL_FILE="metadata.csv"
LOG_FILEPATH="./logs_taxonomy_eval"
TEXT_TYPE="asis"
BATCH_SIZE=3078
WORKERS=16
TEXT_BATCH_SIZE=512
MAX_IMAGES_PER_SPECIES=-1
USE_HYPERBOLIC=0
HYPERBOLIC_SIMILARITY="angle"
OUTPUT_CSV=""
TARGET_KINGDOM=""

EXTRA_ARGS=""
if [[ "$USE_HYPERBOLIC" == "1" ]]; then
    EXTRA_ARGS+=" --use-hyperbolic --hyperbolic-similarity $HYPERBOLIC_SIMILARITY"
fi
if [[ -n "$OUTPUT_CSV" ]]; then
    EXTRA_ARGS+=" --output-csv $OUTPUT_CSV"
fi
if [[ -n "$TARGET_KINGDOM" ]]; then
    EXTRA_ARGS+=" --target-kingdom $TARGET_KINGDOM"
fi

python -m src.evaluation.zero_shot_taxonomy_levels_topdown \
        --model $MODEL_TYPE \
        --pretrained $PRETRAINED \
        --data_root $DATA_ROOT \
        --label_filename $LABEL_FILE \
        --logs $LOG_FILEPATH \
        --text_type $TEXT_TYPE \
        --batch-size $BATCH_SIZE \
        --workers $WORKERS \
        --text-batch-size $TEXT_BATCH_SIZE \
        --max-images-per-species $MAX_IMAGES_PER_SPECIES \
        $EXTRA_ARGS
