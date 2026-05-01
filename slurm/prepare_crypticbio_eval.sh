#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=[partition]
#SBATCH --job-name=crypticbio-prepare
#SBATCH --time=8:00:00
#SBATCH --mem=64GB

set -euo pipefail

module load miniconda3/24.1.2-py310
conda activate bioclip-test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ID="gmanolache/CrypticBio"
OUT_DIR="[output-dir]/CrypticBio_eval"
CACHE_DIR=""
HF_TOKEN=""
IMAGE_COLUMN=""
LABEL_COLUMN=""
MAX_PER_SPLIT=-1
JPG_QUALITY=95
DOWNLOAD_TIMEOUT=30

# Optional: set specific splits, for example:
# SPLITS=("train" "validation" "test")
SPLITS=()

CMD=(
    python scripts/prepare_crypticbio_eval.py
    --dataset-id "${DATASET_ID}"
    --out-dir "${OUT_DIR}"
    --max-per-split "${MAX_PER_SPLIT}"
    --jpg-quality "${JPG_QUALITY}"
    --download-timeout "${DOWNLOAD_TIMEOUT}"
)

if [[ -n "${CACHE_DIR}" ]]; then
    CMD+=(--cache-dir "${CACHE_DIR}")
fi

if [[ -n "${HF_TOKEN}" ]]; then
    CMD+=(--token "${HF_TOKEN}")
fi

if [[ -n "${IMAGE_COLUMN}" ]]; then
    CMD+=(--image-column "${IMAGE_COLUMN}")
fi

if [[ -n "${LABEL_COLUMN}" ]]; then
    CMD+=(--label-column "${LABEL_COLUMN}")
fi

if ((${#SPLITS[@]} > 0)); then
    CMD+=(--splits "${SPLITS[@]}")
fi

echo "[INFO] ${CMD[*]}"
"${CMD[@]}"
