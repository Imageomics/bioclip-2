#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=[partition]
#SBATCH --job-name=species-tsne
#SBATCH --time=00:20:00
#SBATCH --mem=120GB

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

module load miniconda3/24.1.2-py310
conda activate bioclip-test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_TYPE="ViT-B-16"
PRETRAINED="[checkpoint-path]"
METADATA_PATH="[dataset-root]/metadata.csv"
SPECIES_BINOMIAL=""
TARGET_KINGDOM=""
LOG_FILEPATH="./plots_species_context"
TEXT_BATCH_SIZE=256
MAX_SIBLINGS_PER_LEVEL=-1
TSNE_PERPLEXITY=10
TSNE_N_ITER=10000
SEED_START=0
SEED_END=2
USE_HYPERBOLIC=0
HYPERBOLIC_SIMILARITY="inner"
PANEL_TITLE_FONTSIZE=20

# Optional side-by-side comparison.
# Single-checkpoint run:
#   Leave all four COMPARE_* arrays empty, and the script will use PRETRAINED only.
# Side-by-side comparison:
#   Fill COMPARE_PRETRAINED with 2+ checkpoints.
#   Then fill COMPARE_NAMES / COMPARE_MODES / COMPARE_HYPERBOLIC_SIMILARITIES
#   in the same order, one entry per checkpoint.
# Example:
#   COMPARE_PRETRAINED=("/path/to/baseline.pt" "/path/to/ours.pt")
#   COMPARE_NAMES=("BioCLIP" "Ours")
#   COMPARE_MODES=("euclidean" "hyperbolic")
#   COMPARE_HYPERBOLIC_SIMILARITIES=("inner" "inner")
# Notes:
#   1. COMPARE_MODES entries must be "euclidean" or "hyperbolic".
#   2. If one checkpoint is euclidean, its COMPARE_HYPERBOLIC_SIMILARITIES
#      entry is ignored, but keep a placeholder so the lengths still match.
#   3. When COMPARE_PRETRAINED is non-empty, the figure is produced as
#      side-by-side panels, one panel per checkpoint.
COMPARE_PRETRAINED=()
COMPARE_NAMES=()
COMPARE_MODES=()
COMPARE_HYPERBOLIC_SIMILARITIES=()

for SEED in $(seq "${SEED_START}" "${SEED_END}"); do
    SEED_LOG_DIR="${LOG_FILEPATH}/seed_${SEED}"
    CMD=(
        python scripts/visualize_species_ancestors_siblings_tsne.py
        --model "${MODEL_TYPE}"
        --pretrained "${PRETRAINED}"
        --metadata-path "${METADATA_PATH}"
        --species-binomial "${SPECIES_BINOMIAL}"
        --text-batch-size "${TEXT_BATCH_SIZE}"
        --max-siblings-per-level "${MAX_SIBLINGS_PER_LEVEL}"
        --tsne-perplexity "${TSNE_PERPLEXITY}"
        --tsne-n-iter "${TSNE_N_ITER}"
        --seed "${SEED}"
        --panel-title-fontsize "${PANEL_TITLE_FONTSIZE}"
        --logs "${SEED_LOG_DIR}"
    )

    if [[ -n "${TARGET_KINGDOM}" ]]; then
        CMD+=(--target-kingdom "${TARGET_KINGDOM}")
    fi

    if [[ "${USE_HYPERBOLIC}" == "1" ]]; then
        CMD+=(--use-hyperbolic --hyperbolic-similarity "${HYPERBOLIC_SIMILARITY}")
    fi

    if ((${#COMPARE_PRETRAINED[@]} > 0)); then
        CMD+=(--compare-pretrained "${COMPARE_PRETRAINED[@]}")
    fi

    if ((${#COMPARE_NAMES[@]} > 0)); then
        CMD+=(--compare-names "${COMPARE_NAMES[@]}")
    fi

    if ((${#COMPARE_MODES[@]} > 0)); then
        CMD+=(--compare-modes "${COMPARE_MODES[@]}")
    fi

    if ((${#COMPARE_HYPERBOLIC_SIMILARITIES[@]} > 0)); then
        CMD+=(--compare-hyperbolic-similarities "${COMPARE_HYPERBOLIC_SIMILARITIES[@]}")
    fi

    echo "[INFO] Running seed=${SEED}: ${CMD[*]}"
    "${CMD[@]}"
done
