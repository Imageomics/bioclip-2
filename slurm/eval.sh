#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bioclip-eval
#SBATCH --time=8:00:00
#SBATCH --mem=400GB

export CUDA_VISIBLE_DEVICES=0

LOG_FILEPATH="../storage/logs"
MODEL_TYPE="ViT-L-14"
PRETRAINED="hf-hub:imageomics/bioclip-2"

TASK_TYPE="all"
TEXT_TYPE="taxon_com"

DATA_ROOTS=(
    "[test-set-dir]/CameraTrap/images/desert-lion/"
    "[test-set-dir]/CameraTrap/images/ENA24/"
    "[test-set-dir]/CameraTrap/images/island/"
    "[test-set-dir]/CameraTrap/images/orinoquia/"
    "[test-set-dir]/CameraTrap/images/ohio-small-animals/"
)
LABEL_FILES=(
    "[test-set-dir]/CameraTrap/desert-lion-balanced.csv"
    "[test-set-dir]/CameraTrap/ENA24-balanced.csv"
    "[test-set-dir]/CameraTrap/island-balanced.csv"
    "[test-set-dir]/CameraTrap/orinoquia-balanced.csv"
    "[test-set-dir]/CameraTrap/ohio-small-animals-balanced.csv"
)

for i in "${!DATA_ROOTS[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$i]}
    LABEL_FILE=${LABEL_FILES[$i]}

    python -m src.evaluation.zero_shot_iid \
            --model $MODEL_TYPE \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --text_type $TEXT_TYPE \

    python -m src.evaluation.few_shot \
            --model $MODEL_TYPE \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --task_type $TASK_TYPE \
            --nfold 5 \
            --kshot_list 1 5 \

done

TEXT_TYPE="asis"

DATA_ROOTS=(
    "[test-set-dir]/meta-album/set0/PLK_Mini/val"
    "[test-set-dir]/meta-album/set2/INS_Mini/val"
    "[test-set-dir]/meta-album/set1/INS_2_Mini/val"
    "[test-set-dir]/meta-album/set1/PLT_NET_Mini/val"
    "[test-set-dir]/meta-album/set2/FNG_Mini/val"
    "[test-set-dir]/meta-album/set0/PLT_VIL_Mini/val"
    "[test-set-dir]/meta-album/set1/MED_LF_Mini/val"
    "[test-set-dir]/nabird/images/"
)
LABEL_FILES=(
    "[test-set-dir]/meta-album/PLK_Mini/val/metadata.csv"
    "[test-set-dir]/meta-album/INS_Mini/metadata.csv"
    "[test-set-dir]/meta-album/INS_2_Mini/metadata.csv"
    "[test-set-dir]/meta-album/PLT_NET_Mini/metadata.csv"
    "[test-set-dir]/meta-album/FNG_Mini/metadata.csv"
    "[test-set-dir]/meta-album/PLT_VIL_Mini/val/metadata.csv"
    "[test-set-dir]/meta-album/MED_LF_Mini/metadata.csv"
    "[test-set-dir]/nabird/metadata.csv"
)

for i in "${!DATA_ROOTS[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$i]}
    LABEL_FILE=${LABEL_FILES[$i]}

    python -m src.evaluation.zero_shot_iid \
            --model $MODEL_TYPE \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --text_type $TEXT_TYPE \

    python -m src.evaluation.few_shot \
            --model $MODEL_TYPE \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --task_type $TASK_TYPE \
            --nfold 5 \
            --kshot_list 1 5 \

done


TEXT_TYPE="sci_com"
DATA_ROOT="[test-set-dir]/rare-species/"
LABEL_FILE="[test-set-dir]/rare-species/metadata.csv"

python -m src.evaluation.zero_shot_iid \
        --model $MODEL_TYPE \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --label_filename $LABEL_FILE \
        --log $LOG_FILEPATH \
        --text_type $TEXT_TYPE \

python -m src.evaluation.few_shot \
        --model $MODEL_TYPE \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --label_filename $LABEL_FILE \
        --log $LOG_FILEPATH \
        --task_type $TASK_TYPE \
        --nfold 5 \
        --kshot_list 1 5 \
