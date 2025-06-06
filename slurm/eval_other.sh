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


DATA_ROOT="[test-set-dir]/newt"

python -m src.evaluation.newt \
        --model $MODEL_TYPE \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --log $LOG_FILEPATH \
        --workers 8 \

DATA_ROOT="[test-set-dir]/fishnet"

python -m src.evaluation.fishnet \
        --model $MODEL_TYPE \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --log $LOG_FILEPATH \
        --lr 1e-4 \
        --epochs 50 \
        --workers 8 \
        --eval_every 10 \


DATA_ROOT="[test-set-dir]/awa2"

python -m src.evaluation.awa2 \
        --model $MODEL_TYPE \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --log $LOG_FILEPATH \
        --lr 1e-5 \
        --epochs 50 \
        --workers 8 \
        --eval_every 10 \


DATA_ROOT="[test-set-dir]/herbarium19"

python -m src.evaluation.herbarium19 \
        --model $MODEL_TYPE \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --log $LOG_FILEPATH \
        --workers 8 \

DATA_ROOT="[test-set-dir]/plantdoc"

python -m src.evaluation.plantdoc \
        --model $MODEL_TYPE \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --log $LOG_FILEPATH \
        --workers 8 \
