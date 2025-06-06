#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --account=[account]
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=bioclip-2
#SBATCH --time=240:00:00
#SBATCH --mem=800GB

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

host_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $host_node

export RDZV_HOST=$host_node
export RDZV_PORT=29400

srun torchrun --nnodes=4 --nproc_per_node 8 \
  --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
  -m src.training.main \
  --save-frequency 1 \
  --train-data '[training-dir]/shard-{00000..24235}.tar' \
  --val-data '[evaluation-dir]/shard-{000000..000031}.tar' \
  --dataset-type 'webdataset' \
  --pretrained 'laion2b_s32b_b82k' \
  --text_type 'random' \
  --dataset-resampled \
  --warmup 1875 \
  --batch-size 2816 \
  --accum-freq 1 \
  --epochs 30 \
  --workers 8 \
  --model ViT-L-14 \
  --log-every-n-steps 1 \
  --lr 1e-4 \
  --seed 42 \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --logs './logs' \
  --precision bf16 \
  --continual-data '[laion-dir]/{00000..03999}.tar' \
  --continual_text_type '' \
  --continual-batch-size 320 \
