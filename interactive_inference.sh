#!/bin/bash
#SBATCH --job-name=pytorch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=97G
#SBATCH --time=48:00:00
#SBATCH --output=logs_test/run-%j.log
#SBATCH --qos=gpu_access

source /mnt/whitsett/yilinliu/miniconda3/etc/profile.d/conda.sh
conda activate longlive2

# Array of seeds to run
SEEDS=(1 3 42 1024 2048 4096 8192)
CONFIG_PATH="configs/inference.yaml"

for SEED in "${SEEDS[@]}"; do
  echo "========== Running with seed=$SEED =========="
  # Update seed in config file
  sed -i "s/^  seed: .*/  seed: $SEED/" "$CONFIG_PATH"
  
  CUDA_VISIBLE_DEVICES=5 python \
    inference.py \
    --config_path "$CONFIG_PATH"
done 
