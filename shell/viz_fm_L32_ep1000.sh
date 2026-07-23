#!/bin/bash -l
#SBATCH --job-name=fm_viz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=./logs/fm_viz_%j.out
#SBATCH --error=./logs/fm_viz_%j.err

module load miniforge
source activate neuralrg
mkdir -p logs

python -u analyzers/fm_visualize_field.py \
    --folder data/L32_T2.269_flowmatching_h64 \
    --epoch 1000 \
    --n_samples 6 --n_frames 6 --steps 100 \
    --device cuda:0

echo "Done."
date
