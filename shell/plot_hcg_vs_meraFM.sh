#!/bin/bash -l
#SBATCH --job-name=plot_hcg_vs_fm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/plot_hcg_vs_fm_%j.out
#SBATCH --error=./logs/plot_hcg_vs_fm_%j.err

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs figures/hcg_vs_meraFM

python -u analyzers/rg_fixed_point/plot_hcg_vs_meraFM.py \
    --N 1000 --batch 32 --device cuda \
    --out figures/hcg_vs_meraFM

date
