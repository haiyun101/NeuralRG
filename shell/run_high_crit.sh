#!/bin/bash
#SBATCH --job-name=32_A_2.5to2.4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1              # 依然建议用 A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00                
#SBATCH --output=neuralrg_32_anneal_2.5to2.4_%j.log

module load miniforge
source activate neuralrg

# T = 2.4 
python ./main.py \
  -L 32 \
  -T 2.4 \
  -lr 0.0005 \
  -folder ./opt/32Ising_Anneal_2.5to2.4 \
  -load \
  -batch 128 \
  -epochs 8000 \
  -nlayers 10 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 100 \
  -symmetry \
  -cuda 0