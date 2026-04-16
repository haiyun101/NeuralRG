#!/bin/bash
#SBATCH --job-name=2.5_wt_con
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1              # 依然建议用 A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=16:00:00                
#SBATCH --output=32Ising_T2.5_weightTying_continue_%j.log

module load miniforge
source activate neuralrg

# T = 2.6
python ./main.py \
  -L 32 \
  -T 2.3 \
  -folder ./opt/32Ising_T2.5_weightTying_continue\
  -load \
  -lr 0.0001 \
  -batch 128 \
  -epochs 8000 \
  -nlayers 10 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 10 \
  -symmetry \
  -cuda 0