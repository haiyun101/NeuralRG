#!/bin/bash
#SBATCH --job-name=2.4_s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=32Ising_T2.4_nvsy%j.log

module load miniforge
source activate neuralrg


python ./main.py \
  -L 32 \
  -T 2.4 \
  -folder ./opt/32Ising_T2.4_nvsy \
  -batch 128 \
  -epochs 800 \
  -nlayers 10 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 10 \
  -cuda 0 \