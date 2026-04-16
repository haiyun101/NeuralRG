#!/bin/bash
#SBATCH --job-name=32_LowT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1              # 依然建议用 A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00                # 收敛通常比临界态快，12小时足够
#SBATCH --output=neuralrg_low_%j.log

module load miniforge
source activate neuralrg

# T = 2.0 (低于临界点 2.269)
# 结果保存到 ./opt/32Ising_LowT
python ./main.py \
  -L 32 \
  -T 2.0 \
  -folder ./opt/32Ising_LowT \
  -batch 128 \
  -epochs 80000 \
  -nlayers 10 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 1000 \
  -symmetry \
  -cuda 0