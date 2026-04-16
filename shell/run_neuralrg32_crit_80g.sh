#!/bin/bash
#SBATCH --job-name=NeuralRG_32_Crit_80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G        # <--- 关键修改：修正了标签拼写
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                    # 80G显存通常配大内存节点，申请64G没问题
#SBATCH --time=24:00:00
#SBATCH --output=neuralrg32_crit_80g_%j.log

module load miniforge
source activate neuralrg

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# T = 2.269 (临界温度)
# Batch = 512 (80GB 显存应该能吃得下，之前 40GB 卡是在 512 时爆的)
# Epochs = 20000 (因为 Batch 大了，Epochs 可以回落到 10000)

python ./main.py \
  -L 32 \
  -T 2.269 \
  -folder ./opt/32Ising_Crit_80G \
  -batch 512 \
  -epochs 20000 \
  -nlayers 10 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 500 \
  -symmetry \
  -cuda 0