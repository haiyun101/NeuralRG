#!/bin/bash
#SBATCH --job-name=NeuralRG_32_Crit_H100
#SBATCH --partition=gpu                 # 如果排队太久，可以改为 preempt 试试
#SBATCH --gres=gpu:h100:1               # 直接申请 H100 显卡
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                       # H100 节点通常内存充足，给大一点
#SBATCH --time=24:00:00
#SBATCH --output=neuralrg_crit_h100_%j.log

module load miniforge
source activate neuralrg

# 显存优化（防止碎片化）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# T = 2.269 (临界温度)
# Batch = 256 (H100 80GB 应该能轻松吃下，如果还不行就降到 128)
# Epochs = 40000 (保证总训练样本量足够)

python ./main.py \
  -L 32 \
  -T 2.269 \
  -folder ./opt/32Ising_Crit_H100 \
  -batch 256 \
  -epochs 40000 \
  -nlayers 10 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 1000 \
  -symmetry \
  -cuda 0