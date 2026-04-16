#!/bin/bash
#SBATCH --job-name=64_H_40g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                    # 内存也申请大一点
#SBATCH --time=24:00:00
#SBATCH --output=neuralrg_64_high_4layers_40g_%j.log

module load miniforge
source activate neuralrg
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# 显存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 关键参数调整 ---
# T = 3.0 higher than crit = 2.269
# L = 64 (目标尺寸)
# Batch = 64 (为了安全放入 80GB 显存，显存占用约为 L=32,B=256 的水平)
# Epochs = 80000 (因为 Batch 小，且系统大难训练，需要更多轮数)
# 错误之前nlayers = 12 (比 L=32 时增加 2 层，以增强表达能力)，但是过拟合到铁磁态
# 按照paper nlayer = 4 or smaller
python -u ./main.py \
  -L 64 \
  -T 3.0 \
  -folder ./opt/64Ising_High_4layers_40g \
  -batch 64 \
  -epochs 80000 \
  -nlayers 4 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 1000 \
  -symmetry \
  -cuda -1