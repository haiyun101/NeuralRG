#!/bin/bash
#SBATCH --job-name=32_C_2layers   # 作业名称
#SBATCH --partition=gpu               # 使用 GPU 分区 (L=32 强烈建议用 GPU)
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 4 个 CPU 用于数据加载
#SBATCH --mem=16G                     # 申请 32GB 内存 (大模型需要更多内存)
#SBATCH --time=24:00:00               # 运行时间限制
#SBATCH --output=neuralrg_32_2layers%j.log   # 日志文件

# 加载环境
module load miniforge
source activate neuralrg

# 运行命令
# -L 32: 系统尺寸 32x32
# -T 2.269: 2D Ising 模型的临界温度
# -nhidden 64: 增加神经元数量以捕捉分形特征 (L=32 时建议加大)
# -cuda 0: 使用第 0 号 GPU
# -nlayers :2 instead of 3， 4, 10
python ./main.py \
  -L 32 \
  -T 2.269 \
  -folder ./opt/32Ising_Crit_2layers \
  -batch 128 \
  -epochs 80000 \
  -nlayers 2 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 1000 \
  -symmetry \
  -cuda 0