#!/bin/bash
#SBATCH --job-name=NeuralRG_32_Crit   # 作业名称
#SBATCH --partition=gpu               # 使用 GPU 分区 (L=32 强烈建议用 GPU)
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80g    # <--- 关键！指定要 80GB 版本
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 4 个 CPU 用于数据加载
#SBATCH --mem=32G                     # 申请 32GB 内存 (大模型需要更多内存)
#SBATCH --time=24:00:00               # 运行时间限制
#SBATCH --output=neuralrg_32_%j.log   # 日志文件

# 加载环境
module load miniforge
source activate neuralrg

# 运行命令
# -L 32: 系统尺寸 32x32
# -T 2.269: 2D Ising 模型的临界温度
# -nhidden 64: 增加神经元数量以捕捉分形特征 (L=32 时建议加大)
# -folder: 结果保存到 ./opt/32Ising_Crit，避免覆盖旧数据
# -cuda 0: 使用第 0 号 GPU
python ./main.py \
  -L 32 \
  -T 2.269 \
  -folder ./opt/32Ising_Crit_80 \
  -batch 512 \
  -epochs 20000 \
  -nlayers 10 \
  -nmlp 3 \
  -nhidden 64 \
  -nrepeat 1 \
  -savePeriod 1000 \
  -symmetry \
  -cuda 0