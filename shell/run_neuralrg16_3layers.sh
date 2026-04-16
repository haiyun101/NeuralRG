#!/bin/bash
#SBATCH --job-name=16c_2_32   # 作业名称
#SBATCH --partition=gpu              # 使用 batch 队列（如果是 GPU 运行请用 gpu 分区）
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                      # 使用 1 个节点
#SBATCH --ntasks=1                     # 1 个任务
#SBATCH --cpus-per-task=4              # 申请 4 个 CPU 核心
#SBATCH --mem=8G                      # 申请 16GB 内存
#SBATCH --time=24:00:00                # 最长运行时间 24 小时
#SBATCH --output=neuralrg_16_crit_2_32_%j.log       # 日志输出文件

# 加载环境
module load miniforge
source activate neuralrg

# 运行命令 
python ./main.py -epochs 5000 -folder ./opt/16Ising_2layers_32hidden -batch 512 -nlayers 2 -nmlp 3 -nhidden 32 -L 16 -nrepeat 1 -savePeriod 100 -sym -cuda -1
