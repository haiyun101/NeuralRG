#!/bin/bash

# 定义温度列表
# temps=(3 2.9 2.8 2.7 2.6 2.5 2.4 2.35 2.3 2.28 2.269)
temps=(5 4.5 4 3.5)

# 循环遍历每个温度
for T in "${temps[@]}"
do
    # 定义任务名称和文件夹路径
    # 例如：T3_nsym, T2.269_nsym
    JOB_NAME="T${T}_nsym"
    OUT_DIR="./data/32Ising_T${T}_nsym"
    
    # 自动创建数据文件夹（如果不存在）
    mkdir -p $OUT_DIR

    echo "Submitting job for Temperature: $T"

    # 使用 sbatch 提交任务
    # 通过 --export 将变量传递给脚本，或直接在命令行构建指令
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=${OUT_DIR}/${JOB_NAME}_%j.log

module load miniforge
source activate neuralrg

# 执行 Python 程序
# 注意：根据你之前的截图，我将 -folder 指向了新定义的 OUT_DIR
python ./main.py \\
    -L 32 \\
    -T ${T} \\
    -folder ${OUT_DIR} \\
    -batch 128 \\
    -epochs 1000 \\
    -nlayers 10 \\
    -nmlp 3 \\
    -nhidden 64 \\
    -nrepeat 1 \\
    -savePeriod 10 \\
    -cuda 0 
EOT

done