#!/bin/bash

# ==========================================
# NeuralRG 迁移学习辅助工具 (Transfer Tool)
# 功能: 自动复制最新模型权重，并修改参数文件中的温度(T)和学习率(LR)建议
# 用法: bash transfer_tool.sh [源目录] [目标目录] [新温度]
# ==========================================

# 1. 检查参数输入
if [ "$#" -ne 3 ]; then
    echo "❌ 错误: 参数不足！"
    echo "正确用法: bash transfer_tool.sh <源目录> <目标目录> <新温度>"
    echo "示例: bash transfer_tool.sh ./opt/32Ising_HighT_2.5 ./opt/32Ising_Anneal_2.4 2.4"
    exit 1
fi

SRC=$1
DST=$2
NEW_T=$3

echo "----------------------------------------"
echo "📂 源目录 (Source): $SRC"
echo "📂 目标目录 (Target): $DST"
echo "🌡️  设定新温度 (New T): $NEW_T"
echo "----------------------------------------"

# 2. 检查源目录是否存在
if [ ! -d "$SRC" ]; then
    echo "❌ 错误: 找不到源目录 $SRC"
    exit 1
fi

# 3. 创建目标目录并复制关键文件
echo "[1/3] 正在创建目录并复制文件..."
mkdir -p "$DST/savings"

# 复制参数文件
cp "$SRC/parameters.hdf5" "$DST/"

# 复制最新的模型权重 (自动找到数字最大的 .saving 文件)
# ls -v 会按数字自然排序 (epoch100 < epoch1000)，tail -n 1 取最后一个
LATEST_SAVING=$(ls -v "$SRC/savings/"*.saving 2>/dev/null | tail -n 1)

if [ -z "$LATEST_SAVING" ]; then
    echo "⚠️ 警告: 源目录里没有找到 .saving 文件！将只复制 parameters.hdf5。"
else
    echo "✅ 复制最新模型权重: $(basename "$LATEST_SAVING")"
    cp "$LATEST_SAVING" "$DST/savings/"
fi

# 4. 调用 Python 修改 parameters.hdf5
echo "[2/3] 正在修改 HDF5 参数文件..."

python3 -c "
import h5py
import sys
import numpy as np

# 从 Shell 变量获取参数
file_path = '$DST/parameters.hdf5'
new_val = float('$NEW_T')

try:
    with h5py.File(file_path, 'r+') as f:
        # --- 修改温度 T ---
        if 'T' in f:
            old_val = f['T'][()]
            print(f'  - 原温度 T: {old_val}')
            del f['T']
        f.create_dataset('T', data=new_val)
        print(f'  - 新温度 T: {new_val}')
        
        # --- 如果接近临界点，自动修正 K 值 (仅针对 2D Ising) ---
        # Tc ≈ 2.269, Kc ≈ 0.44068
        if abs(new_val - 2.269185) < 0.001:
            Kc = 0.4406867935
            if 'K' in f: 
                print(f'  - 原耦合 K: {f[\"K\"][()]}')
                del f['K']
            f.create_dataset('K', data=Kc)
            print(f'  - 自动修正 K -> {Kc} (Critical Point)')

    print('✅ 参数修改成功 (SUCCESS)')

except Exception as e:
    print(f'❌ 错误 (ERROR): {e}')
    sys.exit(1)
"

# 检查 Python 是否执行成功
if [ $? -eq 0 ]; then
    echo "[3/3] 准备就绪！"
    echo "----------------------------------------"
    echo "🚀 下一步建议:"
    echo "请运行: python main.py -folder $DST -load -lr 0.0005 -savePeriod 100 ..."
    echo "(注意: 建议使用较小的学习率，如 0.0005 或 0.0001)"
else
    echo "❌ 错误: Python 脚本执行失败，请检查。"
    exit 1
fi
