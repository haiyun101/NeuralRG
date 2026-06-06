import os
import h5py
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# --- 配置区 ---
DATA_DIR = "data"
EXACT_FILE = "etc/exactz.md"
OUTPUT_FILE = "analyzers/loss_comparison_report.md"
L_TARGET = 32  # 我们关心的系统尺寸

# --- 1. 解析 exactz.md 获取理论值 ---
def get_theoretical_values(exact_path, L):
    theory_data = {}
    with open(exact_path, 'r') as f:
        content = f.read()
    
    # 定位到对应 L 的表格部分
    section_pattern = rf"Ising n={L}.*?\| T.*?\|(.*?)(?:\n\n|\n#|$)"
    match = re.search(section_pattern, content, re.DOTALL)
    if not match:
        return {}
    
    table_rows = match.group(1).strip().split('\n')
    for row in table_rows:
        if '---' in row: continue
        cols = [c.strip() for c in row.split('|') if c.strip()]
        if len(cols) >= 3:
            try:
                t_val = float(cols[0])
                lnz = float(cols[1])
                fix = float(cols[2])
                # 计算理论最小 Loss = -(lnZ + fix)
                theory_data[f"{t_val:.1f}"] = -(lnz + fix)
            except ValueError:
                continue
    return theory_data

# --- 2. 扫描 data 目录提取实验值 ---
def collect_experimental_results(data_path):
    results = {} # 结构: { "T": { "method": min_loss } }
    
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    for folder in folders:
        # 提取温度和方法名。例如: 32Ising_T2.3_nsym_HP
        # 匹配模式: T{数字} 后面跟着方法名
        match = re.search(r"T(\d+\.\d+)_(\w+)", folder)
        if not match: continue
        
        temp_str = f"{float(match.group(1)):.1f}"
        method = match.group(2)
        
        record_path = os.path.join(data_path, folder, "records")
        if not os.path.exists(record_path): continue
        
        # 找到序号最大的（最新的）HDF5 文件
        hdf5_files = [f for f in os.listdir(record_path) if f.endswith(".hdf5")]
        if not hdf5_files: continue
        
        latest_file = sorted(hdf5_files)[-1]
        
        try:
            with h5py.File(os.path.join(record_path, latest_file), "r") as f:
                # 假设 LOSS 存储为 [N, 2] 的数组，第一列是均值
                loss_history = f["LOSS"][:]
                if loss_history.ndim == 2:
                    min_loss = np.min(loss_history[:, 0])
                else:
                    min_loss = np.min(loss_history)
                
                if temp_str not in results:
                    results[temp_str] = {}
                results[temp_str][method] = min_loss
        except Exception as e:
            print(f"Error reading {folder}: {e}")
            
    return results

# # --- 3. 生成 Markdown 表格 ---
# def generate_report():
#     theory = get_theoretical_values(EXACT_FILE, L_TARGET)
#     experiment = collect_experimental_results(DATA_DIR)
    
#     # 确定所有出现过的方法，用于表头
#     all_methods = set()
#     for t in experiment:
#         all_methods.update(experiment[t].keys())
#     sorted_methods = sorted(list(all_methods))
    
#     # 开始构建 Markdown
#     lines = [
#         f"# Ising L={L_TARGET} 训练结果对比报告",
#         f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
#         "| 温度 (T) | " + " | ".join(sorted_methods) + " | **理论最小值 (Exact+Fix)** |",
#         "| :--- | " + " | ".join([":---:"] * len(sorted_methods)) + " | :--- |"
#     ]
    
#     # 按温度排序填入数据
#     for t_str in sorted(experiment.keys(), key=float):
#         row = [f"**{t_str}**"]
#         for m in sorted_methods:
#             val = experiment[t_str].get(m, "N/A")
#             row.append(f"{val:.4f}" if isinstance(val, float) else val)
        
#         # 填入理论值
#         t_min = theory.get(t_str, "N/A")
#         row.append(f"**{t_min:.4f}**" if isinstance(t_min, float) else t_min)
        
#         lines.append("| " + " | ".join(row) + " |")
    
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines))
    
#     print(f"报告已生成: {OUTPUT_FILE}")

# --- 4. 生成对比图 ---
def plot_comparison(experiment, theory, sorted_methods, l_target):
    plt.figure(figsize=(10, 6))
    
    # 提取温度并排序
    temps = sorted([t for t in experiment.keys()], key=float)
    t_numeric = [float(t) for t in temps]
    
    # 1. 绘制理论值曲线
    theory_vals = []
    for t in temps:
        val = theory.get(t, None)
        theory_vals.append(val)
    
    if any(v is not None for v in theory_vals):
        plt.plot(t_numeric, theory_vals, 'k--', label='Theoretical Min (Exact+Fix)', linewidth=2, zorder=3)

    # 2. 绘制各实验方法曲线
    markers = ['o', 's', '^', 'v', 'D', 'x']
    for i, method in enumerate(sorted_methods):
        method_vals = []
        valid_t = []
        for t in temps:
            val = experiment[t].get(method, None)
            if isinstance(val, (int, float)):
                method_vals.append(val)
                valid_t.append(float(t))
        
        if method_vals:
            plt.plot(valid_t, method_vals, marker=markers[i % len(markers)], 
                     label=f'Method: {method}', alpha=0.8)

    plt.title(f"Ising L={l_target} Loss Comparison: Experiment vs Theory")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Total Loss")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 保存图片
    plot_file = "analyzers/loss_comparison_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {plot_file}")
    return plot_file

# --- 3. 修改后的生成报告函数 ---
def generate_report():
    theory = get_theoretical_values(EXACT_FILE, L_TARGET)
    experiment = collect_experimental_results(DATA_DIR)
    
    all_methods = set()
    for t in experiment:
        all_methods.update(experiment[t].keys())
    sorted_methods = sorted(list(all_methods))
    
    # 构建 Markdown
    lines = [
        f"# Ising L={L_TARGET} 训练结果对比报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 1. 数据统计表",
        "| 温度 (T) | " + " | ".join(sorted_methods) + " | **理论最小值 (Exact+Fix)** |",
        "| :--- | " + " | ".join([":---:"] * len(sorted_methods)) + " | :--- |"
    ]
    
    for t_str in sorted(experiment.keys(), key=float):
        row = [f"**{t_str}**"]
        for m in sorted_methods:
            val = experiment[t_str].get(m, "N/A")
            row.append(f"{val:.4f}" if isinstance(val, float) else val)
        t_min = theory.get(t_str, "N/A")
        row.append(f"**{t_min:.4f}**" if isinstance(t_min, float) else t_min)
        lines.append("| " + " | ".join(row) + " |")
    
    # 增加图片引用到 Markdown
    plot_path = plot_comparison(experiment, theory, sorted_methods, L_TARGET)
    lines.append("\n## 2. 可视化对比图")
    lines.append(f"![Loss Comparison](./{plot_path})")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"报告已生成: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()