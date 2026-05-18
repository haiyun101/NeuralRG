import os
import h5py
import re
import glob
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import torch

# --- 配置区 ---
DATA_DIR = "data"
EXACT_FILE = "etc/exactz.md"
L_TARGET = 32  # 我们关心的系统尺寸

# --- 1. 解析 exactz.md 获取理论值 ---
def get_theoretical_values(exact_path, L):
    theory_data = {}
    if not os.path.exists(exact_path):
        print(f"警告: 找不到理论值文件 {exact_path}")
        return theory_data
        
    with open(exact_path, 'r') as f:
        content = f.read()
    
    section_pattern = rf"Ising n={L}.*?\| T.*?\|(.*?)(?:\n\n|\n#|$)"
    match = re.search(section_pattern, content, re.DOTALL)
    if not match:
        return {}
    
    table_rows = match.group(1).strip().split('\n')
    
    # 临时列表，用于存储所有行的数据以便进行数值求导
    t_list = []
    lnz_list = []
    fix_list = []
    
    for row in table_rows:
        if '---' in row: continue
        cols = [c.strip() for c in row.split('|') if c.strip()]
        if len(cols) >= 3:
            try:
                t_list.append(float(cols[0]))
                lnz_list.append(float(cols[1]))
                fix_list.append(float(cols[2]))
            except ValueError:
                continue
                
    if not t_list:
        return {}

    # 将数据按温度 T 从小到大排序 (这对计算梯度至关重要)
    sorted_indices = np.argsort(t_list)
    t_arr = np.array(t_list)[sorted_indices]
    lnz_arr = np.array(lnz_list)[sorted_indices]
    fix_arr = np.array(fix_list)[sorted_indices]
    
    # 总的 ln Z (即 lnz + fix)
    total_lnz_arr = lnz_arr + fix_arr
    
    # 计算理论最小 Loss
    loss_arr = -total_lnz_arr
    
    # 使用中心差分计算 d(lnZ)/dT，必须使用总的 lnZ！
    d_total_lnz_dt = np.gradient(total_lnz_arr, t_arr)
    
    # 根据热力学公式计算 E 和 TS
    # E = T^2 * d(lnZ)/dT
    energy_arr = (t_arr**2) * d_total_lnz_dt
    
    # TS = E + T * lnZ
    entropy_t_arr = energy_arr + (t_arr * total_lnz_arr)
    
    # 将计算结果存回字典
    for i in range(len(t_arr)):
        theory_data[t_arr[i]] = {
            "min_loss": loss_arr[i],
            "energy": energy_arr[i],
            "entropy_T": entropy_t_arr[i]
        }
        
    return theory_data

# --- 2. 针对指定温度扫描 data 目录提取实验值 ---
def collect_experimental_results_for_T(data_path, target_t):
    results = {} 
    
    if not os.path.exists(data_path):
        print(f"警告: 找不到数据目录 {data_path}")
        return results

    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    target_t_str = f"{target_t:g}"
    
    for folder in folders:
        # 提取温度和方法名。例如: 32Ising_T2.3_nsym_HP
        match = re.search(r"T(\d+\.\d+)_(\w+)", folder)
        if not match: continue
        
        folder_t = float(match.group(1))
        # 允许 0.001 的误差，这样 2.269 就能匹配 2.269185...
        if not math.isclose(folder_t, target_t, abs_tol=1e-3):
            continue  # 跳过不符合目标温度的文件夹
            
        method = match.group(2)
        record_path = os.path.join(data_path, folder, "records")
        if not os.path.exists(record_path): continue
        
        # 使用 glob 和正确的排序逻辑获取最新的 record 文件
        all_records = sorted(
            glob.glob(os.path.join(record_path, "*Record_epoch*.hdf5")),
            key=lambda x: int(x.split('epoch')[-1].split('.')[0]) if 'epoch' in x else -1
        )
        
        if not all_records: continue
        latest_record = all_records[-1]
        
        try:
            with h5py.File(latest_record, "r") as rf:
                # 获取 Loss
                full_loss = np.array(rf["LOSS"]).flatten()
                
                # 找到最小 Loss 及其索引
                min_idx = np.argmin(full_loss)
                min_loss = full_loss[min_idx]
                
                # 提取对应的 Energy 和 Entropy (如果存在)
                corr_energy = "N/A"
                corr_entropy_t = "N/A"
                
                if "ENERGY" in rf:
                    full_energy = np.array(rf["ENERGY"]).flatten()
                    if min_idx < len(full_energy):
                        corr_energy = full_energy[min_idx]
                        
                if "ENTROPY" in rf:
                    full_entropy = np.array(rf["ENTROPY"]).flatten()
                    if min_idx < len(full_entropy):
                        # Entropy 乘以 温度 T
                        corr_entropy_t = full_entropy[min_idx] * target_t
                
                results[method] = {
                    "min_loss": min_loss,
                    "energy": corr_energy,
                    "entropy_T": corr_entropy_t
                }
                
        except Exception as e:
            print(f"Error reading {latest_record}: {e}")
            
    return results

# --- 新增: 计算 MCMC 样本的平均能量 ---
def get_mcmc_energy(mcmc_dir, L, target_t):
    """
    搜索并读取对应的 MCMC .pt 文件，计算并返回平均能量。
    假设哈密顿量为 H = -J * sum(s_i * s_j)，且 J=1。
    """
    if not os.path.exists(mcmc_dir):
        return "N/A"
        
    # 寻找匹配的 MCMC 文件 (允许一定温度容差)
    mcmc_files = glob.glob(os.path.join(mcmc_dir, f"mcmc_wolff_L{L}_T*.pt"))
    target_file = None
    
    for fpath in mcmc_files:
        # 提取文件名中的温度
        match = re.search(r"_T([\d\.]+)_", fpath)
        if match:
            file_t = float(match.group(1))
            if math.isclose(file_t, target_t, abs_tol=1e-3):
                target_file = fpath
                break
                
    if not target_file:
        return "N/A"
        
    try:
        # 载入数据，形状应为 (N, 1, L, L)
        samples = torch.load(target_file, weights_only=True)
        
        # 使用 PyTorch 的张量平移 (roll) 计算周期性边界条件的近邻相互作用
        # dim=-2 是竖直方向 (行)，dim=-1 是水平方向 (列)
        interaction_v = samples * torch.roll(samples, shifts=-1, dims=-2)
        interaction_h = samples * torch.roll(samples, shifts=-1, dims=-1)
        
        # 计算每个样本的总能量 H = - (E_v + E_h)
        # 沿最后三个维度 (C, H, W) 求和
        energy_per_sample = -torch.sum(interaction_v + interaction_h, dim=(-3, -2, -1))
        
        # 返回所有样本的平均能量
        mean_energy = torch.mean(energy_per_sample).item()
        return mean_energy
        
    except Exception as e:
        print(f"读取 MCMC 文件出错 {target_file}: {e}")
        return "N/A"


# --- 3. 生成 Markdown 报告 ---
def generate_report_for_T(target_t):
    # 确保输出目录存在
    os.makedirs("analyzers", exist_ok=True)
    output_file = f"analyzers/loss_report_T{target_t:g}.md"
    
    target_t_str = f"{target_t:g}"
        
    # 1. 获取理论值
    theory = get_theoretical_values(EXACT_FILE, L_TARGET)
    t_theory_loss, t_theory_energy, t_theory_entropy = "N/A", "N/A", "N/A"
    
    # 使用容差遍历匹配理论温度
    for t_val, data in theory.items():
        if math.isclose(t_val, target_t, abs_tol=1e-3):
            t_theory_loss = data["min_loss"]
            t_theory_energy = data["energy"]
            t_theory_entropy = data["entropy_T"]
            break
    
    # 2. 获取该温度下的所有实验数据
    experiment_data = collect_experimental_results_for_T(DATA_DIR, target_t)
    
    if not experiment_data:
        print(f"未找到 T={target_t_str} 的实验数据。")
        return
    
    # 3. 开始构建 Markdown
    lines = [
        f"# Ising L={L_TARGET} 特定温度分析报告 (T={target_t_str})",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "| 方法 (Method) | 最小 Loss (Min Loss) | 对应 Energy @ Min Loss | 对应 Entropy * T @ Min Loss |",
        "| :--- | :---: | :---: | :---: |"
    ]
    
    # 填入理论值
    loss_str = f"**{t_theory_loss:.6f}**" if isinstance(t_theory_loss, (float, np.floating)) else t_theory_loss
    energy_str = f"**{t_theory_energy:.6f}**" if isinstance(t_theory_energy, (float, np.floating)) else t_theory_energy
    entropy_str = f"**{t_theory_entropy:.6f}**" if isinstance(t_theory_entropy, (float, np.floating)) else t_theory_entropy
    
    lines.append(f"| **理论值 (Exact+Fix)** | {loss_str} | {energy_str} | {entropy_str} |")
    
    # 获取 MCMC 基准能量 (假设你的数据保存在 data/mcmc_data)
    mcmc_dir = os.path.join("data", "mcmc_data")
    mcmc_energy = get_mcmc_energy(mcmc_dir, L_TARGET, target_t)
    
    mcmc_e_str = f"**{mcmc_energy:.6f}**" if isinstance(mcmc_energy, float) else mcmc_energy
    
    # MCMC 无法直接给出确切的 Loss 和 Entropy，所以记为 N/A
    lines.append(f"| **MCMC 基准 (Wolff)** | N/A | {mcmc_e_str} | N/A |")

    # 填入实验方法的数据并按方法名字典序排列
    for method in sorted(experiment_data.keys()):
        data = experiment_data[method]
        
        ml_str = f"{data['min_loss']:.6f}" if isinstance(data['min_loss'], (float, np.floating)) else data['min_loss']
        e_str = f"{data['energy']:.6f}" if isinstance(data['energy'], (float, np.floating)) else data['energy']
        st_str = f"{data['entropy_T']:.6f}" if isinstance(data['entropy_T'], (float, np.floating)) else data['entropy_T']
        
        lines.append(f"| {method} | {ml_str} | {e_str} | {st_str} |")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"特定温度报告已生成: {output_file}")

if __name__ == "__main__":
    # 使用 argparse 接收命令行参数
    parser = argparse.ArgumentParser(description="Generate minimum loss report for a specific temperature.")
    parser.add_argument("-t", "--temp", type=float, required=True, help="Specify the temperature T (e.g. 2.3)")
    
    args = parser.parse_args()
    
    generate_report_for_T(args.temp)