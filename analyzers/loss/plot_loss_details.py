import torch
import numpy as np
import h5py
import os
import glob
import matplotlib.pyplot as plt
import argparse
import sys

# 导入 NeuralRG 源码 (假设该脚本放在 analyzers/ 下)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import source

def analyze_from_hdf5(folder_path):
    # 1. 提取超参数
    param_file = os.path.join(folder_path, "parameters.hdf5")
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"找不到 {param_file}")
        
    with h5py.File(param_file, "r") as f:
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        T = float(np.array(f["T"]))
        
    device = torch.device("cpu")
    target = source.Ising(L, d, T).to(device=device, dtype=torch.float32)

    # 2. 提取完整的 Loss 曲线
    record_files = sorted(glob.glob(os.path.join(folder_path, "records", "*Record_epoch*.hdf5")),
                          key=lambda x: int(x.split('_epoch')[-1].split('.')[0]))
    
    if not record_files:
        print(f"在 {folder_path} 找不到 record 文件。")
        return [], [], [], [], T
        
    latest_record = record_files[-1]
    with h5py.File(latest_record, "r") as f:
        full_loss = f["LOSS"][:] # 包含每一轮的 loss 历史 (由于代码逻辑是 append，取最后一列的完整数组)
    
    # 3. 读取所有的 HMCresult 里的样本来计算能量
    hmc_files = sorted(glob.glob(os.path.join(folder_path, "records", "*HMCresult_epoch*.hdf5")),
                       key=lambda x: int(x.split('_epoch')[-1].split('.')[0]))
    
    sample_epochs = []
    energies = []
    neg_entropies = [] # 存储 -S_cont
    
    print(f"正在从 {folder_path} 的 HDF5 样本中反推物理量...")
    for hf in hmc_files:
        epoch = int(hf.split('_epoch')[-1].split('.')[0])
        
        with h5py.File(hf, "r") as f:
            if "X" not in f:
                continue
            X = f["X"][:] # 提取网络生成的采样数据
        
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        
        # 使用真实的物理对象计算有效能量
        with torch.no_grad():
            E_eff = target.energy(x_tensor).mean().item()
        
        # 反推负熵: 因为 Loss = E_eff - S_cont, 所以 -S_cont = Loss - E_eff
        if epoch < len(full_loss):
            current_loss = full_loss[epoch]
            neg_S_cont = current_loss - E_eff
            
            sample_epochs.append(epoch)
            energies.append(E_eff)
            neg_entropies.append(neg_S_cont)
        
    return full_loss, sample_epochs, energies, neg_entropies, T

def plot_reconstructed(full_loss, sample_epochs, energies, neg_entropies, T, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 共用一个坐标轴
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Value (Total Loss / Energy / -Entropy)', fontsize=12)
    
    # 绘制逐轮的密集 Loss 线
    ax1.plot(range(len(full_loss)), full_loss, color='tab:red', label='Total Loss (F)', alpha=0.5, linewidth=1.5)
    
    # 绘制离散采样的 Energy
    ax1.plot(sample_epochs, energies, color='tab:orange', label='Energy $\\langle E_{eff} \\rangle$', marker='o', markersize=4, linewidth=2)
    
    # 绘制离散采样的负熵 (-S_cont)
    ax1.plot(sample_epochs, neg_entropies, color='tab:blue', label='Negative Entropy ($-S_{cont}$)', marker='x', markersize=4, linewidth=2)
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)

    # 标题加入等式，方便物理理解
    plt.title(f'Reconstructed VI Dynamics from Samples (T={T})\n$Loss = \\langle E_{{eff}} \\rangle + (-S_{{cont}})$', fontsize=14)
    fig.tight_layout()  
    
    # 保存图片
    plt.savefig(save_path, dpi=300)
    print(f"图表已成功保存至: {save_path}")

if __name__ == "__main__":
    # 要求 1: 使用 argparse 指定文件夹
    parser = argparse.ArgumentParser(description="Analyze Loss components from HDF5")
    parser.add_argument("-folder", type=str, required=True, help="需要分析的实验数据目录路径")
    args = parser.parse_args()
    
    folder = args.folder
    # 保证路径格式正确
    if not folder.endswith(os.sep):
        folder += os.sep

    # 提取并计算数据
    full_loss, sample_epochs, energies, neg_entropies, T = analyze_from_hdf5(folder)
    
    if len(full_loss) > 0:
        # 要求 2: 将图片保存进对应的 folder 内
        save_name = os.path.join(folder, f"Decompose_loss.png")
        
        # 要求 3: 画出共享 Y 轴的图像
        plot_reconstructed(full_loss, sample_epochs, energies, neg_entropies, T, save_name)