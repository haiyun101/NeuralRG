import os
import glob
import re
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 引入您熟悉的进度条

def calculate_correlation_numpy(samples):
    L = samples.shape[-1]
    max_r = L // 2
    G_r = np.zeros(max_r)
    
    for r in range(max_r):
        shift_x = np.roll(samples, shift=-r, axis=-1)
        shift_y = np.roll(samples, shift=-r, axis=-2)
        G_r[r] = (np.mean(samples * shift_x) + np.mean(samples * shift_y)) / 2.0

    return G_r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    args = parser.parse_args()

    target_path = os.path.join('./opt', args.folder)
    search_pattern = os.path.join(target_path, '**/*IsingHMCresult*.hdf5')
    files = glob.glob(search_pattern, recursive=True)
    
    def extract_epoch(filename):
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    files = sorted(files, key=extract_epoch)

    if not files:
        print("未找到任何 .hdf5 文件。")
        return

    print(f"共找到 {len(files)} 个文件，开始批量计算并绘图...")

    # 使用 tqdm 循环遍历所有的 epoch 文件
    for target_file in tqdm(files, desc="Processing Epochs"):
        epoch_num = extract_epoch(target_file)

        try:
            with h5py.File(target_file, 'r') as f:
                spins_raw = np.array(f['X'])
                L = spins_raw.shape[-1]
                spins_raw = spins_raw.reshape(-1, L, L)
                
                # === 核心修复 1：将连续变量二值化为 +1 / -1 物理自旋 ===
                spins = np.sign(spins_raw)
                
                # 计算 G(r)
                g_r_values = calculate_correlation_numpy(spins)
                
        except Exception as e:
            print(f"\n读取文件 {target_file} 出错: {e}")
            continue

        # === 核心修复 2：画图时切掉 r=0 ===
        plt.figure(figsize=(8, 6))
        
        r_values = np.arange(1, len(g_r_values))
        g_r_plot = g_r_values[1:]
        
        plt.plot(r_values, g_r_plot, marker='o', linestyle='-', color='b', label=f'Epoch {epoch_num}')
        
        # 画理论参考线 (斜率 -0.25)
        eta = 0.25
        c = g_r_plot[0] * (r_values[0] ** eta) 
        reference_line = c * (r_values ** -eta)
        plt.plot(r_values, reference_line, 'r--', label='Theory Critical ~ r^-0.25')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Distance r')
        plt.ylabel('Correlation G(r)')
        plt.title(f'Spin-Spin Correlation (Log-Log) for: {args.folder}\nL={L} | Epoch={epoch_num}')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # 保存图片：按照您原有的命名习惯，保持整齐
        save_name = f'Gr_epoch_{epoch_num:05d}.png'
        save_path = os.path.join(target_path, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 【极其关键的一步】：批量画图后必须关闭图表，否则会引发内存泄漏！
        plt.close()

    print(f"\n🎉 完美！所有 epoch 的 G(r) 图像已重新生成并保存在 {target_path} 目录下！")

if __name__ == "__main__":
    main()