import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
import argparse

def get_magnetization_curve():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Plot magnetization curve for a specific folder under ./opt/')
    parser.add_argument('folder', type=str, help='The folder name under ./opt/ (e.g., 32Ising_Crit)')
    args = parser.parse_args()

    # 构造目标路径
    target_path = os.path.join('./opt', args.folder)
    if not os.path.exists(target_path):
        print(f"错误: 找不到文件夹 {target_path}")
        return

    print(f"正在搜索 {target_path} 下的文件...")
    
    # 2. 只搜索指定文件夹下的 HMCResult 文件
    search_pattern = os.path.join(target_path, '**/*IsingHMCresult*.hdf5')
    files = glob.glob(search_pattern, recursive=True)
    
    # 按 epoch 数字排序
    def extract_epoch(filename):
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    files = sorted(files, key=extract_epoch)
    print(f"找到 {len(files)} 个文件。开始处理...")

    if not files:
        print("未找到任何 .hdf5 文件，请检查文件夹名称是否正确。")
        return

    # 3. 遍历文件读取数据
    epochs = []
    m_firsts = []  # 第一张图的 |M|
    m_means = []   # 平均 |M|
    
    for file_path in files:
        try:
            epoch = extract_epoch(file_path)
            if epoch == -1: continue

            with h5py.File(file_path, 'r') as f:
                # --- 情况 A: 包含图像数据 'X' ---
                if 'X' in f.keys():
                    X = np.array(f['X'])
                    spins = np.sign(X) # 离散化
                    
                    # 第一张图的 |M|
                    m1 = np.abs(np.mean(spins[0]))
                    # 所有图的平均 |M|
                    m_per_sample = np.abs(np.mean(spins, axis=(1, 2, 3)))
                    m_avg = np.mean(m_per_sample)
                    
                    epochs.append(epoch)
                    m_firsts.append(m1)
                    m_means.append(m_avg)
                    print(f"Epoch {epoch}: Image Data -> |M|={m_avg:.4f}")

                # --- 情况 B: 只有 'XOBS' 记录 ---
                elif 'XOBS' in f.keys():
                    xobs = np.array(f['XOBS'])
                    if xobs.shape[1] >= 2:
                        m_val = np.mean(xobs[:, 1]) # 第2列通常是 |M|
                        epochs.append(epoch)
                        m_firsts.append(m_val)
                        m_means.append(m_val)
                        print(f"Epoch {epoch}: Record Data -> |M|={m_val:.4f}")
                
        except Exception as e:
            print(f"读取文件 {file_path} 出错: {e}")

    # 4. 绘图
    if not epochs:
        print("未提取到有效数据。")
        return

    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, m_firsts, 'o--', color='orange', alpha=0.3, label='Snapshot |M|')
    plt.plot(epochs, m_means, 's-', color='blue', linewidth=2, label='Average |M|')
    
    # 参考线
    plt.axhline(y=1.0, color='red', linestyle=':', label='Ordered (1.0)')
    plt.axhline(y=0.0, color='green', linestyle=':', label='Disordered (0.0)')
    plt.axhspan(0.5, 0.7, color='yellow', alpha=0.1, label='Critical Region')

    plt.xlabel('Epoch')
    plt.ylabel('Magnetization |M|')
    plt.title(f'Magnetization Curve for: {args.folder}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片，文件名包含文件夹名
    save_name = f'magnetization_{args.folder}.png'
    plt.savefig(save_name)
    print(f"绘图完成！图片已保存为: {save_name}")

if __name__ == "__main__":
    get_magnetization_curve()