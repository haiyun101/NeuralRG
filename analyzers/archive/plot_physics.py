import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Probe Mode Collapse via ZOBS and ZACC")
    parser.add_argument('-folder', default='./opt/32Ising_T2.4_unsy/', help='Folder containing the record files')
    args = parser.parse_args()

    record_path = os.path.join(args.folder, "records")
    files = [f for f in os.listdir(record_path) if f.endswith('.hdf5') and 'Record' in f]
    
    if not files:
        print(f"Error: No HDF5 record file found in {record_path}")
        return
        
    # 假设读取第一个找到的 record 文件
    file_to_read = os.path.join(record_path, files[0])
    print(f"Loading data from: {file_to_read}")

    with h5py.File(file_to_read, "r") as f:
        # 直接读取我们刚刚确认存在的 Keys
        loss = np.array(f['LOSS'])
        zobs = np.array(f['ZOBS'])
        zacc = np.array(f['ZACC'])
        
        # 创建 3 个纵向排列的子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # 图 1: Loss (呈现假性繁荣)
        ax1.plot(loss, color='tab:red')
        ax1.set_ylabel('Total LOSS', color='tab:red', fontweight='bold')
        ax1.set_title('Evidence of Mode Collapse (unsymmetrized model)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 图 2: ZOBS 磁化强度 (揭露铁磁态坍缩)
        ax2.plot(zobs, color='tab:purple')
        ax2.set_ylabel('Magnetization (ZOBS)', color='tab:purple', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Theoretical Paramagnetic (0)')
        ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # 图 3: ZACC 接受率 (物理视角的终极宣判)
        ax3.plot(zacc, color='tab:orange')
        ax3.set_ylabel('HMC Acceptance (ZACC)', color='tab:orange', fontweight='bold')
        ax3.set_xlabel('Epochs / Steps', fontweight='bold')
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()
        save_path = os.path.join(args.folder, "pic", "Collapse_Proof.pdf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    main()