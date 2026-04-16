import torch
import numpy as np
import h5py
import argparse
import os
import glob
import matplotlib.pyplot as plt

import train
import flow

def main():
    parser = argparse.ArgumentParser(description='Visualize marginal distributions layer by layer')
    parser.add_argument("-folder", required=True, help="Path to the saved model folder (e.g., ./opt/16Ising/)")
    parser.add_argument("-batch", type=int, default=10000, help="Batch size (number of configurations to sample)")
    parser.add_argument("-cuda", type=int, default=-1, help="GPU ID to use, -1 for CPU")
    args = parser.parse_args()

    rootFolder = args.folder
    if rootFolder[-1] != '/':
        rootFolder += '/'

    # 1. 从 parameters.hdf5 中读取模型超参数
    print(f"Loading parameters from {rootFolder}parameters.hdf5 ...")
    with h5py.File(rootFolder + "parameters.hdf5", "r") as f:
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        nlayers = int(np.array(f["nlayers"])) # 单个解耦器内部的 RealNVP 层数
        nmlp = int(np.array(f["nmlp"]))
        nhidden = int(np.array(f["nhidden"]))
        nrepeat = int(np.array(f["nrepeat"]))
        try:
            depthMERA = int(np.array(f["depthMERA"]))
            if depthMERA == -1: depthMERA = None
        except:
            depthMERA = None

    device = torch.device("cpu" if args.cuda < 0 else "cuda:" + str(args.cuda))
    dtype = torch.float32

    # NeuralRG 中 Ising 模型默认带有对称化包装
    def op(x): return -x
    sym = [op]

    # 2. 初始化模型架构
    fw = train.symmetryMERAInit(L, d, nlayers, nmlp, nhidden, nrepeat, sym, device, dtype, name="Visualizer", depthMERA=depthMERA)

    # 3. 寻找并加载最新训练好的权重
    saving_files = glob.glob(rootFolder + 'savings/*.saving')
    if not saving_files:
        raise FileNotFoundError(f"No .saving files found in {rootFolder}savings/")
    name = max(saving_files, key=os.path.getctime)
    print(f"Loading weights from {name} ...")
    # fw.load(torch.load(name, map_location=device))
    # 修改后 (尝试加上 weights_only=False 以兼容旧版序列化)
    # fw.load(torch.load(name, map_location=device, weights_only=False))
        # 替换为以下逻辑：
    state_dict = torch.load(name, map_location=device, weights_only=False)
    # 如果 fw 是一个 torch.nn.Module，直接用 load_state_dict
    if hasattr(fw, 'load_state_dict'):
        fw.load_state_dict(state_dict, strict=False)
    else:
        # 如果 fw 是自定义的包装类，可能需要修改它的 load 方法
        fw.flow.load_state_dict(state_dict, strict=False) 
    
    print("Caution: Loaded with strict=False. Some weights might be missing!")

    # 提取底层的 MERA 网络（剥离 Symmetrized 包装器以进行精细操作）
    mera_flow = fw.flow if hasattr(fw, 'flow') else fw
    
    # 这里的 len(layerList) 就是你所说的 ~log_N 个解耦器的数量
    num_disentanglers = len(mera_flow.layerList)
    print(f"Total Disentangler Blocks (RG steps): {num_disentanglers}")

    mera_flow.eval()
    with torch.no_grad():
        # 初始化最顶层的纯高斯潜变量空间: (Batch, Channel=1, L, L)
        z = torch.randn(args.batch, 1, L, L).to(device)

        # 准备画布，为每一个解耦器画一张图
        fig, axes = plt.subplots(num_disentanglers, 1, figsize=(8, 3 * num_disentanglers))
        if num_disentanglers == 1: axes = [axes]

        print("Generating layer by layer...")
        
        # 4. 手动步进执行 Inverse Flow (从宏观顶层向微观物理层生成)
        # 循环顺序从 top block (no最大) 到 bottom block (no=0)
        for idx, no in enumerate(reversed(range(num_disentanglers))):
            
            # 使用 im2col.dispatch 切出当前尺度下要被更新的局部块 (2x2 patches)
            # 在中间层，切出来的 z_ 自然包含了上一层生成的宏观变量和尚未触碰的高斯噪声（完美契合你的 1+3 逻辑）
            z, z_ = flow.hierarchy.im2col.dispatch(mera_flow.indexI[no], mera_flow.indexJ[no], z)

            # 通过当前的解耦器进行逆向生成 (4输入 -> 4输出)
            z_, logProbability = mera_flow.layerList[no].inverse(z_.reshape(-1, 1, 2, 2))

            # 将更新后的非高斯变量塞回完整的 L x L 晶格中
            z = flow.hierarchy.im2col.collect(mera_flow.indexI[no], mera_flow.indexJ[no], z, z_)

            # --- 提取边缘分布 ---
            # z_ 包含了这个尺度下刚刚生成的全部变量，我们将其拍平(flatten)，即可得到单点边缘分布
            active_vars = z_.detach().cpu().numpy().flatten()

            # 绘制直方图
            ax = axes[idx]
            ax.hist(active_vars, bins=150, density=True, alpha=0.75, color='teal', edgecolor='black', linewidth=0.2)
            
            # 标注信息
            ax.set_title(f"RG Step {idx+1} | Disentangler Block ID: {no}")
            ax.set_xlabel("Variable Value $x$")
            ax.set_ylabel("$p(x)$")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(rootFolder, "marginal_distributions_layer_by_layer.pdf")
        plt.savefig(save_path)
        print(f"Distributions plotted and saved to: {save_path}")
        # plt.show() # 如果在带界面的环境下运行，可以取消注释
        
if __name__ == "__main__":
    main()