import os
import glob
import argparse
import h5py
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import train
import source

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Plot disentangler joint distributions.')
    parser.add_argument("-folder", type=str, required=True, help='Path to the model folder under opt/ (e.g., ./opt/16Ising/)')
    parser.add_argument("-layer", type=int, default=0, help='Which disentangler layer to visualize (default: 0)')
    parser.add_argument("-samples", type=int, default=10000, help='Number of samples to generate (default: 10000)')
    args = parser.parse_args()

    rootFolder = args.folder
    if rootFolder[-1] != '/':
        rootFolder += '/'

    # 2. 读取 parameters.hdf5 以重构完全相同的模型结构
    param_file = rootFolder + "parameters.hdf5"
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Cannot find {param_file}. Please check the folder path.")

    with h5py.File(param_file, "r") as f:
        L = int(np.array(f["L"]))
        d = int(np.array(f["d"]))
        nlayers = int(np.array(f["nlayers"]))
        nmlp = int(np.array(f["nmlp"]))
        nhidden = int(np.array(f["nhidden"]))
        nrepeat = int(np.array(f["nrepeat"]))
        # 兼容旧版本可能没有 depthMERA 的情况
        depthMERA = int(np.array(f["depthMERA"])) if "depthMERA" in f else None 
        # 尝试读取是否使用了 symmetry (默认如果 parameters 里没有存，可以通过文件夹名字推断或默认开启)
        # 这里假设你训练时开启了 symmetry
        use_symmetry = True 

    print(f"Loaded parameters: L={L}, d={d}, nlayers={nlayers}, nmlp={nmlp}, nhidden={nhidden}")

    # 3. 初始化模型
    device = torch.device('cpu')
    dtype = torch.float32

    if use_symmetry:
        def op(x): return -x
        sym = [op]
    else:
        sym = None

    model = train.symmetryMERAInit(L, d, nlayers, nmlp, nhidden, nrepeat, sym, device, dtype, depthMERA=depthMERA)

    # 4. 自动寻找最新的 .saving 文件
    # 搜索 savings 文件夹下所有的 .saving 文件
    search_pattern = os.path.join(rootFolder, 'savings', '*.saving')
    saving_files = glob.glob(search_pattern)
    
    if not saving_files:
        raise FileNotFoundError(f"No .saving files found in {rootFolder}savings/")
    
    # 按照创建时间排序，取最后一个（最新）
    latest_saving = max(saving_files, key=os.path.getctime)
    print(f"Found latest saving file: {latest_saving}")

    # 加载权重
    model.load(torch.load(latest_saving, map_location=device))

    # 5. 提取指定的解耦器层
    # 如果使用了 Symmetrized 包装器，真正的 MERA 结构在 model.flow 中
    mera_flow = model.flow if hasattr(model, 'flow') else model
    
    try:
        target_layer = mera_flow.layerList[args.layer]
    except IndexError:
        raise IndexError(f"Layer {args.layer} is out of bounds. The model has {len(mera_flow.layerList)} layers.")

    print(f"Extracting layer {args.layer}: {target_layer.name}")

    # 6. 生成高斯噪声并进行反向映射 (Inverse Flow)
    print(f"Generating {args.samples} samples...")
    # 对于二维系统，MERA 的 kernelShape 通常是 [2, 2]，channel 默认为 1
    z = torch.randn(args.samples, 1, 2, 2, dtype=dtype, device=device)
    
    with torch.no_grad():
        x, _ = target_layer.inverse(z)

    # 7. 绘图
    x_flat = x.view(args.samples, 4).cpu().numpy()
    df = pd.DataFrame(x_flat, columns=['x1', 'x2', 'x3', 'x4'])

    print("Plotting joint distributions...")
    # 使用 corner=True 只画下三角，显得更简洁
    g = sns.pairplot(df, kind="kde", corner=True, 
                     plot_kws={'alpha': 0.7, 'levels': 5, 'fill': True})
    
    # 构建包含模型信息和层数的标题
    model_info = f"Ising {L}x{L} | Layers:{nlayers} MLP:{nmlp}x{nhidden}"
    title = f"Disentangler Layer {args.layer} Joint Distribution\n({model_info})"
    g.fig.suptitle(title, y=1.02, fontsize=14)

    # 8. 保存图片到指定的文件夹的 pic/ 目录下
    pic_folder = os.path.join(rootFolder, "pic")
    if not os.path.exists(pic_folder):
        os.makedirs(pic_folder)
        
    save_path = os.path.join(pic_folder, f"layer_{args.layer}_distribution.png")
    g.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Plot successfully saved to: {save_path}")

if __name__ == "__main__":
    import numpy as np # 确保 numpy 在 main 里面可用
    main()