import matplotlib.pyplot as plt
import re

# 1. 配置文件路径（请替换为你实际的日志文件名）
log_file_path = "32Ising_T2.3_hp_continue_34503576.log"

epochs = []
obs_z_vals = []
obs_x_vals = []

current_epoch = -1

# 2. 解析日志文件
with open(log_file_path, 'r') as file:
    for line in file:
        # 捕获当前的 epoch 数值
        if line.startswith("epoch:"):
            match = re.search(r"epoch:\s*(\d+)", line)
            if match:
                current_epoch = int(match.group(1))
        
        # 捕获 obs_z
        elif line.startswith("accratio_z:"):
            match = re.search(r"obs_z:\s*([0-9\.\-]+)", line)
            if match and current_epoch != -1:
                epochs.append(current_epoch)
                obs_z_vals.append(float(match.group(1)))
                
        # 捕获 obs_x (横场/相干项)
        elif line.startswith("accratio_x:"):
            match = re.search(r"obs_x:\s*([0-9\.\-]+)", line)
            if match and current_epoch != -1:
                obs_x_vals.append(float(match.group(1)))

# 3. 绘制趋势图
plt.figure(figsize=(10, 6))

# 画出 obs_z 的下降漂移曲线
plt.plot(epochs, obs_z_vals, marker='o', markersize=3, linestyle='-', color='#1f77b4', label='obs_z (Magnetization)', alpha=0.8)

# 可选：如果也想看 obs_x 的变化，可以取消下面这行的注释
# plt.plot(epochs, obs_x_vals, marker='s', markersize=3, linestyle='-', color='#ff7f0e', label='obs_x', alpha=0.8)

plt.title('Order Parameter (obs_z) Drift Over Time at T=2.3\n(Critical Slowing Down Observation)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Expectation Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 保存图片或直接显示
plt.tight_layout()
plt.savefig("obs_drift_trend.png", dpi=300)
print("Plot saved as obs_drift_trend.png")
# plt.show() # 如果在本地带 GUI 环境运行，可以取消注释直接弹窗