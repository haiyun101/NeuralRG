# import numpy as np
# import matplotlib.pyplot as plt

# def wolff_step(lattice, prob_add):
#     """
#     执行单次 Wolff 集群更新
#     """
#     L = lattice.shape[0]
    
#     # 1. 随机选择一个种子自旋
#     i, j = np.random.randint(0, L), np.random.randint(0, L)
#     seed_spin = lattice[i, j]
    
#     # 2. 翻转种子自旋，并将其压入栈中
#     lattice[i, j] = -seed_spin
#     stack = [(i, j)]
    
#     # 3. 集群生长
#     while stack:
#         cx, cy = stack.pop()
        
#         # 获取周期性边界条件下的 4 个邻居
#         neighbors = [
#             ((cx + 1) % L, cy),
#             ((cx - 1) % L, cy),
#             (cx, (cy + 1) % L),
#             (cx, (cy - 1) % L)
#         ]
        
#         for nx, ny in neighbors:
#             # 如果邻居与种子自旋的初始方向相同
#             if lattice[nx, ny] == seed_spin:
#                 # 以概率 P_add 将其加入集群
#                 if np.random.rand() < prob_add:
#                     lattice[nx, ny] = -seed_spin # 翻转以标记已加入集群
#                     stack.append((nx, ny))

# def calc_correlation(lattice):
#     """
#     计算沿 x 和 y 轴的平均空间自旋关联函数 G(r) = <s_i s_{i+r}>
#     """
#     L = lattice.shape[0]
#     max_r = L // 2
#     corr = np.zeros(max_r)
    
#     for r in range(max_r):
#         # 沿 x 轴平移 r 个单位求内积平均，再沿 y 轴做同样操作
#         corr_x = np.mean(lattice * np.roll(lattice, shift=r, axis=0))
#         corr_y = np.mean(lattice * np.roll(lattice, shift=r, axis=1))
#         corr[r] = (corr_x + corr_y) / 2.0
        
#     return corr

# def run_wolff_simulation(L=32, T=2.3, J=1.0, steps=5000, burn_in=1000):
#     """
#     运行 Wolff MCMC 模拟
#     """
#     print(f"开始 Wolff MCMC 模拟 (L={L}, T={T})...")
    
#     # 初始化随机晶格
#     lattice = np.random.choice([-1, 1], size=(L, L))
    
#     # 计算集群生长的接受概率: P = 1 - exp(-2J/T)
#     prob_add = 1.0 - np.exp(-2.0 * J / T)
    
#     # 热化阶段 (Burn-in)，让系统达到平衡
#     for _ in range(burn_in):
#         wolff_step(lattice, prob_add)
        
#     # 测量阶段
#     m_history = []
#     corr_history = []
    
#     for step in range(steps):
#         wolff_step(lattice, prob_add)
        
#         # 记录磁化强度 m
#         m = np.sum(lattice) / (L * L)
#         m_history.append(m)
        
#         # 记录关联函数 G(r)
#         corr_history.append(calc_correlation(lattice))
        
#         if (step + 1) % 1000 == 0:
#             print(f"已完成 {step + 1}/{steps} 步采样")
            
#     avg_corr = np.mean(corr_history, axis=0)
#     return np.array(m_history), avg_corr

# def generate_sample_images(L=32, T=2.3, J=1.0, num_samples=16, steps_between=50):
#     """
#     使用 Wolff 算法生成独立的晶格位形样本并画成网格图
#     """
#     print(f"开始生成 MCMC 采样图集 (L={L}, T={T})...")
    
#     lattice = np.random.choice([-1, 1], size=(L, L))
#     prob_add = 1.0 - np.exp(-2.0 * J / T)
    
#     # 充分热化 (Burn-in)，确保系统进入平衡态
#     for _ in range(1000):
#         wolff_step(lattice, prob_add)
        
#     samples = []
    
#     # 采集独立样本
#     for i in range(num_samples):
#         # 间隔一定步数，降低样本间的自相关性
#         for _ in range(steps_between):
#             wolff_step(lattice, prob_add)
#         # 保存当前晶格的拷贝
#         samples.append(lattice.copy())
        
#     # ==========================================
#     # 开始拼接并绘制网格图
#     # ==========================================
#     fig, axes = plt.subplots(4, 4, figsize=(10, 10))
#     fig.suptitle(f'Exact MCMC Samples for 2D Ising (L={L}, T={T})', fontsize=16)
    
#     for i, ax in enumerate(axes.flatten()):
#         if i < num_samples:
#             # 用灰度图显示：+1 (黑), -1 (白)
#             ax.imshow(samples[i], cmap='Greys', interpolation='nearest', vmin=-1, vmax=1)
#         ax.axis('off') # 隐藏坐标轴
        
#     plt.tight_layout(rect=[0, 0, 1, 0.96]) # 留出标题空间
    
#     save_name = f"MCMC_Samples_L{L}_T{T}.png"
#     plt.savefig(save_name, dpi=300, bbox_inches='tight')
#     print(f"采样图集已保存至: {save_name}")
#     plt.show()

# # ==========================================
# # 运行模拟并绘图
# # ==========================================
# if __name__ == "__main__":
#     L_val = 32
#     T_val = 2.3
    
#     # 运行模拟 (获取 m 的时间序列和平均关联曲线)
#     m_data, avg_correlation = run_wolff_simulation(L=L_val, T=T_val, steps=5000, burn_in=1000)
    
#     # 开始绘图
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     # 1. 绘制磁化强度 m 的分布直方图
#     ax1.hist(m_data, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7)
#     ax1.set_title(f'Magnetization Distribution $p(m)$ at T={T_val}')
#     ax1.set_xlabel('Magnetization $m$')
#     ax1.set_ylabel('Probability Density')
#     ax1.set_xlim(-1.1, 1.1)
#     ax1.grid(True, alpha=0.3)
    
#     # 2. 绘制自旋关联函数 G(r) 曲线
#     r_values = np.arange(L_val // 2)
#     ax2.plot(r_values, avg_correlation, marker='o', linestyle='-', color='tab:red', markersize=6)
#     ax2.set_title(f'Spin Correlation Function $G(r)$ at T={T_val}')
#     ax2.set_xlabel('Distance $r$')
#     ax2.set_ylabel('Correlation $\\langle s_i s_{i+r} \\rangle$')
#     ax2.axhline(0, color='black', linestyle='--', linewidth=1)
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(f"Wolff_MCMC_L{L_val}_T{T_val}.png", dpi=300)
#     print("图表已生成并保存！")
#     plt.show()

#     generate_sample_images(L=32, T=2.3, num_samples=16, steps_between=50)

import numpy as np
import matplotlib.pyplot as plt

def wolff_step(lattice, prob_add):
    """
    执行单次 Wolff 集群更新
    """
    L = lattice.shape[0]
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    seed_spin = lattice[i, j]
    
    lattice[i, j] = -seed_spin
    stack = [(i, j)]
    
    while stack:
        cx, cy = stack.pop()
        neighbors = [
            ((cx + 1) % L, cy),
            ((cx - 1) % L, cy),
            (cx, (cy + 1) % L),
            (cx, (cy - 1) % L)
        ]
        
        for nx, ny in neighbors:
            if lattice[nx, ny] == seed_spin:
                if np.random.rand() < prob_add:
                    lattice[nx, ny] = -seed_spin 
                    stack.append((nx, ny))

def calc_correlation(lattice):
    """
    计算空间自旋关联函数 G(r)
    """
    L = lattice.shape[0]
    max_r = L // 2
    corr = np.zeros(max_r)
    for r in range(max_r):
        corr_x = np.mean(lattice * np.roll(lattice, shift=r, axis=0))
        corr_y = np.mean(lattice * np.roll(lattice, shift=r, axis=1))
        corr[r] = (corr_x + corr_y) / 2.0
    return corr

def calc_energy(lattice, J=1.0):
    """
    计算晶格的总能量 (最近邻相互作用)
    """
    E_x = np.sum(lattice * np.roll(lattice, shift=1, axis=0))
    E_y = np.sum(lattice * np.roll(lattice, shift=1, axis=1))
    return -J * (E_x + E_y)

def run_wolff_simulation(L=32, T=2.3, J=1.0, steps=3000, burn_in=1000):
    """
    运行 Wolff MCMC 并返回可观测物理量及 16 个独立样本
    """
    print(f"  -> 正在运行 MCMC 采样，当前 T = {T:.2f}...")
    lattice = np.random.choice([-1, 1], size=(L, L))
    prob_add = 1.0 - np.exp(-2.0 * J / T)
    
    for _ in range(burn_in):
        wolff_step(lattice, prob_add)
        
    m_history = []
    e_history = []
    corr_history = []
    samples = []
    
    sample_interval = max(1, steps // 16)
    
    for step in range(steps):
        wolff_step(lattice, prob_add)
        
        if step % 5 == 0:
            m_history.append(np.sum(lattice)/ (L * L))
            e_history.append(calc_energy(lattice, J))
            
        if step % 100 == 0:
            corr_history.append(calc_correlation(lattice))
            
        if (step + 1) % sample_interval == 0 and len(samples) < 16:
            samples.append(lattice.copy())
            
    return {
        'm_hist': np.array(m_history),
        'avg_corr': np.mean(corr_history, axis=0),
        'avg_e': np.mean(e_history),
        'samples': samples
    }


if __name__ == "__main__":
    L_val = 32
    J_val = 1.0
    
    # 1. 直接生成均匀的温度列表 (例如从 4.0 降温到 1.5)
    T_min, T_max = 1.5, 4.0
    num_T = 20
    T_list = np.linspace(T_max, T_min, num_T) 
    
    # 对应的 beta 列表 (自动为升序，便于从 0 开始积分)
    betas = 1.0 / T_list 
    
    # 为了热力学积分准确，我们在开头插入 beta=0 (T=infinity) 的点
    betas_int = np.insert(betas, 0, 0.0)
    e_avgs_int = np.zeros(len(betas_int))
    e_avgs_int[0] = 0.0  # T无穷大时，平均能量为 0
    
    sim_results = {}
    
    print("--- 1. 开始温度扫描 (Temperature Sweep) ---")
    for i, T in enumerate(T_list):
        res = run_wolff_simulation(L=L_val, T=T, J=J_val, steps=2000, burn_in=500)
        e_avgs_int[i+1] = res['avg_e']
        sim_results[T] = res

    print("\n--- 2. 计算热力学积分 (Thermodynamic Integration) ---")
    # 数值积分：integral = \int_{0}^{\beta} <e(\beta')> d\beta'
    # 梯形法则支持非均匀间距的 betas_int
    integral = np.zeros(len(betas_int))
    for i in range(1, len(betas_int)):
        integral[i] = integral[i-1] + 0.5 * (betas_int[i] - betas_int[i-1]) * (e_avgs_int[i] + e_avgs_int[i-1])
        
    #整个系统的总自由能：F(beta) = (-L^2 * ln(2) + integral) / beta
    total_spins = L_val ** 2
    f_vals = (-total_spins * np.log(2) + integral[1:]) / betas
    e_vals = e_avgs_int[1:]
    ts_vals = f_vals - e_vals  # 总的 -TS 项

    print("\n--- 3. 生成图表 ---")
    
    # A) 为每个温度生成单独的三合一拼接图
    for i, T in enumerate(T_list):
        data = sim_results[T]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # 在三合一子图的标题部分：
        title_str = (f'2D Ising Wolff MCMC | T = {T:.2f}\n'
                     f'Total System:  $\\langle E \\rangle$ = {e_vals[i]:.2f}  |  '
                     f'$\\langle F \\rangle$ = {f_vals[i]:.2f}  |  '
                     f'$-T\\langle S \\rangle$ = {ts_vals[i]:.2f}')
        fig.suptitle(title_str, fontsize=15, fontweight='bold')
        
        # 子图 1: 16个独立样本的网格图
        samples = data['samples']
        rows = [np.hstack(samples[j*4:(j+1)*4]) for j in range(4) if len(samples) >= (j+1)*4]
        if rows:
            img_grid = np.vstack(rows)
            axes[0].imshow(img_grid, cmap='Greys', vmin=-1, vmax=1, interpolation='nearest')
        axes[0].axis('off')
        axes[0].set_title('16 Independent Lattice Samples')
        
        # 子图 2: 磁化强度分布
        axes[1].hist(data['m_hist'], bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].set_title('Magnetization Distribution $p(m)$')
        axes[1].set_xlabel('Magnetization $m$')
        axes[1].set_ylabel('Density')
        axes[1].set_xlim(-1.1, 1.1)
        axes[1].grid(True, alpha=0.3)
        
        # 子图 3: 空间关联函数
        r_vals = np.arange(L_val // 2)
        axes[2].plot(r_vals, data['avg_corr'], marker='o', linestyle='-', color='tab:red', markersize=6)
        axes[2].set_title('Spin Correlation Function $G(r)$')
        axes[2].set_xlabel('Distance $r$')
        axes[2].set_ylabel('$\\langle s_i s_{i+r} \\rangle$')
        axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.90]) 
        plt.savefig(f"MCMC_L{L_val}_T{T:.2f}.png", dpi=200)
        plt.close()
        
    print(f"已保存全部 {num_T} 个温度的独立图表。")

    # B) 随温度变化的热力学总结图
    fig_th, ax_th = plt.subplots(figsize=(10, 6))
    
    # 按照温度升序绘制（因为 T_list 是降序的，可以翻转一下顺序让图表横坐标从左到右递增，这里直接 plot 会自动连接）
    ax_th.plot(T_list, e_vals, 'o-', label='Energy $\\langle e \\rangle$', color='tab:blue', linewidth=2)
    ax_th.plot(T_list, f_vals, 's-', label='Free Energy $f$', color='tab:green', linewidth=2)
    ax_th.plot(T_list, ts_vals, '^-', label='$-T \\cdot s$ term', color='tab:orange', linewidth=2)
    
    Tc = 2.0 / np.log(1.0 + np.sqrt(2))
    ax_th.axvline(Tc, color='red', linestyle='--', label=f'Critical $T_c \\approx {Tc:.3f}$')
    
    ax_th.set_xlabel('Temperature $T$', fontsize=12)
    # 在最终总结图的 Y 轴标签部分：
    ax_th.set_ylabel('Total Quantities ($E, F, -TS$)', fontsize=12)
    ax_th.set_title(f'Thermodynamics vs Temperature (L={L_val})', fontsize=14)
    ax_th.legend(fontsize=11)
    ax_th.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f"Thermodynamics_Summary_L{L_val}.png", dpi=300)
    print("\n已保存最终热力学总结图表 (Thermodynamics_Summary_L32.png)。")
    plt.show()