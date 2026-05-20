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
L_TARGET = 32  # default; overridden by -L CLI argument

# --- 1. 理论值: 离散 (Ising 自旋) 与 连续 (Hubbard-Stratonovich 场) 两套热力学量 ---
#
# 两套图像，每套都有 自由能 / 能量 / 熵 三个量，单位统一用 nat (与训练 loss 同单位):
#
#   F = -lnZ            (自由能, 即 reverse-KL loss 的理论极小)
#   E = U / T           (能量项, nat 单位; U 为物理自旋能量)
#   S                   (熵, nat 单位)
#   恒等式 (每套图像内部):   F = E - S      <=>   T*S = U + T*lnZ
#
# 离散 <-> 连续 的唯一差别是 HS 高斯归一化常数 fix:
#   lnZ_continuous = lnZ_discrete + fix   =>   F_discrete - F_continuous = fix
#
# 连续图像的 "能量" 是 HS 作用量 A(x) = ½ xᵀK⁻¹x - Σ log cosh(x_i),
#   ⟨A⟩ = E_p[A],   S_continuous = ⟨A⟩ + lnZ_continuous = H(p_HS)
#   (H(p_HS) 即 forward-KL / MLE loss 在 HS 样本上的理论极小)

def _build_K(L, T):
    """构造 ising.py 中同款 K 矩阵 (近邻 Adj / T + 对角 offset 保正定)。"""
    from scipy.linalg import eigh
    N = L * L
    Adj = np.zeros((N, N))
    for i in range(N):
        r, c = divmod(i, L)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            Adj[i, ((r + dr) % L) * L + (c + dc) % L] = 1.0
    K_raw = Adj / T
    offset = 0.1 - eigh(K_raw, eigvals_only=True).min()
    return K_raw + np.eye(N) * offset


def _continuous_energy_entropy(L, T_exact, lnZ_continuous, mcmc_dir):
    """用 HS 样本蒙特卡洛估计 连续图像 的 ⟨A⟩ 与 S_continuous = H(p_HS)。

    需要 data/mcmc_data/hs_L{L}_T*.pt。缺失则返回 (None, None)。
    """
    from scipy.linalg import inv
    files = glob.glob(os.path.join(mcmc_dir, f"hs_L{L}_T*.pt"))
    target = None
    for fp in files:
        m = re.search(r"_T([\d.]+)_", fp)
        if m and math.isclose(float(m.group(1)), T_exact, abs_tol=1e-3):
            target = fp
            break
    if target is None:
        return None, None
    try:
        x = torch.load(target, weights_only=True).reshape(-1, L * L).numpy()
        x = x[:8000]  # 8k 样本足够 (标准误 < 0.1 nat)
        Kinv = inv(_build_K(L, T_exact))
        A = 0.5 * np.sum((x @ Kinv) * x, axis=1) - np.log(np.cosh(x)).sum(axis=1)
        Ec = float(A.mean())
        Sc = Ec + lnZ_continuous  # = H(p_HS)
        return Ec, Sc
    except Exception as e:
        print(f"连续图像能量/熵计算失败 ({target}): {e}")
        return None, None


def get_theoretical_values(exact_path, L, target_t, mcmc_dir):
    """返回目标温度下两套图像的全部热力学量 (nat 单位)。"""
    if not os.path.exists(exact_path):
        print(f"警告: 找不到理论值文件 {exact_path}")
        return None

    content = open(exact_path, "r").read()
    match = re.search(rf"Ising n\s*=\s*{L}.*?\| T.*?\|(.*?)(?:\n\n|\n#|$)",
                      content, re.DOTALL)
    if not match:
        return None

    t_list, lnz_list, fix_list = [], [], []
    for row in match.group(1).strip().split("\n"):
        if "---" in row:
            continue
        cols = [c.strip() for c in row.split("|") if c.strip()]
        if len(cols) >= 3:
            try:
                t_list.append(float(cols[0]))
                lnz_list.append(float(cols[1]))
                fix_list.append(float(cols[2]))
            except ValueError:
                continue
    if not t_list:
        return None

    o = np.argsort(t_list)
    t_arr = np.array(t_list)[o]
    lnz_arr = np.array(lnz_list)[o]
    fix_arr = np.array(fix_list)[o]

    i = int(np.argmin(np.abs(t_arr - target_t)))
    if not math.isclose(t_arr[i], target_t, abs_tol=1e-3):
        return None
    T = t_arr[i]

    # U_discrete = T^2 d(lnZ_d)/dT, 用相邻点中心差分 (np.gradient 在非均匀网格、
    # 临界点附近的加权会失真)。
    if 0 < i < len(t_arr) - 1:
        dlnz = (lnz_arr[i + 1] - lnz_arr[i - 1]) / (t_arr[i + 1] - t_arr[i - 1])
    elif i == 0:
        dlnz = (lnz_arr[1] - lnz_arr[0]) / (t_arr[1] - t_arr[0])
    else:
        dlnz = (lnz_arr[-1] - lnz_arr[-2]) / (t_arr[-1] - t_arr[-2])

    lnZ_d = float(lnz_arr[i])
    fix = float(fix_arr[i])
    lnZ_c = lnZ_d + fix

    U_d = (T ** 2) * dlnz                 # 物理自旋能量 (能量单位)
    E_d = U_d / T                          # 能量项 (nat)
    F_d = -lnZ_d                           # 离散自由能 (nat)
    S_d = E_d - F_d                        # 离散熵 (nat);  T*S_d = U_d + T*lnZ_d
    F_c = -lnZ_c                           # 连续自由能 (nat) = reverse-KL loss 极小

    E_c, S_c = _continuous_energy_entropy(L, T, lnZ_c, mcmc_dir)  # nat

    return {
        "T": T, "fix": fix,
        "F_d": F_d, "E_d": E_d, "S_d": S_d, "U_d": U_d, "TS_d": T * S_d,
        "F_c": F_c, "E_c": E_c, "S_c": S_c,
    }

# --- 2. 针对指定温度扫描 data 目录提取实验值 ---
def collect_experimental_results_for_T(data_path, target_t, L=None):
    results = {}

    if not os.path.exists(data_path):
        print(f"警告: 找不到数据目录 {data_path}")
        return results

    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    target_t_str = f"{target_t:g}"

    for folder in folders:
        # 提取 L、温度和方法名。例如: 32Ising_T2.3_nsym_HP
        match = re.search(r"(\d+)Ising_T(\d+\.\d+)_(\w+)", folder)
        if not match: continue

        folder_l = int(match.group(1))
        folder_t = float(match.group(2))

        if L is not None and folder_l != L:
            continue  # 跳过不符合目标 L 的文件夹

        # 允许 0.001 的误差，这样 2.269 就能匹配 2.269185...
        if not math.isclose(folder_t, target_t, abs_tol=1e-3):
            continue  # 跳过不符合目标温度的文件夹

        method = match.group(3)
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
                
                # 连续图像 的 能量/熵 (nat 单位, reverse-KL 下 flow 采样测得):
                #   ENERGY  = E_q[-log p_unnorm(x)] = ⟨A⟩   -> E_c
                #   ENTROPY = -E_q[log q(x)] = H(q)         -> S_c
                #   恒等式:  LOSS = ENERGY - ENTROPY        (F_c = E_c - S_c)
                corr_energy = "N/A"
                corr_entropy = "N/A"

                if "ENERGY" in rf:
                    full_energy = np.array(rf["ENERGY"]).flatten()
                    if min_idx < len(full_energy):
                        corr_energy = float(full_energy[min_idx])

                if "ENTROPY" in rf:
                    full_entropy = np.array(rf["ENTROPY"]).flatten()
                    if min_idx < len(full_entropy):
                        corr_entropy = float(full_entropy[min_idx])

                results[method] = {
                    "min_loss": min_loss,
                    "energy": corr_energy,     # E_c (nat)
                    "entropy": corr_entropy,   # S_c (nat)
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


def _is_data_driven(method):
    return "dataDriven" in method or "datadriven" in method.lower()


def _fmt_table(rows, col_aligns):
    """Format a list-of-lists as an aligned markdown pipe table."""
    n_cols = len(rows[0])
    widths = [max(len(row[i]) for row in rows) for i in range(n_cols)]

    def fmt_row(row):
        cells = []
        for i, val in enumerate(row):
            w = widths[i]
            cells.append(val.ljust(w) if col_aligns[i] == "left" else val.center(w))
        return "| " + " | ".join(cells) + " |"

    sep_cells = []
    for i, align in enumerate(col_aligns):
        w = widths[i]
        if align == "left":
            sep_cells.append(":" + "-" * (w - 1))
        else:
            sep_cells.append(":" + "-" * (w - 2) + ":")
    sep = "| " + " | ".join(sep_cells) + " |"

    out = [fmt_row(rows[0]), sep]
    for row in rows[1:]:
        out.append(fmt_row(row))
    return out


# --- 3. 生成 Markdown 报告 (6 列热力学分解表) ---
def _classify(method):
    """返回方法的 loss 对应哪一列。

    - reverse-KL (sym/nsym/hp/wt...)     -> 'F_c'  (loss = -lnZ_continuous)
    - HS 样本上的 forward-KL/MLE          -> 'S_c'  (loss = H(p_HS) = S_continuous)
    - 离散自旋(去量化)上的 MLE            -> 'S_disc_dequant' (目标分布不同, 单列标注)
    """
    m = method.lower()
    if "datadriven" not in m:
        return "F_c"
    return "S_c" if "hs" in m else "S_disc_dequant"


def generate_report_for_T(target_t, L):
    os.makedirs("analyzers", exist_ok=True)
    output_file = f"analyzers/loss_report_L{L}_T{target_t:g}.md"
    target_t_str = f"{target_t:g}"
    mcmc_dir = os.path.join(DATA_DIR, "mcmc_data")

    th = get_theoretical_values(EXACT_FILE, L, target_t, mcmc_dir)
    if th is None:
        print(f"未找到 L={L}, T={target_t_str} 的理论值。")
        return

    experiment_data = collect_experimental_results_for_T(DATA_DIR, target_t, L=L)
    if not experiment_data:
        print(f"未找到 T={target_t_str} 的实验数据。")
        return

    def f6(v):
        return f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else "N/A"

    def fb(v):
        return f"**{f6(v)}**" if isinstance(v, (int, float, np.floating)) else "N/A"

    # 列: Method | F_d | E_d | S_d | F_c | E_c | S_c
    header = ["Method",
              "F discrete (-lnZ_d)", "E discrete (U/T)", "S discrete",
              "F cont. (-lnZ_c)", "E cont. (<A>)", "S cont. (H)"]
    col_aligns = ["left"] + ["center"] * 6
    rows = [header]

    # --- 理论行: 6 个量全填 ---
    rows.append([
        "**Exact (theory)**",
        fb(th["F_d"]), fb(th["E_d"]), fb(th["S_d"]),
        fb(th["F_c"]), fb(th["E_c"]), fb(th["S_c"]),
    ])

    # --- MCMC 基准: 只有离散物理能量 (放在 E discrete 列, 与理论同列可比) ---
    mcmc_U = get_mcmc_energy(mcmc_dir, L, target_t)
    mcmc_E_d = mcmc_U / th["T"] if isinstance(mcmc_U, (int, float, np.floating)) else "N/A"
    rows.append([
        "**MCMC baseline (Wolff)**",
        "N/A", fb(mcmc_E_d), "N/A", "N/A", "N/A", "N/A",
    ])

    # --- 各方法: loss 只填它真正对应的那一列 ---
    notes = []
    for method in sorted(experiment_data.keys()):
        ml = experiment_data[method]["min_loss"]
        col = _classify(method)
        cells = ["N/A"] * 6  # F_d E_d S_d F_c E_c S_c
        if col == "F_c":
            # reverse-KL: flow 采样, 三个连续量都测得 (F_c = E_c - S_c)
            cells[3] = f6(ml)                                  # F_c = loss
            cells[4] = f6(experiment_data[method]["energy"])   # E_c = ⟨A⟩
            cells[5] = f6(experiment_data[method]["entropy"])  # S_c = H(q)
        elif col == "S_c":
            cells[5] = f6(ml)            # HS-MLE loss -> 连续熵列 (= H(p_HS))
        else:  # S_disc_dequant
            cells[5] = f6(ml) + " †"     # 去量化离散 MLE: 标注, 目标分布不同
            notes.append(method)
        rows.append([method] + cells)

    # --- 图例 / 物理量换算 ---
    Td = th["T"]
    legend = [
        "",
        "## How to read this table",
        "",
        "All numbers are in **nats** (same unit as the training loss). Two thermodynamic",
        "pictures, each with free energy / energy / entropy:",
        "",
        "- **Discrete** — the ±1 Ising spins. `F_d = -lnZ_d`, `E_d = U_d/T`, `S_d`.",
        "- **Continuous** — the Hubbard-Stratonovich field x. `F_c = -lnZ_c`,",
        "  `E_c = <A>` with action `A(x)=½xᵀK⁻¹x − Σ log cosh x_i`, `S_c = <A> + lnZ_c`.",
        "",
        "Identities (hold within each picture, and across):",
        "",
        "```",
        "  per picture :  F = E - S          (i.e. T*S = U + T*lnZ)",
        "  across      :  F_d - F_c = fix    (HS Gaussian normalisation)",
        "```",
        "",
        f"At T = {Td:.6f} :  fix = {th['fix']:.4f}   "
        f"(check: F_d - F_c = {th['F_d'] - th['F_c']:.4f})",
        "",
        "Physical (energy units) for the discrete picture:",
        f"`U_d = {th['U_d']:.4f}`,  `T*S_d = {th['TS_d']:.4f}`  "
        f"(MCMC `U_d = {mcmc_U if not isinstance(mcmc_U, str) else 'N/A'}`).",
        "",
        "**Which column each method's loss lands in** (and why it is comparable to the",
        "theory value *in that same column*):",
        "",
        "- reverse-KL runs (`sym`, `nsym`, …): the flow is sampled, so **all three**",
        "  continuous columns are measured — `F cont. = loss`, `E cont. = E_q[<A>]`,",
        "  `S cont. = H(q)`, satisfying `F_c = E_c - S_c`. Each is directly comparable",
        "  to the theory value in the same column (gap = how far the flow is from p_HS).",
        "- HS data-driven (`hs_dataDriven`): loss → **S cont.** "
        "(MLE minimum = `H(p_HS) = S_c`).",
    ]
    if notes:
        legend += [
            "",
            f"† `{', '.join(notes)}` is MLE on **dequantised discrete spins**, a different",
            "  target distribution than the HS field — its loss is *not* comparable to the",
            "  `S cont.` theory value or to `hs_dataDriven`. Shown for reference only.",
        ]

    header_lines = [
        f"# Ising L={L} Thermodynamic Report (T={target_t_str})",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    table_lines = _fmt_table(rows, col_aligns)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines + table_lines + legend) + "\n")

    print(f"Report written: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate minimum loss report for a specific temperature.")
    parser.add_argument("-t", "--temp", type=float, required=True, help="Temperature T (e.g. 2.269)")
    parser.add_argument("-L", "--lattice", type=int, default=32, help="Lattice size L (e.g. 8, 16, 32)")

    args = parser.parse_args()

    generate_report_for_T(args.temp, args.lattice)