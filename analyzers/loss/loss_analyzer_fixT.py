import os
import h5py
import re
import glob
import json
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import torch

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
def collect_experimental_results_for_T(data_path, target_t, L=None, exclude=None):
    results = {}

    if not os.path.exists(data_path):
        print(f"警告: 找不到数据目录 {data_path}")
        return results

    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    target_t_str = f"{target_t:g}"

    for folder in folders:
        # 提取 L、温度和方法名。例如: 32Ising_T2.3_nsym_HP
        # NOTE: use [\w.-]+ instead of \w+ so folder names containing `-`
        # or `.` (e.g. "..._vp1e-3_...", "..._gc5.0_...") don't get truncated
        # at the first non-word character, which caused method-name
        # collisions (multiple λ values mapped to the same key).
        match = re.search(r"(\d+)Ising_T(\d+\.\d+)_([\w.-]+)", folder)
        if not match: continue

        folder_l = int(match.group(1))
        folder_t = float(match.group(2))

        if L is not None and folder_l != L:
            continue  # 跳过不符合目标 L 的文件夹

        # 允许 0.001 的误差，这样 2.269 就能匹配 2.269185...
        if not math.isclose(folder_t, target_t, abs_tol=1e-3):
            continue  # 跳过不符合目标温度的文件夹

        method = match.group(3)
        if "broken" in method.lower():
            continue  # 跳过显式标记为 BROKEN 的运行 (文件夹名含 BROKEN)
        if exclude and any(x in method for x in exclude):
            continue  # 跳过被排除的方法 (如仍在训练的 bignet)
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
                # 获取 Loss (may include regularizer penalties like VP)
                full_loss = np.array(rf["LOSS"]).flatten()
                # ENTROPY = pure MLE loss = -E_data[log q(x)]
                # For non-regularized runs, ENTROPY == LOSS.
                # For runs with VP or entropy penalties, ENTROPY < LOSS
                # because LOSS = ENTROPY + penalty. ENTROPY is the fair
                # cross-variant metric (a.k.a. "F" in cross-variant tables).
                full_entropy = np.array(rf["ENTROPY"]).flatten() if "ENTROPY" in rf else None

                # Pick best by ENTROPY (pure MLE) so VP-regularized runs
                # are not artificially penalized by their own regularizer
                # in the "best of each mode" comparison. Fall back to
                # LOSS when ENTROPY is unavailable (legacy runs).
                if full_entropy is not None and len(full_entropy) == len(full_loss):
                    min_idx = int(np.argmin(full_entropy))
                else:
                    min_idx = int(np.argmin(full_loss))
                min_loss = float(full_loss[min_idx])

                # Best-200: lowest 200-epoch rolling mean of ENTROPY.
                # Smoothes out lucky-batch spikes → what actually
                # characterizes the trained model. Reduces or eliminates
                # the artifactual negative KL(p‖q) seen when using the
                # single-epoch minimum (which drops below H(p_HS) purely
                # from batch noise).
                # Also record the CENTER epoch of the min-mean window so
                # downstream tier1 / flow_sample_diagnostic can use that
                # checkpoint for consistent physics analysis.
                best_200_entropy = None
                best_200_epoch = None
                if full_entropy is not None and len(full_entropy) >= 200:
                    _cs = np.cumsum(full_entropy)
                    _window_sums = _cs[199:] - np.concatenate([[0.0], _cs[:-200]])
                    _window_means = _window_sums / 200.0
                    _min_start = int(np.argmin(_window_means))
                    best_200_entropy = float(_window_means[_min_start])
                    best_200_epoch = _min_start + 100  # center of the 200-ep window
                elif full_entropy is not None and len(full_entropy) > 0:
                    best_200_entropy = float(full_entropy.mean())
                    best_200_epoch = int(np.argmin(full_entropy))

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

                if full_entropy is not None:
                    if min_idx < len(full_entropy):
                        corr_entropy = float(full_entropy[min_idx])

                # If two folders reduce to the same method name (should not
                # happen after the [\w.-]+ regex fix, but guard anyway),
                # keep the one with lower min_loss.
                existing = results.get(method)
                if existing is not None and existing["min_loss"] <= min_loss:
                    continue
                results[method] = {
                    "min_loss": min_loss,
                    "energy": corr_energy,     # E_c (nat)
                    "entropy": corr_entropy,   # S_c (nat)
                    "best_200_entropy": best_200_entropy,  # smoothed S
                    "best_200_epoch": best_200_epoch,      # center of min-mean window
                    "folder": os.path.join(data_path, folder),
                }

                # Post-hoc flow diagnostic (from analyzers/flow_sample_diagnostic.py).
                # Independent of training mode: model-side <A>_q, H(q), F_c^q,
                # plus both KL directions KL(q||p) and KL(p||q).
                diag_path = os.path.join(data_path, folder, "flow_diagnostic.json")
                if os.path.exists(diag_path):
                    try:
                        with open(diag_path) as df:
                            results[method]["diag"] = json.load(df)
                    except Exception as e:
                        print(f"Warning: failed to read {diag_path}: {e}")

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


# --- 3. 生成 Markdown 报告 (3 列 F/E/S; 离散/连续作为行) ---
def _classify(method):
    """返回方法的 loss 对应哪个量 (F_c / S_c / S_disc_dequant)。

    - HS 连续场上的 forward-KL/MLE (hs_*)    -> 'S_c'  (loss = H(p_HS) = S_continuous)
    - 离散自旋(去量化)上的 MLE (*dataDriven)  -> 'S_disc_dequant' (目标分布不同, 排除)
    - reverse-KL (sym/nsym/hp/wt...)          -> 'F_c'  (loss = -lnZ_continuous)

    注意: hs_* 系列 (hs_dataDriven, hs_bignet, hs_haarPrior, ...) 都是 HS
    forward-KL, 名字里不一定含 'datadriven', 所以先判断 'hs' 前缀。
    """
    m = method.lower()
    if m.startswith("hs"):
        return "S_c"
    if "datadriven" in m:
        return "S_disc_dequant"
    return "F_c"


def generate_report_for_T(target_t, L, exclude=None):
    os.makedirs("analyzers", exist_ok=True)
    output_file = f"analyzers/loss_report_L{L}_T{target_t:g}.md"
    target_t_str = f"{target_t:g}"
    mcmc_dir = os.path.join(DATA_DIR, "mcmc_data")

    th = get_theoretical_values(EXACT_FILE, L, target_t, mcmc_dir)
    if th is None:
        print(f"未找到 L={L}, T={target_t_str} 的理论值。")
        return

    experiment_data = collect_experimental_results_for_T(DATA_DIR, target_t, L=L, exclude=exclude)
    if not experiment_data:
        print(f"未找到 T={target_t_str} 的实验数据。")
        return

    def f6(v):
        return f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else "N/A"

    def fb(v):
        return f"**{f6(v)}**" if isinstance(v, (int, float, np.floating)) else "N/A"

    # 列: Method | Picture | F | E | S  (离散行相邻分组, theory 两行)
    header = ["Method", "Picture", "F (-lnZ)", "E", "S"]
    col_aligns = ["left", "center", "center", "center", "center"]
    rows = [header]

    mcmc_U = get_mcmc_energy(mcmc_dir, L, target_t)
    mcmc_E_d = mcmc_U / th["T"] if isinstance(mcmc_U, (int, float, np.floating)) else "N/A"

    # --- 离散图像: 理论 + MCMC 基准 (相邻, 便于对比) ---
    rows.append(["**Exact (theory)**", "discrete",
                 fb(th["F_d"]), fb(th["E_d"]), fb(th["S_d"])])
    rows.append(["**MCMC baseline (Wolff)**", "discrete",
                 "N/A", fb(mcmc_E_d), "N/A"])

    # --- 连续图像: 理论 + 每种训练模式 loss 最优的一个 run ---
    rows.append(["**Exact (theory)**", "continuous",
                 fb(th["F_c"]), fb(th["E_c"]), fb(th["S_c"])])

    # --- 每种训练模式只保留 loss 最优的一个 run, 保持表格简洁 ---
    rev = {m: d for m, d in experiment_data.items() if _classify(m) == "F_c"}
    fwd = {m: d for m, d in experiment_data.items() if _classify(m) == "S_c"}
    excluded = sorted(m for m in experiment_data if _classify(m) == "S_disc_dequant")

    best_rev = min(rev, key=lambda m: rev[m]["min_loss"]) if rev else None
    best_fwd = min(fwd, key=lambda m: fwd[m]["min_loss"]) if fwd else None

    if best_rev is not None:
        d = experiment_data[best_rev]
        # reverse-KL: flow 采样, 连续图像三个量全测得 (F = E - S)
        rows.append([f"{best_rev}  (best reverse-KL)", "continuous",
                     f6(d["min_loss"]), f6(d["energy"]), f6(d["entropy"])])
    if best_fwd is not None:
        d = experiment_data[best_fwd]
        # HS forward-KL/MLE: loss = 连续熵 S (= H(p_HS))
        rows.append([f"{best_fwd}  (best forward-KL)", "continuous",
                     "N/A", "N/A", f6(d["min_loss"])])

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
        "The two pictures are the **Picture** column (one theory row each); the",
        "`F`/`E`/`S` columns hold `F_d,E_d,S_d` or `F_c,E_c,S_c` per the row's picture.",
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
        "**Only the single best run per training mode is shown** (lowest loss); the",
        "full per-run comparison is in the flow-diagnostic table below. What each",
        "kept run reports:",
        "",
        "- reverse-KL run (`sym`, `nsym`, …): the flow is sampled, so **all three**",
        "  continuous quantities are measured — `F = loss`, `E = E_q[<A>]`,",
        "  `S = H(q)`, satisfying `F = E - S`. Each is directly comparable",
        "  to the continuous theory row (gap = how far the flow is from p_HS).",
        "- HS data-driven (`hs_dataDriven`): loss → **S** in the continuous picture",
        "  (MLE minimum = `H(p_HS) = S_c`).",
    ]
    if excluded:
        legend += [
            "",
            f"Excluded for clarity: `{', '.join(excluded)}` — MLE on **dequantised",
            "  discrete spins**, a different target distribution than the HS field, so",
            "  not comparable to the continuous theory row. (Still appears in the flow",
            "  diagnostic below, which is training-mode agnostic.)",
        ]

    header_lines = [
        f"# Ising L={L} Thermodynamic Report (T={target_t_str})",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]
    table_lines = _fmt_table(rows, col_aligns)

    # --- Flow diagnostic sub-table (from flow_sample_diagnostic.py) ---
    # Independent post-hoc check. Model-side quantities from x ~ q (flow.sample);
    # both KL directions, with KL(p||q) estimated from x ~ p_HS (HS-sample files).
    diag_rows = [["Method", "<A>_q", "H(q)", "F_c^q", "KL(q‖p)", "KL(p‖q)", "epoch"]]
    diag_aligns = ["left", "center", "center", "center", "center", "center", "center"]
    diag_rows.append([
        "**Exact (theory)**",
        fb(th["E_c"]), fb(th["S_c"]), fb(th["F_c"]), "**0**", "**0**", "-",
    ])

    def _kl(v):
        return f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else "N/A"

    for method in sorted(experiment_data.keys()):
        diag = experiment_data[method].get("diag")
        if diag is None:
            continue
        diag_rows.append([
            method,
            f"{diag['EA']:+.4f} ± {diag['sem_EA']:.3f}",
            f"{diag['Hq']:+.4f} ± {diag['sem_Hq']:.3f}",
            f"{diag['Fcq']:+.4f}",
            _kl(diag.get("KL_qp")),
            _kl(diag.get("KL_pq")),
            str(diag["epoch"]),
        ])

    diag_section = []
    if len(diag_rows) > 2:  # at least one method has diagnostic
        diag_section = [
            "",
            "## Flow diagnostic (post-hoc, independent of training mode)",
            "",
            "Model side — sample `x ~ q` from the trained flow:",
            "`<A>_q = E_q[A(x)]`, `H(q) = -E_q[log q(x)]`, `F_c^q = <A>_q - H(q)`.",
            "`F_c^q` is a variational upper bound on `-lnZ_c` (Gibbs).",
            "",
            "**How the two KL directions are obtained — and which training",
            "mode minimizes each:**",
            "```",
            "  KL(q‖p) = E_q[log q - log p] = F_c^q + lnZ_c      (mode-seeking)",
            "    Obtained from FLOW samples x ~ q : draw x ~ q from the trained",
            "    flow, score log q(x) and the action A(x); KL = F_c^q + lnZ_c.",
            "    >>> This IS the REVERSE-KL (energy-based) training objective:",
            "        reverse-KL loss = F_c^q = KL(q‖p) - lnZ_c, so minimising",
            "        the loss minimises KL(q‖p).  No data needed.",
            "",
            "  KL(p‖q) = E_p[log p - log q] = CE - H(p_HS)       (mass-covering)",
            "    CE = -E_p[log q].  Obtained from HS DATA samples x ~ p_HS",
            "    (the hs_L*.pt files): draw x ~ p_HS, score log q(x); subtract",
            "    the MC entropy H(p_HS) = E_p[A] + lnZ_c.",
            "    >>> This IS the FORWARD-KL / MLE (data-driven) objective:",
            "        MLE loss = CE = H(p_HS) + KL(p‖q), so minimising the loss",
            "        minimises KL(p‖q).  Needs samples from p (cannot be done",
            "        reliably by importance-reweighting flow samples).",
            "```",
            "Each training mode minimises only ONE direction; this diagnostic",
            "measures BOTH for every run regardless of how it was trained — so",
            "the *off-objective* KL is the honest cross-check.",
            "",
            "`KL(q‖p)` small but `KL(p‖q)` large ⇒ mode-dropping (flow ignores",
            "modes of `p`). Both small ⇒ genuinely good fit. For data-driven runs",
            "this is the only way to see the *model* `H(q)`: the training-time",
            "`ENTROPY` logs the data-side cross-entropy `-E_data[log q]`, which can",
            "dip below `H(p_HS)` purely from training-batch overfitting.",
            "",
        ]
        diag_section += _fmt_table(diag_rows, diag_aligns)
        diag_section += [
            "",
            "Standardized data-driven runs (flow trained on `u = x/σ`) are converted",
            "back to physical scale via `log q_X(x) = log q_U(x/σ) - N·logσ`; `σ` is",
            "read from each run's `flow_input_sigma.json`.",
        ]
    else:
        diag_section = [
            "",
            "## Flow diagnostic",
            "",
            "_No `flow_diagnostic.json` files found. Run "
            "`analyzers/flow_sample_diagnostic.py <folder> [<folder> ...]` "
            "(or `sbatch shell/flow_diagnostic.sh`) to populate them._",
        ]

    # --- Summary table: superset of the two tables above, in one place ---
    # F/E/S + both KL directions, for theory, the two datasets, and the best
    # run of each mode.  Font marks the SOURCE of each number:
    #   **bold**  = exact theory
    #   *italic*  = training-measured (run's HDF5 records)
    #   plain     = sample-measured (dataset average, or post-hoc flow
    #               diagnostic that draws x ~ q)
    def fi(v):
        return f"*{f6(v)}*" if isinstance(v, (int, float, np.floating)) else "N/A"

    hs_diag = None
    for _d in experiment_data.values():
        _dg = _d.get("diag")
        if _dg and isinstance(_dg.get("Hp_mc"), (int, float, np.floating)) \
                and isinstance(_dg.get("lnZ_c"), (int, float, np.floating)):
            hs_diag = _dg
            break

    sum_rows = [["Source", "Picture", "F (-lnZ)", "E", "S",
                 "KL(q‖p)", "KL(p‖q)"]]
    sum_aligns = ["left"] + ["center"] * 6

    # discrete picture: theory + MCMC dataset (grouped)
    sum_rows.append(["**Exact (theory)**", "discrete",
                     fb(th["F_d"]), fb(th["E_d"]), fb(th["S_d"]), "—", "—"])
    sum_rows.append(["MCMC dataset (Wolff)", "discrete",
                     "N/A", f6(mcmc_E_d), "N/A", "—", "—"])
    # continuous picture: theory + HS dataset + best run of each mode
    sum_rows.append(["**Exact (theory)**", "continuous",
                     fb(th["F_c"]), fb(th["E_c"]), fb(th["S_c"]),
                     "**0**", "**0**"])
    if hs_diag is not None:
        sum_rows.append(["HS dataset (x ~ p_HS)", "continuous", "N/A",
                         f6(hs_diag["Hp_mc"] - hs_diag["lnZ_c"]),
                         f6(hs_diag["Hp_mc"]), "—", "—"])
    else:
        sum_rows.append(["HS dataset (x ~ p_HS)", "continuous",
                         "N/A", "N/A", "N/A", "—", "—"])

    def _summary_method_rows(method):
        """Two rows per flow: training-measured (italic) + post-hoc diagnostic.

        Each KL direction appears once. The *training* row carries the
        ON-objective KL — the one that mode minimises, recovered from the loss:
          reverse-KL : KL(q||p) = loss + lnZ_c   (= min_loss - F_c)
          forward-KL : KL(p||q) = loss - H(p_HS)  (= min_loss - S_c)
        The *diagnostic* row carries the OFF-objective KL, which training
        cannot see and only x ~ q / x ~ p sampling provides.
        """
        d = experiment_data.get(method, {})
        ml = d.get("min_loss")
        ml_num = isinstance(ml, (int, float, np.floating))
        out = []
        if _classify(method) == "F_c":     # reverse-KL: F/E/S logged
            kl_qp_tr = (ml - th["F_c"]) if ml_num else None     # loss + lnZ_c
            out.append([f"*{method} — training*", "continuous",
                        fi(ml), fi(d.get("energy")), fi(d.get("entropy")),
                        fi(kl_qp_tr), "N/A"])
        else:                              # forward-KL: S is pure MLE (ENTROPY)
            # Use Best-200 (200-epoch rolling minimum-mean of ENTROPY)
            # as the reported S. Smoothed metric avoids the artifactual
            # negative KL(p‖q) from single-epoch minima dipping below
            # H(p_HS) due to batch noise.
            b200 = d.get("best_200_entropy")
            ent = d.get("entropy")
            b200_num = isinstance(b200, (int, float, np.floating))
            ent_num = isinstance(ent, (int, float, np.floating))
            fair_s = b200 if b200_num else (ent if ent_num else ml)
            fair_s_num = b200_num or ent_num or ml_num
            kl_pq_tr = (fair_s - th["S_c"]) if fair_s_num else None
            out.append([f"*{method} — training*", "continuous",
                        "N/A", "N/A", fi(fair_s), "N/A", fi(kl_pq_tr)])
        dg = d.get("diag")
        if dg and all(isinstance(dg.get(k), (int, float, np.floating))
                      for k in ("Fcq", "EA", "Hq")):
            ep = dg.get("epoch", "?")
            # diagnostic row carries only the OFF-objective KL
            if _classify(method) == "F_c":            # reverse-KL -> KL(p||q)
                kl_qp_d, kl_pq_d = "N/A", f6(dg.get("KL_pq"))
            else:                                     # forward-KL -> KL(q||p)
                kl_qp_d, kl_pq_d = f6(dg.get("KL_qp")), "N/A"
            out.append([f"{method} — diagnostic (epoch {ep})", "continuous",
                        f6(dg["Fcq"]), f6(dg["EA"]), f6(dg["Hq"]),
                        kl_qp_d, kl_pq_d])
        return out

    # Include ALL trained variants (not just best-of-mode) so nothing gets
    # hidden. Group by mode, sort by min_loss (ascending). The former
    # "best-of-mode" filter obscured every HCG/VP variant when a baseline
    # happened to have a marginally lower L (which is unfair for VP runs
    # whose L is inflated by the penalty term).
    # Sort forward-KL runs by ENTROPY (pure MLE, penalty-free) so VP variants
    # rank fairly. Fall back to min_loss when ENTROPY unavailable.
    def _fair_key(d):
        b200 = d.get("best_200_entropy")
        if isinstance(b200, (int, float, np.floating)):
            return b200
        ent = d.get("entropy")
        if isinstance(ent, (int, float, np.floating)):
            return ent
        return d.get("min_loss", float("inf"))
    rev_sorted = sorted(rev.items(), key=lambda kv: _fair_key(kv[1]))
    fwd_sorted = sorted(fwd.items(), key=lambda kv: _fair_key(kv[1]))
    for method, _ in rev_sorted:
        sum_rows += _summary_method_rows(method)
    for method, _ in fwd_sorted:
        sum_rows += _summary_method_rows(method)

    summary_section = [
        "",
        "## Summary — everything in one table",
        "",
        "Superset of the two tables above: free energy / energy / entropy **and**",
        "both KL directions, for exact theory, the two **datasets**, and the best",
        "trained flow of each mode. Discrete rows are grouped first.",
        "",
        "Font marks where each number comes from (Markdown has no portable text",
        "colour, so font carries the distinction):",
        "",
        "- **bold** — exact theory (Onsager / `exactz.md`).",
        "- *italic* — training-measured, read from the run's HDF5 records. A",
        "  reverse-KL run logs `F/E/S` of the flow; a forward-KL run logs only",
        "  `S` (the MLE loss `-E_data[log q]`) — its `F/E` are N/A.",
        "- plain — sample-measured: a dataset sample-average, or the post-hoc",
        "  flow diagnostic that draws `x ~ q` (the only way to get a forward-KL",
        "  run's model-side `F/E`).",
        "",
    ]
    summary_section += _fmt_table(sum_rows, sum_aligns)
    summary_section += [
        "",
        "Notes:",
        "- Each flow gets **two rows** — *training* and *diagnostic* — the same run",
        "  as the optimiser logged it vs. as a fresh `x ~ q` sample measures it. For",
        "  a converged reverse-KL run the two should agree.",
        "- **Datasets**: `E` is a plain sample average; `F = -lnZ` cannot be",
        "  estimated from samples (needs the partition function) → N/A. HS",
        "  `S_c = E_p[A] + lnZ_c` is an MC entropy estimate (uses exact `lnZ_c`);",
        "  MCMC gives only `E_d`.",
        "- `KL(q‖p)` / `KL(p‖q)`: each direction appears once per flow. The",
        "  *training* row carries the **on-objective** KL — the one that mode",
        "  minimises, recovered from the loss (reverse-KL `KL(q‖p)=loss+lnZ_c`;",
        "  forward-KL `KL(p‖q)=loss-H(p_HS)`). The *diagnostic* row carries the",
        "  **off-objective** KL, which training cannot see. `—` = not applicable",
        "  (theory-discrete / dataset rows); `0` for continuous theory.",
        "- A **negative** training-row `KL(p‖q)` means the MLE loss dipped below",
        "  the entropy floor `H(p_HS)` — training-set overfitting (seen at L=8/16).",
        "- The per-run breakdown for *all* methods stays in the flow-diagnostic",
        "  table above; this summary keeps only the best of each mode.",
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines + table_lines + legend
                          + diag_section + summary_section) + "\n")

    print(f"Report written: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate minimum loss report for a specific temperature.")
    parser.add_argument("-t", "--temp", type=float, required=True, help="Temperature T (e.g. 2.269)")
    parser.add_argument("-L", "--lattice", type=int, default=32, help="Lattice size L (e.g. 8, 16, 32)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="skip methods whose name contains any of these substrings")

    args = parser.parse_args()

    generate_report_for_T(args.temp, args.lattice, exclude=args.exclude)