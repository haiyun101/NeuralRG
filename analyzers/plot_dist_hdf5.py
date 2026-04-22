import os
import glob
import re
import math
import argparse
import h5py
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # Required for cluster environments to prevent GUI errors
import matplotlib.pyplot as plt

# Set this to True if you want to re-generate and overwrite existing plots
force_overwrite = False

# --- Physical Constants for 2D Ising Model ---
T_C = 2.0 / math.log(1.0 + math.sqrt(2.0))  # Critical Temperature ~ 2.269185
T_MFC = 4.0

# --- Function to calculate G(r) using numpy ---
def calculate_correlation_numpy(samples):
    L = samples.shape[-1]
    max_r = L // 2
    G_r = np.zeros(max_r)
    
    for r in range(max_r):
        shift_x = np.roll(samples, shift=-r, axis=-1)
        shift_y = np.roll(samples, shift=-r, axis=-2)
        G_r[r] = (np.mean(samples * shift_x) + np.mean(samples * shift_y)) / 2.0
    
    return G_r

def get_spontaneous_magnetization(T):
    """Calculates Onsager's exact spontaneous magnetization for 2D Ising."""
    if T >= T_C:
        return 0.0
    else:
        return (1.0 - math.sinh(2.0 / T)**-4)**0.125

def get_correlation_length(T):
    """Calculates exact correlation length xi for 2D Ising at T > Tc."""
    if T <= T_C:
        return float('inf') 
    K = 1.0 / T
    xi_inv = math.log(1.0 / math.tanh(K)) - 2.0 * K
    return 1.0 / xi_inv

def guess_temperature_from_string(folder_name):
    """Robustly extracts target temperature from folder names."""
    cleaned = re.sub(r'\d+Ising', '', folder_name)
    matches = re.findall(r'\d+\.\d+|\d+', cleaned)
    if matches:
        return float(matches[-1]) 
    return T_C 

def get_theory_min(target_T, L_target, filepath="../etc/exactz.md"):
    """
    1. 定位 L=target_L 的 Section
    2. 匹配温度 (保留两位有效数字进行匹配)
    """
    if not os.path.exists(filepath):
        # 尝试备用路径，防止运行路径不一致
        filepath = os.path.join(os.path.dirname(__file__), "../etc/exactz.md")
        if not os.path.exists(filepath): return None

    target_T_str = f"{float(target_T):.1f}" # 根据截图，T 是一列如 2.3 的格式
    section_found = False
    
    with open(filepath, 'r') as f:
        for line in f:
            # 只有匹配到正确的格点数标题才开始搜索
            if line.startswith("###") and f"n={L_target}" in line:
                section_found = True
                continue
            
            # 如果进入了下一个标题，停止搜索
            if section_found and line.startswith("###") and f"n={L_target}" not in line:
                break
            
            if section_found and line.startswith("|"):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    try:
                        # 尝试精确匹配字符串或数值
                        t_val = float(parts[1])
                        if abs(t_val - float(target_T)) < 1e-3:
                            lnZ = float(parts[2])
                            fix = float(parts[3])
                            return -(lnZ + fix)
                    except ValueError:
                        continue
    return None

# --- Function to generate Interactive HTML Viewer ---
def create_html_viewer(target_path, image_filenames, default_fps):
    # Removed the redundant <h2> title here, relying on matplotlib's suptitle
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>System Evolution Viewer</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-width: 1000px; width: 100%; text-align: center; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }}
        .controls {{ margin-top: 20px; display: flex; flex-direction: column; gap: 15px; width: 100%; align-items: center; }}
        .control-row {{ display: flex; align-items: center; justify-content: center; gap: 15px; width: 80%; }}
        input[type="range"] {{ flex-grow: 1; cursor: pointer; }}
        button {{ padding: 8px 16px; font-size: 16px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 4px; }}
        button:hover {{ background-color: #0056b3; }}
        .label {{ font-weight: bold; min-width: 100px; text-align: left; }}
    </style>
</head>
<body>
    <div class="container">
        <img id="display" src="{image_filenames[0]}" alt="Evolution Frame">
        
        <div class="controls">
            <div class="control-row">
                <button id="playBtn" onclick="togglePlay()">Pause</button>
                <input type="range" id="progressSlider" min="0" max="{len(image_filenames)-1}" value="0" oninput="seekFrame(this.value)">
                <span class="label" id="frameLabel">Frame: 0</span>
            </div>
            <div class="control-row">
                <span class="label">Speed:</span>
                <input type="range" id="speedSlider" min="1" max="60" value="{default_fps}" oninput="changeSpeed(this.value)">
                <span class="label" id="speedLabel">{default_fps} FPS</span>
            </div>
        </div>
    </div>

    <script>
        const frames = {json.dumps(image_filenames)};
        let currentIndex = 0;
        let isPlaying = true;
        let currentFps = {default_fps};
        let playTimer = null;

        const imgElement = document.getElementById('display');
        const progressSlider = document.getElementById('progressSlider');
        const frameLabel = document.getElementById('frameLabel');
        const playBtn = document.getElementById('playBtn');
        const speedSlider = document.getElementById('speedSlider');
        const speedLabel = document.getElementById('speedLabel');

        function updateDisplay(index) {{
            currentIndex = parseInt(index);
            imgElement.src = frames[currentIndex];
            progressSlider.value = currentIndex;
            frameLabel.innerText = 'Frame: ' + currentIndex;
        }}

        function showNextFrame() {{
            updateDisplay((currentIndex + 1) % frames.length);
        }}

        function seekFrame(index) {{
            if (isPlaying) togglePlay(); 
            updateDisplay(index);
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            playBtn.innerText = isPlaying ? "Pause" : "Play";
            isPlaying ? startLoop() : stopLoop();
        }}

        function changeSpeed(newFps) {{
            currentFps = parseInt(newFps);
            speedLabel.innerText = currentFps + ' FPS';
            if (isPlaying) {{ stopLoop(); startLoop(); }}
        }}

        function startLoop() {{
            if (playTimer !== null) clearInterval(playTimer);
            playTimer = setInterval(showNextFrame, 1000 / currentFps);
        }}

        function stopLoop() {{
            if (playTimer !== null) clearInterval(playTimer);
            playTimer = null;
        }}

        startLoop();
    </script>
</body>
</html>
"""
    html_path = os.path.join(target_path, 'Direct_viewer.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    return html_path

def main():
    parser = argparse.ArgumentParser(description="Generate PNGs, GIF, and HTML with Fixed Axes and 16 Samples.")
    parser.add_argument('folder', type=str, help="Subfolder name, e.g., T_2.6")
    parser.add_argument('-base', type=str, default='', help="Base path, default is ./opt")
    parser.add_argument('-fps', type=int, default=5, help="Frames per second for animation")
    parser.add_argument('-T', '--temperature', type=float, default=None, help="Force specific temperature for theory lines")
    args = parser.parse_args()

    target_path = os.path.join(args.base, args.folder)
    
    T_target = args.temperature if args.temperature is not None else guess_temperature_from_string(args.folder)
    m0_theory = get_spontaneous_magnetization(T_target)
    
    print(f"[*] Physical Parameters Loaded: T = {T_target:.4f} (Tc = {T_C:.4f}) | Theoretical m_0 = {m0_theory:.4f}")

    search_pattern = os.path.join(target_path, '**/*HMCresult*.hdf5')
    files = glob.glob(search_pattern, recursive=True)
    
    def extract_epoch(filename):
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    files = sorted(files, key=extract_epoch)

    if not files:
        print(f"[!] No *HMCresult*.hdf5 files found in {target_path}.")
        return

    print(f"[*] Found {len(files)} sampling record files. Generating plots...")

    generated_pngs = []

        # --- 在 for 循环开始前，找到最新的 Record 文件 ---

    # 这里的 target_path 是你传入的数据文件夹路径
    all_records = sorted( glob.glob(os.path.join(target_path, "records", "*Record_epoch*.hdf5")),
                          key=lambda x: int(x.split('epoch')[-1].split('.')[0])
                        )
    
    full_loss, full_xacc, full_zacc = None, None, None
    if all_records:
        latest_record = all_records[-1] # 使用最后一个文件，因为它存了最全的历史
        with h5py.File(latest_record, 'r') as rf:
            full_loss = np.array(rf["LOSS"]).flatten()
        # --- 新增：提取能量和熵 ---
            if "ENERGY" in rf:
                full_energy = np.array(rf["ENERGY"]).flatten()
            if "ENTROPY" in rf:
                full_entropy = np.array(rf["ENTROPY"]).flatten()

            full_xacc = np.array(rf["XACC"]).flatten()
            full_zacc = np.array(rf["ZACC"]).flatten()
    
    #     # --- 1. 在循环之前，确定全局 X 轴范围 ---
    # # 假设 files 是你 glob 出来的 HMCresult 文件列表
    # max_epoch = max([int(re.findall(r'\d+', os.path.basename(f))[-1]) for f in files])
    # print(max_epoch)
    # 直接用数据的长度作为 X 轴的最大值
    max_data_points = len(full_loss) 

    print(f"[*] Data length detected: {max_data_points} points.")
    max_epoch = max_data_points + 9
            
    for target_file in tqdm(files, desc="Plotting Epochs"):
        epoch_num = extract_epoch(target_file)
        save_name = f'DistGr_epoch_{epoch_num:05d}.png'
        save_path = os.path.join(target_path, save_name)

        if not os.path.exists(save_path) or force_overwrite:
            try:
                with h5py.File(target_file, 'r') as f:
                    x_raw = np.array(f['X'])
                    batch_size = x_raw.shape[0]
                    
                    x_flat = x_raw.reshape(batch_size, -1)
                    N_total = x_flat.shape[1]
                    L = int(np.sqrt(N_total))
                    
                    s_flat = np.sign(x_flat)
                    m_values = np.mean(s_flat, axis=1)
                    s_2d = s_flat.reshape(batch_size, L, L)
                    
                    gr_values = calculate_correlation_numpy(s_2d)
                    r_axis = np.arange(len(gr_values))

            except Exception as e:
                print(f"\n[!] Error processing file {target_file}: {e}")
                continue

            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
            plt.subplots_adjust(hspace=0.3)
            
            # --- Add Main Title with Folder Name ---
            fig.suptitle(f"System Evolution | Folder: {args.folder}", fontsize=20, fontweight='bold')
            
            # === Plot 1: x distribution ===
            ax1.hist(x_flat.flatten(), bins=100, color='royalblue', alpha=0.7, density=True, label='Empirical Data')
            
            # Theory: Mean-Field Approximation for continuous field x
            K_eff = T_MFC / T_target
            x_range = np.linspace(-6, 6, 500)
            px_theory = np.exp(-x_range**2 / (2 * K_eff)) * np.cosh(x_range)
            px_theory /= np.trapz(px_theory, x_range)
            ax1.plot(x_range, px_theory, 'k--', linewidth=2, alpha=0.8, label='Mean-Field Theory')
            
            ax1.set_title(f"Continuous Field $x$\n(Epoch: {epoch_num})")
            ax1.set_xlabel("$x$")
            ax1.set_ylabel("Density")
            ax1.grid(True, alpha=0.3)
            # # Fix Axes and Legend
            # ax1.set_xlim(-6.5, 6.5)
            # ax1.set_ylim(0, 0.6)
            ax1.legend(loc='upper right')

            # === Plot 2: 16 Configuration Samples ===
            n_samples = min(16, batch_size)
            grid_dim = int(np.ceil(np.sqrt(n_samples))) # Typically 4x4
            pad = 1 # 1 pixel padding between samples
            comp_size = grid_dim * L + (grid_dim - 1) * pad
            
            # Initialize with 0. In 'binary' cmap (vmin=-1, vmax=1), 0 is perfect gray padding
            composite = np.zeros((comp_size, comp_size)) 
            
            for idx in range(n_samples):
                r = idx // grid_dim
                c = idx % grid_dim
                composite[r*(L+pad):r*(L+pad)+L, c*(L+pad):c*(L+pad)+L] = s_2d[idx]

            ax2.imshow(composite, cmap='binary', vmin=-1, vmax=1)
            ax2.set_title(f"Configuration Samples ({n_samples})\n(Epoch: {epoch_num})")
            ax2.axis('off') # Turn off axes completely for clean images

            # === Plot 3: m distribution ===
            ax3.hist(m_values, bins=50, range=(-1.1, 1.1), color='forestgreen', alpha=0.7, density=True, label='Empirical Data')
            
            # Theory: Landau Phase Transition Approximation for m
            N_landau = 10.0 
            b_landau = 0.5  
            a_landau = T_target - T_C
            m_range = np.linspace(-1.5, 1.5, 500)
            pm_theory = np.exp(-N_landau * (a_landau * m_range**2 + b_landau * m_range**4))
            pm_theory /= np.trapz(pm_theory, m_range)
            ax3.plot(m_range, pm_theory, 'k--', linewidth=2, alpha=0.8, label='Landau Theory')

            # Exact Onsager Magnetization Lines
            if T_target < T_C:
                ax3.axvline(x=m0_theory, color='r', linestyle=':', linewidth=2, alpha=0.8, label=f'Onsager $m_0$ (+{m0_theory:.3f})')
                ax3.axvline(x=-m0_theory, color='r', linestyle=':', linewidth=2, alpha=0.8, label=f'Onsager $m_0$ (-{m0_theory:.3f})')
            else:
                ax3.axvline(x=0.0, color='r', linestyle=':', linewidth=2, alpha=0.8, label='Paramagnetic Center ($m=0$)')
            
            ax3.set_title(f"Order Parameter $m$ ($T={T_target:.3f}$)\n(Epoch: {epoch_num})")
            ax3.set_xlabel("Magnetization $m$")
            ax3.set_ylabel("Density")
            ax3.grid(True, alpha=0.3)
            # # Fix Axes and Legend
            # ax3.set_xlim(-1.2, 1.2)
            # ax3.set_ylim(0, 9.0) # Set high enough to accommodate delta-like peaks
            ax3.legend(loc='upper right')
            
            # === Plot 4: G(r) Log-Log Plot ===
            valid_r = r_axis[1:]
            valid_gr = gr_values[1:]
            pos_mask = valid_gr > 0
            
            if np.any(pos_mask):
                ax4.loglog(valid_r[pos_mask], valid_gr[pos_mask], 'o-', color='purple', markersize=5, label='$G(r)$ Data')
                
                first_r = valid_r[pos_mask][0]
                first_gr = valid_gr[pos_mask][0]

                c_ref_crit = first_gr * (first_r ** 0.25) 
                y_ref_crit = c_ref_crit * (valid_r ** -0.25)
                ax4.loglog(valid_r, y_ref_crit, 'k:', alpha=0.4, label='Critical Decay $r^{-0.25}$')
                
                if T_target > T_C:
                    xi = get_correlation_length(T_target)
                    c_ref_exp = first_gr * np.sqrt(first_r) * np.exp(first_r / xi)
                    y_ref_exp = c_ref_exp * np.exp(-valid_r / xi) / np.sqrt(valid_r)
                    ax4.loglog(valid_r, y_ref_exp, 'r--', linewidth=2, alpha=0.8, label=f'Exp Decay Theory ($\\xi={xi:.2f}$)')

            if T_target < T_C:
                ax4.axhline(y=m0_theory**2, color='r', linestyle='--', alpha=0.8, label=f'LRO Plateau ($m_0^2={m0_theory**2:.4f}$)')

            ax4.set_title(f"Two-Point Correlation $G(r)$\n(Epoch: {epoch_num})")
            ax4.set_xlabel("Distance $r$ (log scale)")
            ax4.set_ylabel("$G(r)$ (log scale)")
            ax4.grid(True, which="both", ls="--", alpha=0.3) 
            # ax4.relim()        # Recalculate limits based on current data
            # ax4.autoscale_view() # Apply the limits
            # # Fix Axes and Legend
            # ax4.set_xlim(0.9, (L // 2) + 1)
            # ax4.set_ylim(1e-1, 1.0)
            ax4.legend(loc='lower left', fontsize=9)
            

            
            # # --- 2. 在循环内部 (for target_file in tqdm(files)) ---
            # epoch_num = int(re.findall(r'\d+', os.path.basename(target_file))[0])
            current_idx = epoch_num // 10  # 这里的 10 应与你的 savePeriod 一致
            
            if full_loss is not None:
                # === Plot 5: Loss Plot ===
                ax5.clear()
                display_loss = full_loss[:current_idx + 1]
                
                if len(display_loss) > 0:
                    # X 轴刻度：每一个点对应一个 savePeriod (10)
                    x_loss = np.arange(len(display_loss)) * 10
                    # ax5.plot(x_loss, display_loss, color='orange', label='Loss')

                    # 1. 画总 Loss (红线)
                    ax5.plot(x_loss, display_loss, color='tab:red', label='Total Loss (F)', linewidth=2, zorder=4)
                    
                    # 2. 画 Energy (橙线)
                    if full_energy is not None:
                        display_energy = full_energy[:current_idx + 1]
                        ax5.plot(x_loss, display_energy, color='tab:orange', label='Energy', alpha=0.8, zorder=3)
                    
                    # 3. 画 -Entropy (蓝线)
                    if full_entropy is not None:
                        display_entropy = full_entropy[:current_idx + 1]
                        ax5.plot(x_loss, -display_entropy, color='tab:blue', label='-Entropy', alpha=0.8, zorder=2)
                    
        # 注意：从文件名读取的 L 传递给 get_theory_min
                    t_min = get_theory_min(T_target, L_target=32) 
                    if t_min is not None:
                        ax5.axhline(y=t_min, color='black', linestyle='--', linewidth=1.5, label=f'Exact Min ({t_min:.2f})', zorder=5)

                    # --- 关键修改：添加图例 ---
                    ax5.legend(loc='upper right', fontsize='x-small', framealpha=0.5)                     
                    # 核心修改：锁定全局范围
                    ax5.set_xlim(0, max_epoch) 
                    
                    # 保护性 Log Scale：只有存在正值时才开启
                    if np.any(display_loss > 0):
                        ax5.set_yscale('log')
                        
                    ax5.set_title(f"Loss History (Epoch {epoch_num})")
                    ax5.set_xlabel("Epochs")
                    ax5.grid(True, which='both', alpha=0.3)
            
                # === Plot 6: Acceptance Ratio ===
                ax6.clear()
                display_x = full_xacc[:current_idx + 1]
                display_z = full_zacc[:current_idx + 1]
                
                if len(display_x) > 0:
                    x_acc_axis = np.arange(len(display_x)) * 10
                    ax6.plot(x_acc_axis, display_x, 'o-', label='Phys Acc', markersize=3)
                    ax6.plot(x_acc_axis, display_z, 's-', label='Latent Acc', markersize=3)
                    
                    # 核心修改：锁定全局范围
                    ax6.set_xlim(0, max_epoch)
                    ax6.set_ylim(-0.05, 1.05)
                    
                    ax6.set_title("Acceptance Ratio")
                    ax6.set_xlabel("Epochs")
                    ax6.legend(loc='lower right', fontsize=8)
                    ax6.grid(True, alpha=0.2)

            # Adjust layout to make room for the large suptitle
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path, dpi=120)
            plt.close()

        generated_pngs.append(save_path)

    if not generated_pngs:
        return

    # --- Step 2: Generate GIF ---
    print(f"\n[*] Step 2: Generating GIF...")
    gif_path = os.path.join(target_path, 'Direct_evolution.gif')
    try:
        frames = [Image.open(img) for img in generated_pngs]
        duration_ms = int(1000 / args.fps)
        frames[0].save(
            gif_path, format='GIF', append_images=frames[1:],
            save_all=True, duration=duration_ms, loop=0 
        )
    except Exception as e:
        print(f"[!] Failed to generate GIF: {e}")

    # --- Step 3: Generate Interactive HTML Viewer ---
    print("[*] Step 3: Generating Interactive HTML viewer...")
    basenames = [os.path.basename(p) for p in generated_pngs]
    create_html_viewer(target_path, basenames, args.fps)
    print("\n[+] ALL DONE! Interactive viewer updated.")

if __name__ == '__main__':
    main()