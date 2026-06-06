# import matplotlib
# matplotlib.use('Agg') # Must be called before importing pyplot
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os
# import re

# def calc_correlation_dataset(samples):
#     """
#     Calculate the spatial spin correlation function G(r) averaged over 
#     the entire dataset (all samples) and both spatial directions (x and y).
#     """
#     # samples shape is assumed to be (N, L, L)
#     L = samples.shape[-1]
#     max_r = L // 2
#     G_r = np.zeros(max_r)
    
#     for r in range(max_r):
#         # Shift along X (axis=2) and Y (axis=1)
#         shift_x = np.roll(samples, shift=-r, axis=2)
#         shift_y = np.roll(samples, shift=-r, axis=1)
        
#         # Mean over all samples, and all spatial positions
#         corr_x = np.mean(samples * shift_x)
#         corr_y = np.mean(samples * shift_y)
        
#         G_r[r] = (corr_x + corr_y) / 2.0
        
#     return G_r

# def analyze_and_plot(data_path):
#     print(f"Loading data from {data_path}...")
#     # Load the PyTorch tensor and convert to numpy
#     # Expected shape from generate_mcmc_data.py is (N, 1, L, L)
#     data = torch.load(data_path)
#     samples = data.squeeze().numpy()  # Shape becomes (N, L, L)
    
#     N, L, _ = samples.shape
#     print(f"Successfully loaded {N} samples of size {L}x{L}.")

#     # Extract T and L from filename if possible for the title
#     basename = os.path.basename(data_path)
#     t_match = re.search(r'_T([0-9.]+)', basename)
#     l_match = re.search(r'_L(\d+)', basename)
#     T_str = t_match.group(1) if t_match else "Unknown"
#     L_str = l_match.group(1) if l_match else str(L)

#     print("Calculating Magnetization...")
#     # Average over spatial dimensions (L, L) to get m for each sample
#     m_array = np.mean(samples, axis=(1, 2))
    
#     print("Calculating G(r)...")
#     g_r = calc_correlation_dataset(samples)

#     print("Generating plots...")
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5))
#     fig.suptitle(f'MCMC Data Analysis | L = {L_str} | T = {T_str} | N = {N}', 
#                  fontsize=15, fontweight='bold')
    
#     # ---------------------------------------------------------
#     # Subplot 1: 16 Independent Lattice Samples
#     # ---------------------------------------------------------
#     # Pick the first 16 samples (or fewer if N < 16)
#     num_imgs = min(16, N)
#     img_list = samples[:num_imgs]
    
#     # Construct a 4x4 grid (if we have at least 16)
#     if num_imgs == 16:
#         rows = [np.hstack(img_list[j*4:(j+1)*4]) for j in range(4)]
#         img_grid = np.vstack(rows)
#         axes[0].imshow(img_grid, cmap='Greys', vmin=-1, vmax=1, interpolation='nearest')
#     else:
#         # Fallback if there are very few samples
#         axes[0].imshow(samples[0], cmap='Greys', vmin=-1, vmax=1, interpolation='nearest')
        
#     axes[0].axis('off')
#     axes[0].set_title('16 Lattice Samples')
    
#     # ---------------------------------------------------------
#     # Subplot 2: Magnetization Distribution p(m)
#     # ---------------------------------------------------------
#     axes[1].hist(m_array, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7)
#     axes[1].set_title('Magnetization Distribution $p(m)$')
#     axes[1].set_xlabel('Magnetization $m$')
#     axes[1].set_ylabel('Density')
#     axes[1].set_xlim(-1.1, 1.1)
#     axes[1].grid(True, alpha=0.3)
    
#     # ---------------------------------------------------------
#     # Subplot 3: Spin Correlation Function G(r)
#     # ---------------------------------------------------------
#     r_vals = np.arange(L // 2)
#     axes[2].plot(r_vals, g_r, marker='o', linestyle='-', color='tab:red', markersize=6)
#     axes[2].set_title('Spin Correlation Function $G(r)$')
#     axes[2].set_xlabel('Distance $r$')
#     axes[2].set_ylabel(r'$\langle s_i s_{i+r} \rangle$')
#     axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
#     axes[2].grid(True, alpha=0.3)
    
#     # Finalize and save
#     plt.tight_layout(rect=[0, 0, 1, 0.90]) 
    
#     # Save the figure in the same directory as the data or in analyzers
#     save_filename = f"Data_Analysis_L{L_str}_T{T_str}.png"
#     plt.savefig(save_filename, dpi=200)
#     print(f"Plot successfully saved to {save_filename}")
#     plt.show()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Analyze and plot generated MCMC dataset")
#     parser.add_argument("data_path", type=str, help="Path to the .pt MCMC data file (e.g., ./data/mcmc_data/mcmc_wolff_L32_T2.2692_N50000.pt)")
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.data_path):
#         print(f"Error: File '{args.data_path}' not found.")
#     else:
#         analyze_and_plot(args.data_path)

# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg') # Keeps it safe for cluster environments
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import argparse
# import os

# def analyze_and_animate(data_path):
#     print(f"Loading data from {data_path}...")
#     data = torch.load(data_path, weights_only=True)
#     samples = data.squeeze().numpy() 
    
#     if samples.ndim == 2:
#         samples = samples[np.newaxis, ...]
        
#     N, L, _ = samples.shape
#     N = 400
#     mid = L // 2

#     # 1. Pre-calculate regional magnetization
#     m_tl = np.mean(samples[:, :mid, :mid], axis=(1, 2))
#     m_tr = np.mean(samples[:, :mid, mid:], axis=(1, 2))
#     m_bl = np.mean(samples[:, mid:, :mid], axis=(1, 2))
#     m_br = np.mean(samples[:, mid:, mid:], axis=(1, 2))
    
#     m_data_list = [m_tl, m_tr, m_bl, m_br]
#     titles = ['Top-Left Quadrant', 'Top-Right Quadrant', 
#               'Bottom-Left Quadrant', 'Bottom-Right Quadrant']
#     colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

#     # 2. Setup Figure
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#     axes = axes.flatten()

#     bins = np.linspace(-1.1, 1.1, 51)
    
#     bars_list = []
#     for i, ax in enumerate(axes):
#         ax.set_title(titles[i], fontweight='bold')
#         ax.set_xlim(-1.1, 1.1)
#         ax.set_ylim(0, 2) 
#         ax.set_xlabel('Magnetization $m$')
#         ax.set_ylabel('Density $p(m)$')
#         ax.grid(True, alpha=0.3)
        
#         counts, edges = np.histogram([0], bins=bins, density=True)
#         bars = ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge', 
#                       color=colors[i], edgecolor='black', alpha=0.7)
#         bars_list.append(bars)

#     def update(frame_idx):
#         for i, ax in enumerate(axes):
#             current_history = m_data_list[i][:frame_idx]
#             counts, _ = np.histogram(current_history, bins=bins, density=True)
            
#             for bar, count in zip(bars_list[i], counts):
#                 bar.set_height(count)
                
#             max_count = np.max(counts) if len(counts) > 0 else 0
#             if max_count > ax.get_ylim()[1]:
#                 ax.set_ylim(0, max_count * 1.15)

#         fig.suptitle(f"Magnetization Distribution | Samples: {frame_idx} / {N}", 
#                      fontsize=14, fontweight='bold')
        
#         return [bar for sublist in bars_list for bar in sublist]

#     # Subsample frames (e.g., ~150 frames) to keep HTML size manageable
#     frame_step = max(1, N // 150)
#     frames = list(range(10, N, frame_step))
#     if frames[-1] != N:
#         frames.append(N)

#     print("Building interactive animation... (this might take a minute)")
#     ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

#     # 3. Export to Interactive HTML
#     save_name = data_path.replace('.pt', '_interactive_histograms.html')
    
#     try:
#         # to_jshtml() creates an HTML string with a built-in player (Play, Pause, Slider)
#         html_content = ani.to_jshtml()
#         with open(save_name, "w") as f:
#             f.write(html_content)
#         print(f"Success! Download and open this file in your browser: {save_name}")
#     except Exception as e:
#         print(f"Error saving interactive HTML: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("data_path", type=str)
#     args = parser.parse_args()
    
#     if os.path.exists(args.data_path):
#         analyze_and_animate(args.data_path)
#     else:
#         print(f"Error: {args.data_path} not found.")

# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg') # Essential for headless cluster environments
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import argparse
# import os

# def analyze_and_animate_spins(data_path):
#     print(f"Loading data from {data_path}...")
#     data = torch.load(data_path, weights_only=True)
#     samples = data.squeeze().numpy() 
#     samples = samples[0:400]
    
#     if samples.ndim == 2:
#         samples = samples[np.newaxis, ...]
        
#     N, L, _ = samples.shape
#     n_spin = 50

#     # 1. Randomly select 20 unique grid points (x, y coordinates)
#     np.random.seed(42) # Optional: set seed for reproducibility across runs
#     flat_indices = np.random.choice(L * L, n_spin, replace=False)
#     xs, ys = np.unravel_index(flat_indices, (L, L))
    
#     # 2. Extract the time-series data for these 20 specific spins
#     # Shape of selected_spins: (N, 20)
#     selected_spins = samples[:, xs, ys]
    
#     # 3. Calculate the running time-average for each spin
#     # We use cumulative sum divided by the number of steps up to that point
#     steps = np.arange(1, N + 1).reshape(-1, 1) # Shape: (N, 1)
#     running_avg = np.cumsum(selected_spins, axis=0) / steps # Shape: (N, 20)

#     # Create Labels for the X-axis in the format "(x, y)"
#     labels = [f"({x},{y})" for x, y in zip(xs, ys)]

#     # 4. Setup Figure and Bar Chart
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Initialize 20 bars at 0
#     bars = ax.bar(range(n_spin), np.zeros(n_spin), color='mediumpurple', edgecolor='black', alpha=0.8)
    
#     # Formatting
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_xlim(-0.5, 19.5)
#     ax.set_xticks(range(n_spin))
#     ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
#     ax.axhline(0, color='black', lw=1, ls='--')
#     ax.set_ylabel('Running Time-Average $\\langle m_i \\rangle_t$')
#     ax.set_xlabel('Selected Grid Coordinates $(x, y)$')
#     ax.grid(axis='y', linestyle='--', alpha=0.5)

#     # 5. Animation Update Function
#     def update(frame_idx):
#         # Grab the current running average for all 20 spins
#         current_vals = running_avg[frame_idx]
        
#         # Update the height of each bar
#         for bar, val in zip(bars, current_vals):
#             bar.set_height(val)
#             # Optional: dynamically color positive as red, negative as blue
#             bar.set_color('#ff6666' if val > 0 else '#66b3ff')
#             bar.set_edgecolor('black')
            
#         fig.suptitle(f"Spin Running Time-Average | Sample: {frame_idx + 1} / {N}", 
#                      fontsize=14, fontweight='bold')
#         return bars

#     # Subsample frames to keep the HTML file size lightweight (~150 frames total)
#     frame_step = max(1, N // 150)
#     frames = list(range(0, N, frame_step))
#     if frames[-1] != N - 1:
#         frames.append(N - 1) # Ensure the final frame is captured

#     print("Building interactive animation... (this might take a minute)")
#     ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)

#     # 6. Export to Interactive HTML
#     save_name = data_path.replace('.pt', '_spin_time_avg_interactive.html')
    
#     try:
#         html_content = ani.to_jshtml()
#         with open(save_name, "w") as f:
#             f.write(html_content)
#         print(f"Success! Download and open this file in your browser: {save_name}")
#     except Exception as e:
#         print(f"Error saving interactive HTML: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("data_path", type=str)
#     args = parser.parse_args()
    
#     if os.path.exists(args.data_path):
#         analyze_and_animate_spins(args.data_path)
#     else:
#         print(f"Error: {args.data_path} not found.")


# average latest 30 steps
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential for headless cluster environments
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os

def analyze_and_animate_spins(data_path, window_size=20):
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=True)
    samples = data.squeeze().numpy() 
    
    if samples.ndim == 2:
        samples = samples[np.newaxis, ...]


    samples = samples[0:1000]    
    N, L, _ = samples.shape
    
    if N < window_size:
        raise ValueError(f"Total samples ({N}) is less than window size ({window_size}).")

    # 1. Randomly select 20 unique grid points (x, y coordinates)
    np.random.seed(42) # Set seed for reproducibility
    flat_indices = np.random.choice(L * L, 20, replace=False)
    xs, ys = np.unravel_index(flat_indices, (L, L))
    
    # 2. Extract the time-series data for these 20 specific spins
    selected_spins = samples[:, xs, ys] # Shape: (N, 20)
    
    # 3. Calculate the Sliding Window Moving Average (Strictly full windows)
    print(f"Calculating sliding window average strictly for the last {window_size} steps...")
    cumsum = np.cumsum(selected_spins, axis=0)
    running_avg = np.zeros_like(selected_spins, dtype=float)
    
    # The first full window is at index (window_size - 1)
    running_avg[window_size - 1] = cumsum[window_size - 1] / window_size
    
    # For all subsequent frames, use the sliding window formula
    if N > window_size:
        running_avg[window_size:] = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    labels = [f"({x},{y})" for x, y in zip(xs, ys)]

    # 4. Setup Figure and Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(20), np.zeros(20), color='mediumpurple', edgecolor='black', alpha=0.8)
    
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(range(20))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel(f'Moving Average $\\langle m_i \\rangle_{{t}}$ (strict {window_size}-step window)')
    ax.set_xlabel('Selected Grid Coordinates $(x, y)$')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 5. Animation Update Function
    def update(frame_idx):
        current_vals = running_avg[frame_idx]
        
        for bar, val in zip(bars, current_vals):
            bar.set_height(val)
            # Dynamically color positive as red, negative as blue
            bar.set_color('#ff6666' if val > 0 else '#66b3ff')
            bar.set_edgecolor('black')
            
        fig.suptitle(f"Strict Moving Window (W={window_size}) | Sample: {frame_idx + 1} / {N}", 
                     fontsize=14, fontweight='bold')
        return bars

    # Start the frames list exactly at the first valid full window (window_size - 1)
    start_frame = window_size - 1
    
    # Subsample frames to keep the HTML file manageable (~150 frames total)
    frame_step = max(1, (N - start_frame) // 150)
    frames = list(range(start_frame, N, frame_step))
    
    # Ensure the very last sample of the simulation is included
    if frames[-1] != N - 1:
        frames.append(N - 1)

    print("Building interactive animation... (this might take a minute)")
    ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)

    # 6. Export to Interactive HTML
    save_name = data_path.replace('.pt', f'_spin_strict_moving_avg_W{window_size}_interactive.html')
    
    try:
        html_content = ani.to_jshtml()
        with open(save_name, "w") as f:
            f.write(html_content)
        print(f"Success! Download and open this file in your browser: {save_name}")
    except Exception as e:
        print(f"Error saving interactive HTML: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    # Allows you to dynamically change the window size from the command line
    parser.add_argument("--window", type=int, default=20, help="Window size for the moving average")
    args = parser.parse_args()
    
    if os.path.exists(args.data_path):
        analyze_and_animate_spins(args.data_path, window_size=args.window)
    else:
        print(f"Error: {args.data_path} not found.")