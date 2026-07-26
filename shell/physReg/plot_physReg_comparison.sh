#!/bin/bash -l
#SBATCH --job-name=plot_physReg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/plot_physReg_%j.out
#SBATCH --error=./logs/plot_physReg_%j.err

# Physics comparison plots for physReg experiments.
# Compares plain champion vs physReg at 3 different lambda values.
# Same 4 plot types as L=128 transfer report.

module load miniforge
source activate neuralrg
mkdir -p logs figures/physReg/L32 figures/physReg/L64

# ── L=32 physReg sweep comparison ──
echo "==================================================================="
echo "=== L=32 physReg comparison (plain champion vs 3 sweep cells) ==="
echo "==================================================================="
python -u analyzers/physReg/plot_physReg_comparison.py \
    --N 1500 --batch 64 --device cuda \
    --out figures/physReg/L32 \
    --cells \
        "plain_champion:data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64:9500" \
        "physReg_λ0.01:data/32Ising_T2.269_physReg_chi0.01_u40.01:800" \
        "physReg_λ0.1:data/32Ising_T2.269_physReg_chi0.1_u40.1:800" \
        "physReg_λ1.0:data/32Ising_T2.269_physReg_chi1.0_u41.0:800"

# ── L=64 physReg comparison ──
echo
echo "==================================================================="
echo "=== L=64 physReg comparison (plain champion vs warm+physReg vs fresh+physReg) ==="
echo "==================================================================="
python -u analyzers/physReg/plot_physReg_comparison.py \
    --N 1500 --batch 32 --device cuda \
    --out figures/physReg/L64 \
    --cells \
        "plain_champion:data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16:9500" \
        "warm_physReg_bf16:data/L64_T2.269_champion_physReg_chi0.1_u40.1:5000" \
        "warm_physReg_fp32_v2:data/L64_T2.269_champion_physReg_chi0.1_u40.1_fp32_v2:5000" \
        "fresh_physReg_v2:data/L64_T2.269_physReg_fresh_chi0.05_u40.05_v2:5000"

echo
echo "Done."
date
