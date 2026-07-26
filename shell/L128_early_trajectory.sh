#!/bin/bash -l
#SBATCH --job-name=L128_early4x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L128_early4x_%j.out
#SBATCH --error=./logs/L128_early4x_%j.err

# 4-way early trajectory: warm/fresh/MERAonly/CNNonly at L=128,
# 100 epochs each, savePeriod=5 (gives checkpoints at ep 5, 10, 15, ...,
# 100 including user-requested 5/10/30/50/100).
# Sequential in one job to save queue slots. ~2h per cell = ~8h total.
#
# Key fix: -alpha 0 (disable Z2 symmetry penalty) avoids the second
# forward pass that OOM'd earlier ablation jobs 1805811/1805812.
# -symmetry (Symmetrized wrapper) still enforces Z2 in the flow itself.

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

L64_CKPT=data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving

# Common args (100 epochs, savePeriod=5, alpha=0)
COMMON="-L 128 -T 2.269185314213022 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L128_T2.269185314213022_N100000.pt \
    -epochs 100 -batch 8 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry -alpha 0 \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -savePeriod 5 -cuda 0"

# ==============================================================
# Cell 1: WARM from L=64 (both MERA + CNN transferred)
# ==============================================================
echo "==================================================================="
echo "=== Cell 1: L=128 WARM from L=64 (both MERA + CNN) — 100 epochs"
echo "==================================================================="
python -u main.py $COMMON \
    -folder ./data/L128_early_warm_from_L64 \
    -loadFromSmallerL $L64_CKPT \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -loadFromSmallerLComponents both

# ==============================================================
# Cell 2: FRESH init (nothing transferred)
# ==============================================================
echo
echo "==================================================================="
echo "=== Cell 2: L=128 FRESH init — 100 epochs"
echo "==================================================================="
python -u main.py $COMMON \
    -folder ./data/L128_early_fresh

# ==============================================================
# Cell 3: MERA-only transfer (ablation)
# ==============================================================
echo
echo "==================================================================="
echo "=== Cell 3: L=128 MERA-only from L=64 (CNN fresh)"
echo "==================================================================="
python -u main.py $COMMON \
    -folder ./data/L128_early_MERAonly \
    -loadFromSmallerL $L64_CKPT \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -loadFromSmallerLComponents mera

# ==============================================================
# Cell 4: CNN-only transfer (ablation)
# ==============================================================
echo
echo "==================================================================="
echo "=== Cell 4: L=128 CNN-only from L=64 (MERA fresh)"
echo "==================================================================="
python -u main.py $COMMON \
    -folder ./data/L128_early_CNNonly \
    -loadFromSmallerL $L64_CKPT \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -loadFromSmallerLComponents cnn

echo
echo "Done."
date
