#!/bin/bash -l
#SBATCH --job-name=L128_baseline_xfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/L128_baseline_xfer_%j.out
#SBATCH --error=./logs/L128_baseline_xfer_%j.err

# L=128 GAUSSIAN-PRIOR baseline warm-started from L=64 baseline.
# No CNN transfer possible (Gaussian prior has none). Only MERA blocks
# transferred by scale index. Pure "MERA transfer" test — isolates
# how much the trained MERA (without any HCG CNN structure) helps.
#
# Compare vs:
#   - L128_baseline_freshInit (companion, from-scratch baseline)
#   - L128_MERAonly (HCG-target, MERA from champion, CNN fresh)
#   - L128_from_L64 (both champion transferred)
# to fully attribute physics between MERA and CNN.

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

python -u main.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_baseline_from_L64 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L128_T2.269185314213022_N100000.pt \
    -epochs 15000 -batch 8 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry -alpha 0 \
    -priorType gaussian \
    -loadFromSmallerL data/64Ising_T2.269_hsBignet_baseline_b16/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch19800.saving \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -loadFromSmallerLComponents mera \
    -savePeriod 200 -cuda 0

date
