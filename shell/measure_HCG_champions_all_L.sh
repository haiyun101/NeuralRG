#!/bin/bash -l
#SBATCH --job-name=hcg_phys_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/hcg_phys_all_%j.out
#SBATCH --error=./logs/hcg_phys_all_%j.err

# Measure spin observables (|M|, χ, U₄) for HCG-CNN champions at all
# 3 lattice sizes. Fills the L=64 gap in hcg_vs_meraFM_comparison.md
# (previous physics analyses saved only HS-field basis at L=64).

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

L32_CHAMP=data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64
L64_CHAMP=data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16
L128_WARM=data/L128_T2.269_champion_from_L64      # pre-divergence, ep 5000

python -u analyzers/physReg/measure_physReg_effect.py \
    --N 2000 --batch 32 --device cuda \
    --cells \
        "L32_HCG_champion:$L32_CHAMP:9500" \
        "L64_HCG_champion:$L64_CHAMP:9500" \
        "L128_HCG_warm_ep5000:$L128_WARM:5000"

echo
echo "=== L=32 GT: |M|=0.6544  χ=31.61  U4=0.6110 ==="
echo "=== L=64 GT: |M|=0.6004  χ=106.03  U4=0.6109 ==="
echo "=== L=128 GT: |M|=0.5507  χ=357.40  U4=0.6109 ==="
date
