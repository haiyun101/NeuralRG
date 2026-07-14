#!/bin/bash -l
#SBATCH --job-name=shared_L32
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/shared_L32_%j.out
#SBATCH --error=./logs/shared_L32_%j.err

# Layer-by-layer analyses on shared HCG L=32 winners.
# Analyses that apply:
#   - MERA layer L2/cosine (same as per-scale — flow structure comparison)
#   - σ output per level (same single CNN applied at 4 different levels)
#   - σ-law scatter per level (does the shared CNN express distinct laws
#     via context density, or is the response scale-invariant?)
# NOT applicable: cross-level CNN weight cosine (only 1 CNN in shared).

module load miniforge
source activate neuralrg

mkdir -p logs figures/sigma_law_shared

echo "=== MERA layer stats: shared nr=1 (best@ep 9699) ==="
python -u analyzers/rg_fixed_point/mera_layer_stats.py \
    --device cpu --epoch 9699 \
    --folder data/32Ising_T2.269_hsBignet_hcg_shared_b64

echo
echo "=== MERA layer stats: shared nr=2 (best@ep 14612) ==="
python -u analyzers/rg_fixed_point/mera_layer_stats.py \
    --device cpu --epoch 14612 \
    --folder data/32Ising_T2.269_hsBignet_hcg_shared_nr2_b64

echo
echo "=== σ-law: shared nr=1 ==="
python -u analyzers/rg_fixed_point/hcg_sigma_law.py \
    --N 500 --device cpu --epoch 9699 \
    --out figures/sigma_law_shared/ \
    --folder data/32Ising_T2.269_hsBignet_hcg_shared_b64

echo
echo "=== σ-law: shared nr=2 ==="
python -u analyzers/rg_fixed_point/hcg_sigma_law.py \
    --N 500 --device cpu --epoch 14612 \
    --out figures/sigma_law_shared/ \
    --folder data/32Ising_T2.269_hsBignet_hcg_shared_nr2_b64

echo "Done."
