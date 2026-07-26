#!/bin/bash -l
#SBATCH --job-name=per_block_jac
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/per_block_jac_%j.out
#SBATCH --error=./logs/per_block_jac_%j.err

# Per-block log|det J_MERA| for champion (VP=1e-3) vs Gaussian baseline.
# Same 10-block MERA arch, same HS input batch, only priors + VP flag differ.
#
# L=32 pair:
#   champion @ ep 9500  (fixdil+VP-1e-3 nr=1, HCG per-scale)
#   baseline @ ep 19800 (Gaussian prior, no VP)
#
# L=64 pair added for cross-L consistency check.

module load miniforge
source activate neuralrg
mkdir -p logs analyzers/csv

echo "==================================================================="
echo "=== L=32:  champion vs baseline"
echo "==================================================================="
python -u analyzers/rg_fixed_point/per_block_jacobian.py \
    --N 1000 --device cpu --seed 0 \
    --out analyzers/csv/per_block_jacobian_L32.csv \
    --cells \
        L32_champion:data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64:9500 \
        L32_baseline:data/32Ising_T2.269_hsBignet_baseline_b64:19800

echo
echo "==================================================================="
echo "=== L=64:  champion vs baseline"
echo "==================================================================="
python -u analyzers/rg_fixed_point/per_block_jacobian.py \
    --N 500 --device cpu --seed 0 \
    --out analyzers/csv/per_block_jacobian_L64.csv \
    --cells \
        L64_champion:data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16:13500 \
        L64_baseline:data/64Ising_T2.269_hsBignet_baseline_b16:19800

echo
echo "Done."
date
