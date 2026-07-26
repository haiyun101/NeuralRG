#!/bin/bash -l
#SBATCH --job-name=per_block_vp_sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=./logs/per_block_vp_sweep_%j.out
#SBATCH --error=./logs/per_block_vp_sweep_%j.err

# Per-block log|det J| sweep across VP lambda values.
# Same MERA arch (nlayers=16, nhidden=128, nrepeat=1), same HS input batch,
# only VP penalty strength differs. All at ep 9500 for fair comparison.
#
# Also includes baseline (no VP) for reference.

module load miniforge
source activate neuralrg
mkdir -p logs analyzers/csv

python -u analyzers/rg_fixed_point/per_block_jacobian.py \
    --N 1000 --device cpu --seed 0 \
    --out analyzers/csv/per_block_jacobian_vp_sweep.csv \
    --cells \
        baseline_no_VP:data/32Ising_T2.269_hsBignet_baseline_b64:19800 \
        VP_1e-5:data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-5_b64:9500 \
        VP_1e-4:data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_b64:9500 \
        VP_1e-3_champion:data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64:9500 \
        VP_1e-2:data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_b64:9500

echo "Done."
date
