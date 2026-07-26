#!/bin/bash -l
#SBATCH --job-name=cross_L_cnn_ss
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=./logs/cross_L_cnn_ss_%j.out
#SBATCH --error=./logs/cross_L_cnn_ss_%j.err

# Cross-L HCG CNN weight similarity: does L=128 transfer preserve the
# self-similar structure L=64 learned? Do L=128 warm and fresh converge
# to similar CNN weights?

module load miniforge
source activate neuralrg
mkdir -p logs

python -u analyzers/rg_fixed_point/cross_L_cnn_similarity.py \
    --device cpu \
    --cells \
        "L64_champion:data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16:9500" \
        "L128_warm_ep5000:data/L128_T2.269_champion_from_L64:5000" \
        "L128_fresh_ep8000:data/L128_T2.269_champion_freshInit:8000"

echo
echo "Done."
date
