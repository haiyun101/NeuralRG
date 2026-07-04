#!/bin/bash -l
#SBATCH --job-name=hcg_sim
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=./logs/hcg_sim_%j.out
#SBATCH --error=./logs/hcg_sim_%j.err

# Test 2: per-scale HCG CNN cross-level similarity.
# CPU-only, weights-only analysis. Fast.

module load miniforge
source activate neuralrg

mkdir -p logs

python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --folder \
        data/32Ising_T2.269_hsBignet_hcg_perscale_b64 \
        data/32Ising_T2.269_hsBignet_hcg_perscale_nr2_b64 \
        data/64Ising_T2.269_hsBignet_hcg_perscale_b16

echo "Done."
