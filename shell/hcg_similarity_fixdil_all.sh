#!/bin/bash -l
#SBATCH --job-name=hcg_simA
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/hcg_simA_%j.out
#SBATCH --error=./logs/hcg_simA_%j.err

# Layer-by-layer similarity probe on ALL fixdil per-scale HCG cells:
#   L=32 T_c nr=1  (already probed, included for consistency)
#   L=32 T_c nr=2  (NEW)
#   L=32 T=2.4 nr=1  (already probed)
#   L=64 T_c nr=1  (NEW)
#   L=64 T_c nr=2  (NEW — from ep 19400 checkpoint while training finishes)
#
# For each: Conv0/Conv1/Conv2 weight L2 + σ output stats + swap test.
# Compare to broken-dilation reference numbers to confirm mechanism fix.

module load miniforge
source activate neuralrg

mkdir -p logs

python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --N 500 \
    --device cpu \
    --folder \
        data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b64 \
        data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64 \
        data/32Ising_T2.4_hsBignet_hcg_perscale_fixdil_gc5.0_b64 \
        data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b16 \
        data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b16

echo "Done."
