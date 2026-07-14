#!/bin/bash -l
#SBATCH --job-name=hcg_simfx
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=./logs/hcg_simfx_%j.out
#SBATCH --error=./logs/hcg_simfx_%j.err

# Verify dilation fix (d=strides[k] not strides[k-1]) revived Conv0 at
# Levels 2+ in per-scale HCG. Compares:
#   - T_c fixdil nr=1 vs broken (from earlier probe)
#   - T=2.4 fixdil nr=1 (off-critical, first-ever conditional per-scale)

module load miniforge
source activate neuralrg

mkdir -p logs

python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --N 500 \
    --device cpu \
    --folder \
        data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b64 \
        data/32Ising_T2.4_hsBignet_hcg_perscale_fixdil_gc5.0_b64

echo "Done."
