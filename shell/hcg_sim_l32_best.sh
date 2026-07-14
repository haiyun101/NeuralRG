#!/bin/bash -l
#SBATCH --job-name=hcg_simL32
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/hcg_simL32_%j.out
#SBATCH --error=./logs/hcg_simL32_%j.err

# Layer-by-layer similarity probe on the BEST L=32 per-scale HCG cells:
#   nr=2 winner: _adam_nr2 (E2 lr=1e-3), best@ep 7707 → use ep 7800
#   nr=1 winner: _adam_lr3e-4_l40, best@ep 16793 → use ep 16800
# Uses --epoch to avoid picking the drifted last checkpoint.

module load miniforge
source activate neuralrg

mkdir -p logs

echo "=== nr=2 winner (best@ep 7707, use ep 7800) ==="
python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --N 500 --device cpu --epoch 7707 \
    --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_nr2_gc5.0_b64

echo
echo "=== nr=1 winner (best@ep 16793, use ep 16800) ==="
python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --N 500 --device cpu --epoch 16793 \
    --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_l40_gc5.0_b64

echo "Done."
