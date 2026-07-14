#!/bin/bash -l
#SBATCH --job-name=sigma_law
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=./logs/sigma_law_%j.out
#SBATCH --error=./logs/sigma_law_%j.err

# Option B (scatter of σ vs 4 scalar summaries of z_slow)
# + Plot 3 (predicted σ² vs empirical Var(z_fast|bin))
# on the L=32 winners at their best epoch.

module load miniforge
source activate neuralrg

mkdir -p logs figures/sigma_law

echo "=== nr=2 winner (best@ep 7707) ==="
python -u analyzers/rg_fixed_point/hcg_sigma_law.py \
    --N 500 --device cpu --epoch 7707 \
    --out figures/sigma_law/ \
    --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_nr2_gc5.0_b64

echo
echo "=== nr=1 winner (best@ep 16793) ==="
python -u analyzers/rg_fixed_point/hcg_sigma_law.py \
    --N 500 --device cpu --epoch 16793 \
    --out figures/sigma_law/ \
    --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_l40_gc5.0_b64

echo "Done."
