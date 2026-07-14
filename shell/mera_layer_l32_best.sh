#!/bin/bash -l
#SBATCH --job-name=mera_L32
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/mera_L32_%j.out
#SBATCH --error=./logs/mera_L32_%j.err

# Layer-by-layer MERA weight analysis on best-checkpoint of L=32 winners.
# Companion to hcg_perscale_similarity.py (which covers the CNN side).
# Together they show whether the winning per-scale HCG structures MERA and
# CNN into matching "coarse-dead / fine-active" bands.

module load miniforge
source activate neuralrg

mkdir -p logs

echo "=== nr=2 winner (best@ep 7707) ==="
python -u analyzers/rg_fixed_point/mera_layer_stats.py \
    --device cpu --epoch 7707 \
    --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_nr2_gc5.0_b64

echo
echo "=== nr=1 winner (best@ep 16793) ==="
python -u analyzers/rg_fixed_point/mera_layer_stats.py \
    --device cpu --epoch 16793 \
    --folder data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_l40_gc5.0_b64

echo "Done."
