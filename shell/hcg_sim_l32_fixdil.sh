#!/bin/bash -l
#SBATCH --job-name=fixdil_law
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=./logs/fixdil_law_%j.out
#SBATCH --error=./logs/fixdil_law_%j.err

# σ-law visualization on the FIXDIL variant, where Conv0 stays alive at
# every level. Contrast to nodilate (project_hcg_nodilate_beats_fixdil):
# fixdil sees a nontrivial σ(z_slow) even at coarse Levels 1–2 because
# dilation-matched context reaches real coarse-scale information.
#
# Best-epoch ckpts were pruned; using latest (ep 19800). Structural σ
# response is not epoch-sensitive at that magnitude.

module load miniforge
source activate neuralrg

mkdir -p logs figures/sigma_law_fixdil

for F in data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b64 \
         data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64; do
    echo "==== $F ===="
    python -u analyzers/rg_fixed_point/hcg_sigma_law.py \
        --N 500 --device cpu \
        --out figures/sigma_law_fixdil/ \
        --folder "$F"
    echo
done

echo "Done."
