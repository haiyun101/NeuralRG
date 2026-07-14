#!/bin/bash -l
#SBATCH --job-name=vp_layer_b200
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=./logs/vp_layer_b200_%j.out
#SBATCH --error=./logs/vp_layer_b200_%j.err

# Layer-by-layer analysis of top L=64 forward-KL variants at their
# Best-200-center epoch (consistent with the loss ranking and physics
# observables).
#
# Runs 3 probes per cell:
#   A. hcg_perscale_similarity.py — cross-level CNN weight+sigma+swap
#   B. mera_layer_stats.py        — MERA per-layer L2 + cosine profile
#   C. hcg_sigma_law.py           — σ vs local z_slow scalar summaries + calibration

module load miniforge
source activate neuralrg

mkdir -p logs figures/vp_layer_b200

# Get top-10 cells with their Best-200 epochs
python3 analyzers/dump_best_200_epochs.py -L 64 -t 2.269 --top 10 > /tmp/b200_l64_layer.txt
echo "=== Cells to analyze ==="
cat /tmp/b200_l64_layer.txt
echo

N_SIM=200
N_LAW=200

while IFS=$'\t' read -r folder ep S; do
    label=$(basename $folder | sed 's/64Ising_T2.269_hsBignet_//')
    echo
    echo "==================================================================="
    echo "==== $label  @ ep $ep (Best-200 S=$S)"
    echo "==================================================================="

    # A. per-scale similarity: only meaningful for per-scale HCG variants;
    # skip shared or non-HCG (baseline/D). The script itself will detect
    # and print a skip message.
    echo "--- A. CNN cross-level similarity ---"
    python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
        --N $N_SIM --device cpu --epoch $ep \
        --folder "$folder" 2>&1 | tail -50

    echo
    echo "--- B. MERA per-layer profile ---"
    python -u analyzers/rg_fixed_point/mera_layer_stats.py \
        --device cpu --epoch $ep \
        --folder "$folder" 2>&1 | tail -50

    echo
    echo "--- C. σ-law + calibration ---"
    python -u analyzers/rg_fixed_point/hcg_sigma_law.py \
        --N $N_LAW --device cpu --epoch $ep \
        --out figures/vp_layer_b200/ \
        --folder "$folder" 2>&1 | tail -50
done < /tmp/b200_l64_layer.txt

echo "Done."
