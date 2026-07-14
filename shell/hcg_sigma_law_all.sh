#!/bin/bash -l
#SBATCH --job-name=sigma_law_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=./logs/sigma_law_all_%j.out
#SBATCH --error=./logs/sigma_law_all_%j.err

# σ-law + marginal-Gaussianize calibration table across all L=32 HCG
# variants, at each variant's best-epoch checkpoint. Produces a
# [MARGINAL_TABLE] block per variant for cross-variant comparison of
# how well-calibrated the CNN's σ² is against the empirical Var(z_fast).

module load miniforge
source activate neuralrg

mkdir -p logs figures/sigma_law_all

# (variant, folder, best_epoch)
declare -a RUNS=(
    "shared_nr1         data/32Ising_T2.269_hsBignet_hcg_shared_b64                                                 9699"
    "shared_nr2         data/32Ising_T2.269_hsBignet_hcg_shared_nr2_b64                                             14612"
    "nodilate_nr1       data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_lr3e-4_l40_gc5.0_b64     16793"
    "nodilate_nr2       data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_nr2_gc5.0_b64            7707"
    "fixdil_nr1         data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b64                                  19800"
    "fixdil_nr2         data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_nr2_gc5.0_b64                              19800"
)

for row in "${RUNS[@]}"; do
    read -r label folder epoch <<<"$row"
    echo
    echo "==================================================================="
    echo "== $label (epoch $epoch) — $folder"
    echo "==================================================================="
    python -u analyzers/rg_fixed_point/hcg_sigma_law.py \
        --N 500 --device cpu --epoch "$epoch" \
        --out figures/sigma_law_all/ \
        --folder "$folder"
done

echo "Done."
