#!/bin/bash -l
#SBATCH --job-name=tier1_L64
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=./logs/tier1_L64_%j.out
#SBATCH --error=./logs/tier1_L64_%j.err

# Tier 1 physical observables at L=64 (larger — needs GPU for speed).

module load miniforge
source activate neuralrg

mkdir -p logs

N=4000

declare -a RUNS=(
    "GT_L64                       GT                                                                                          -"
    "D_i2_nr2_L64                 data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16                                           16813"
    "HCG_shared_nr2_L64           data/64Ising_T2.269_hsBignet_hcg_shared_nr2_b16                                              16813"
    "HCG_fixdil_VP1e-3_nr1_L64    data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16                             12671"
    "HCG_fixdil_VP1e-3_nr2_L64    data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16                             7234"
    "HCG_fixdil_VP1e-4_nr2_L64    data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16                             4960"
)

for row in "${RUNS[@]}"; do
    read -r label folder epoch <<<"$row"
    echo
    echo "==================================================================="
    echo "== $label"
    echo "==================================================================="
    if [ "$folder" = "GT" ]; then
        python -u analyzers/tier1_observables.py \
            --folder GT --L 64 --T 2.269185314213022 --N $N \
            --label "$label"
    else
        python -u analyzers/tier1_observables.py \
            --folder "$folder" --epoch "$epoch" --N $N \
            --label "$label"
    fi
done

echo
echo "==================================================================="
echo "== TIER1 L=64 SUMMARY ==="
echo "==================================================================="
grep "^\[TIER1_ROW\]" ./logs/tier1_L64_${SLURM_JOB_ID}.out || true

echo "Done."
