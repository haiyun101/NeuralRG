#!/bin/bash -l
#SBATCH --job-name=tier1_L32
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=./logs/tier1_L32_%j.out
#SBATCH --error=./logs/tier1_L32_%j.err

# Tier 1 physical observables across all key L=32 variants incl. VP winners.
# CPU-only fallback: sampling 10k configs from a flow at L=32 is CPU-tractable
# (~30 min per model).

module load miniforge
source activate neuralrg

mkdir -p logs

N=10000

declare -a RUNS=(
    "GT                       GT                                                                                          -"
    "D_i2_nr2                 data/32Ising_T2.269_hsBignet_i2_stride8h32_nr2_b64                                           10796"
    "A_Gaussian_nr1           data/32Ising_T2.269_hsBignet_baseline_b64                                                    19624"
    "HCG_shared_nr1           data/32Ising_T2.269_hsBignet_hcg_shared_b64                                                  9699"
    "HCG_shared_nr2           data/32Ising_T2.269_hsBignet_hcg_shared_nr2_b64                                              14612"
    "HCG_nodilate_E2_nr2      data/32Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_adam_nr2_gc5.0_b64             7707"
    "HCG_fixdil_nr1           data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_gc5.0_b64                                   19800"
    "HCG_fixdil_VP1e-3_nr1    data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64                                  9999"
    "HCG_shared_VP1e-5_nr1    data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-5_b64                                           9999"
)

for row in "${RUNS[@]}"; do
    read -r label folder epoch <<<"$row"
    echo
    echo "==================================================================="
    echo "== $label"
    echo "==================================================================="
    if [ "$folder" = "GT" ]; then
        python -u analyzers/tier1_observables.py \
            --folder GT --L 32 --T 2.269185314213022 --N $N \
            --label "$label" --device cpu
    else
        python -u analyzers/tier1_observables.py \
            --folder "$folder" --epoch "$epoch" --N $N \
            --label "$label" --device cpu
    fi
done

echo
echo "==================================================================="
echo "== TIER1 SUMMARY ==="
echo "==================================================================="
grep "^\[TIER1_ROW\]" ./logs/tier1_L32_${SLURM_JOB_ID}.out || true

echo "Done."
