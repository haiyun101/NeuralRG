#!/bin/bash -l
#SBATCH --job-name=tier1_gt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/tier1_gt_%j.out
#SBATCH --error=./logs/tier1_gt_%j.err

# Ground truth physical observables at multiple temperatures. Uses HS
# Wolff MCMC samples as reference. Produces a reference physics curve
# spanning the transition: below-T_c (2.15, 2.22), at T_c (2.269), above
# T_c (2.32, 2.40). Each row is a [TIER1_ROW] line with E, absM, M², χ, U₄, G(r).

module load miniforge
source activate neuralrg

mkdir -p logs

N=10000
declare -a TEMPS=(2.15 2.22 2.269185314213022 2.32 2.4)
declare -a LABELS=(GT_T2.15 GT_T2.22 GT_Tc GT_T2.32 GT_T2.4)

for i in "${!TEMPS[@]}"; do
    T="${TEMPS[$i]}"
    LABEL="${LABELS[$i]}"
    echo
    echo "==================================================================="
    echo "== $LABEL  (T=$T)"
    echo "==================================================================="
    python -u analyzers/tier1_observables.py \
        --folder GT --L 32 --T "$T" --N $N \
        --label "$LABEL" --device cpu
done

echo
echo "==================================================================="
echo "== GT SUMMARY ==="
echo "==================================================================="
grep "^\[TIER1_ROW\]" "./logs/tier1_gt_${SLURM_JOB_ID}.out" || true

echo "Done."
