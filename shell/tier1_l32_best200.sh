#!/bin/bash -l
#SBATCH --job-name=tier1_L32_b200
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/tier1_L32_b200_%j.out
#SBATCH --error=./logs/tier1_L32_b200_%j.err

module load miniforge
source activate neuralrg
mkdir -p logs

# Get top-15 L=32 variants with Best-200 epochs
python3 analyzers/dump_best_200_epochs.py -L 32 -t 2.269 --top 15 > /tmp/b200_l32.txt
echo "Cells to analyze:"
cat /tmp/b200_l32.txt
echo

# Also add GT reference row
echo
echo "=== GT_L32 (Wolff MCMC, T=T_c) ==="
python -u analyzers/tier1_observables.py --folder GT --L 32 \
    --T 2.269185314213022 --N 10000 --label "GT_L32"

# Then each variant at its Best-200 epoch
while IFS=$'\t' read -r folder ep S; do
    label=$(basename $folder | sed 's/32Ising_T2.269_hsBignet_//' | head -c 45)
    echo
    echo "=== b200_${label} @ ep $ep ==="
    python -u analyzers/tier1_observables.py \
        --folder "$folder" --epoch "$ep" --N 10000 \
        --label "b200_${label}"
done < /tmp/b200_l32.txt

echo
echo "==================================================================="
echo "== TIER1 L=32 Best-200 SUMMARY"
echo "==================================================================="
grep "^\[TIER1_ROW\]" logs/tier1_L32_b200_${SLURM_JOB_ID}.out || true

echo "Done."
