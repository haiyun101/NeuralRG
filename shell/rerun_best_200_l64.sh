#!/bin/bash -l
#SBATCH --job-name=rerun_b200_L64
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/rerun_b200_L64_%j.out
#SBATCH --error=./logs/rerun_b200_L64_%j.err

# Rerun Tier 1 physical observables + flow_sample_diagnostic (KL, plots)
# at the Best-200-center epoch of each top L=64 variant, so all analyses
# are anchored on the same model state that Best-200 loss reports.
#
# Also skips the diverged latest-checkpoint samples for variants like
# fixdil+VP-1e-4 nr=1 whose ep 14500 gave numerical garbage.

module load miniforge
source activate neuralrg

mkdir -p logs figures/tier1_best_200

N_SAMPLES=4000
BATCH=256

echo
echo "=== Extracting Best-200 epochs for L=64 top 15 ==="
python3 analyzers/dump_best_200_epochs.py -L 64 -t 2.269 --top 15 > /tmp/b200_l64.txt
cat /tmp/b200_l64.txt
echo

echo
echo "=== 1) flow_sample_diagnostic at Best-200 epoch ==="
while IFS=$'\t' read -r folder ep S; do
    echo
    echo "==== $folder  @ ep $ep (Best-200 S=$S)"
    # Backup existing flow_diagnostic.json (from latest ckpt) to *_latest.json
    # so we don't lose it for later comparison
    if [ -f "$folder/flow_diagnostic.json" ]; then
        cp "$folder/flow_diagnostic.json" "$folder/flow_diagnostic_latest.json" 2>/dev/null || true
    fi
    python -u analyzers/flow_sample_diagnostic.py "$folder/" \
        -n $N_SAMPLES -b $BATCH --epoch "$ep" \
        > "logs/rerun_b200_diag_$(basename $folder)_ep${ep}.out" 2>&1
    grep -E "ckpt epoch|F_c\^q|KL\(q\|\||KL\(p\|\|" \
        "logs/rerun_b200_diag_$(basename $folder)_ep${ep}.out" | head -6
done < /tmp/b200_l64.txt

echo
echo "=== 2) Tier1 observables at Best-200 epoch ==="
while IFS=$'\t' read -r folder ep S; do
    label=$(basename $folder | sed 's/64Ising_T2.269_hsBignet_//' | head -c 40)
    echo
    echo "==== $label  @ ep $ep"
    python -u analyzers/tier1_observables.py \
        --folder "$folder" --epoch "$ep" --N 4000 \
        --label "b200_${label}"
done < /tmp/b200_l64.txt

echo "Done."
