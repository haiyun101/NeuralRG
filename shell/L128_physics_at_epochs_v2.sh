#!/bin/bash -l
#SBATCH --job-name=L128_phys_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/L128_phys_v2_%j.out
#SBATCH --error=./logs/L128_phys_v2_%j.err

# L=128 physics sampling v2 — fix for original 1801713 issues:
#   (a) Save per-epoch JSON (mv after each run, else overwritten)
#   (b) Focus on PRE-DIVERGENCE checkpoints for warm-start (diverged at ep 6904)
#   (c) Reduced batch size (16 instead of 32) — safer for L=128 target-side comp
#   (d) Include ep 5000 for warm (last healthy) + more granular fresh
#
# Answers: does L=64→L=128 warm-start give physics-meaningful samples
# EARLY (ep 200), or need extra L=128 training?

module load miniforge
source activate neuralrg
mkdir -p logs

WARM=data/L128_T2.269_champion_from_L64
FRESH=data/L128_T2.269_champion_freshInit

# Sampling
N=1000
B=16

# ── L=128 warm-start (pre-divergence checkpoints only) ──
for ep in 200 1000 3000 5000; do
  echo
  echo "==================================================================="
  echo "L=128 WARM-start @ ep $ep"
  echo "==================================================================="
  python -u analyzers/flow_sample_diagnostic.py $WARM \
      -n $N -b $B --seed 0 --epoch $ep --no-png
  # Preserve per-epoch JSON (default writes to flow_diagnostic.json — overwritten each call)
  if [ -f "$WARM/flow_diagnostic.json" ]; then
    mv "$WARM/flow_diagnostic.json" "$WARM/flow_diagnostic_epoch${ep}.json"
    echo "  → saved to flow_diagnostic_epoch${ep}.json"
  fi
done

# ── L=128 fresh init (still healthy at ep 8000+) ──
for ep in 200 1000 3000 5000 8000; do
  echo
  echo "==================================================================="
  echo "L=128 FRESH-init @ ep $ep"
  echo "==================================================================="
  python -u analyzers/flow_sample_diagnostic.py $FRESH \
      -n $N -b $B --seed 0 --epoch $ep --no-png
  if [ -f "$FRESH/flow_diagnostic.json" ]; then
    mv "$FRESH/flow_diagnostic.json" "$FRESH/flow_diagnostic_epoch${ep}.json"
    echo "  → saved to flow_diagnostic_epoch${ep}.json"
  fi
done

echo
echo "Done."
date
