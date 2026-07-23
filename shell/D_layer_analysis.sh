#!/bin/bash -l
#SBATCH --job-name=D_layer_analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/D_layer_analysis_%j.out
#SBATCH --error=./logs/D_layer_analysis_%j.err

# Third attempt at D flow_capture. Previous attempts:
#  - 41803213 (in A+D batch): walltime cut at 12h, D got no output.
#  - 41812111 (D-only, 6h, N=2000): walltime cut, got through fwd 5/6
#    but the 464 MB .pt save never happened.
#
# Fix: reduce N=500 (4× speedup) + extend walltime to 12h. Small N is
# enough for cross-scale similarity metrics (500 samples is plenty for
# MMD / W1 estimation on the 2×2-projected activations).

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER=data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16
EPOCH=17754

echo "=========================================="
echo "L=64 D layer-level analysis (third try, N=500)"
echo "  target: $FOLDER"
echo "  Best-200 epoch: $EPOCH"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

# ── (1/2) MERA per-layer activation capture ──
echo
echo ">>>>>>>>>> [1/2] mera_layer_flow_capture (N=500) <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/mera_layer_flow_capture.py \
    --folder "$FOLDER" --epoch "$EPOCH" \
    --N 500 --device cpu \
    || { echo "STEP 1 FAILED"; exit 1; }

# ── (2/2) MERA per-layer weight stats ──
echo
echo ">>>>>>>>>> [2/2] mera_layer_stats <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/mera_layer_stats.py \
    --folder "$FOLDER" --epoch "$EPOCH" --device cpu \
    || echo "STEP 2 FAILED (non-fatal)"

echo
echo "=========================================="
echo "Done. Expected outputs:"
echo "  - $FOLDER/mera_layer_flow_capture.pt (should be ~120 MB at N=500)"
echo "  - $FOLDER/mera_layer_flow_capture.json"
echo "  - stdout: mera_layer_stats table (L2 norms + inter-block cosine)"
echo "=========================================="
date
