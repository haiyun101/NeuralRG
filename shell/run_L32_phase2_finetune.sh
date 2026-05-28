#!/bin/bash -l
#SBATCH --job-name=L32_p2_ft
#SBATCH --partition=preempt
# Bignet (10.9M params) reverse-KL needs > 16 GB VRAM; 16-GB cards (T4/P100)
# OOM. l40 (48 GB) has plenty of headroom; override with --gres=gpu:a100:1
# at submit time if l40 is busy.
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=./logs/L32_p2_ft_%j.out
#SBATCH --error=./logs/L32_p2_ft_%j.err

# Phase 2 finetune: load the trained L=32 hs_bignet (Phase 1 forward-KL,
# epoch 9500, sigma=3.5076) and continue with reverse-KL at low LR.
#
# Phase 1 built the bridge between physical modes (mass-covering).
# Phase 2 sharpens the distribution to match exact Boltzmann weights.
# Low LR is critical -- high LR would snap the flow back into a single mode.
#
# Implementation note: the existing reverse-KL branch now auto-detects the
# inherited sigma from flow_input_sigma.json and converts u -> physical x
# before evaluating the action source.logProbability(x).

module load miniforge
source activate neuralrg

mkdir -p logs

SRC="./data/32Ising_T2.269_hs_bignet"
DST="./data/32Ising_T2.269_phase2_finetune"
EPOCHS_PHASE2=2000
LR_PHASE2=1e-5

# Fresh copy each submission, so re-running this script always starts cleanly
# from Phase 1's epoch-9500 checkpoint.
echo "Cloning $SRC -> $DST ..."
rm -rf "$DST"
mkdir -p "$DST/savings" "$DST/records"
cp "$SRC/parameters.hdf5"        "$DST/parameters.hdf5"
cp "$SRC/flow_input_sigma.json"  "$DST/flow_input_sigma.json"
# Carry only the LAST checkpoint -- -load uses max ctime
LATEST=$(ls -t "$SRC/savings"/*.saving | head -1)
cp "$LATEST" "$DST/savings/"
echo "  Copied checkpoint: $(basename $LATEST)"

# Patch parameters.hdf5: new epoch budget + low LR for Phase 2
python - <<PY
import h5py
with h5py.File("$DST/parameters.hdf5", "r+") as f:
    for k, v in [("epochs", $EPOCHS_PHASE2), ("lr", $LR_PHASE2)]:
        old = f[k][()]
        del f[k]
        f.create_dataset(k, data=v)
        print(f"  {k}: {old} -> {v}")
PY

echo ""
echo "=========================================="
echo "L=32 Phase-2 finetune  |  reverse-KL, LR=$LR_PHASE2"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $DST"
echo "=========================================="

python main.py -load -folder "$DST" -cuda 0 -symmetry -skipHMC

echo "Done (Phase 2 finetune)."
