#!/bin/bash -l
#SBATCH --job-name=L32_jb_ext
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --output=./logs/L32_jb_ext_%j.out
#SBATCH --error=./logs/L32_jb_ext_%j.err

# Phase-2 extension of the L=32 jsLoss BIGNET run.
# Phase-1 stopped at epoch 3000 with smoothed slope -2.64 nat / 1000 ep (still
# actively descending). Project shows L_js ~ -220 to -223 at full convergence.
# Extend by 5000 epochs (+500 epochs of optimizer-state burn-in -- see
# project_resume_optimizer_state.md for context).
#
# Backup phase-1 records & savings before patch to preserve the trajectory.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="./data/32Ising_T2.269_jsLoss_bignet_lam0.5"
HS_PT="data/mcmc_data/hs_L32_T2.269185314213022_N200000.pt"
EPOCHS_PHASE2=5000

# --- One-time backup + patch (idempotent: skip if backup already exists) ---
if [ ! -d "$FOLDER/records_phase1" ]; then
    echo "Backing up phase-1 records/savings ..."
    mv "$FOLDER/records" "$FOLDER/records_phase1"
    mv "$FOLDER/savings" "$FOLDER/savings_phase1"
    mkdir -p "$FOLDER/records" "$FOLDER/savings"

    # Carry only the latest checkpoint into fresh savings/ for -load
    LATEST=$(ls -t "$FOLDER/savings_phase1"/*.saving | head -1)
    cp "$LATEST" "$FOLDER/savings/"
    echo "  copied checkpoint: $(basename $LATEST)"

    # Patch epochs in parameters.hdf5 (preserve lr=1e-3 -- JS uses fresh Adam)
    python - <<PY
import h5py
with h5py.File("$FOLDER/parameters.hdf5", "r+") as f:
    old = int(f["epochs"][()])
    del f["epochs"]
    f.create_dataset("epochs", data=$EPOCHS_PHASE2)
    print(f"  epochs: {old} -> $EPOCHS_PHASE2")
PY
else
    echo "Backup already exists (records_phase1/); reusing -- no re-patch."
fi

echo ""
echo "=========================================="
echo "L=32 jsLoss BIGNET extension (phase 2)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

# -load reads architecture from parameters.hdf5; training-mode flags must
# still be passed on the CLI (jsLoss/jsLambda/symmetry/skipHMC/noDeq).
python main.py -load -folder "$FOLDER" -cuda 0 \
    -symmetry -skipHMC \
    -jsLoss -jsLambda 0.5 \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done (jsLoss bignet extension)."
