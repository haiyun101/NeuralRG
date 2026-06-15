#!/bin/bash -l
#SBATCH --job-name=mb_extend
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=./logs/mb_extend_%j.out
#SBATCH --error=./logs/mb_extend_%j.err

# Megabignet extend-training launcher (Phase-2 P2.x verdict follow-up).
# Loads from existing megabignet checkpoint (ep 19800, plateau ~7724) and
# continues for EXTRA_EPOCHS with optional cosine LR schedule and/or a
# larger HS dataset.
#
# Env vars:
#   ORIG_FOLDER  source folder (must contain parameters.hdf5 + savings/)
#                default: ./data/64Ising_T2.269_hsBignet_baseline_l16h192_lr5e-4_gc5.0_b16/
#   DATA_N       which HS dataset (200000 or 500000); default 200000
#   EXTRA_EPOCHS additional epochs to train; default 10000
#   COSINE       1 enables -cosineAnneal; default 0
#   LR           override -lr (only meaningful for the cosine peak LR);
#                default = inherits from parameters.hdf5 (5e-4 for megabignet)
#   ETA_MIN      cosineEtaMin floor; default lr*0.01
#   TAG          extra suffix label for the new folder; default = auto
#
# Output folder pattern:
#   <ORIG_FOLDER>_ext_N${DATA_N}_e${EXTRA_EPOCHS}[_cos][_lr${LR}][_tag${TAG}]

module load miniforge
source activate neuralrg

mkdir -p logs

ORIG_FOLDER="${ORIG_FOLDER:-./data/64Ising_T2.269_hsBignet_baseline_l16h192_lr5e-4_gc5.0_b16/}"
DATA_N="${DATA_N:-200000}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-10000}"
COSINE="${COSINE:-0}"
LR="${LR:-}"
ETA_MIN="${ETA_MIN:-}"
TAG="${TAG:-}"

T=2.269185314213022
L=64
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${DATA_N}.pt"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi
if [ ! -f "${ORIG_FOLDER%/}/parameters.hdf5" ]; then
    echo "ERROR: ORIG_FOLDER missing parameters.hdf5: $ORIG_FOLDER"
    exit 1
fi

# Build new folder name from non-default knobs
ORIG_NOSLASH="${ORIG_FOLDER%/}"
SUFFIX="_ext_N${DATA_N}_e${EXTRA_EPOCHS}"
[ "$COSINE" = "1" ] && SUFFIX="${SUFFIX}_cos"
[ -n "$LR" ] && SUFFIX="${SUFFIX}_lr${LR}"
[ -n "$TAG" ] && SUFFIX="${SUFFIX}_${TAG}"
NEW_FOLDER="${ORIG_NOSLASH}${SUFFIX}/"

echo "=========================================="
echo "Megabignet EXTEND"
echo "  source: $ORIG_FOLDER"
echo "  target: $NEW_FOLDER"
echo "  data:   $HS_PT  (N=$DATA_N)"
echo "  extra epochs: $EXTRA_EPOCHS  cosine=$COSINE  lr_override='$LR'"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

mkdir -p "$NEW_FOLDER"

# 1) Copy parameters.hdf5 + savings/ + records/ — only on first run
if [ ! -f "${NEW_FOLDER%/}/parameters.hdf5" ]; then
    cp "${ORIG_NOSLASH}/parameters.hdf5" "${NEW_FOLDER%/}/parameters.hdf5"
    mkdir -p "${NEW_FOLDER%/}/savings"
    # Only need the LATEST saving for -load (glob picks newest by ctime),
    # but copy all to be safe + auditable.
    cp "${ORIG_NOSLASH}/savings/"*.saving "${NEW_FOLDER%/}/savings/" 2>/dev/null
    if [ -d "${ORIG_NOSLASH}/records" ]; then
        mkdir -p "${NEW_FOLDER%/}/records"
        cp "${ORIG_NOSLASH}/records/"*.hdf5 "${NEW_FOLDER%/}/records/" 2>/dev/null
    fi

    # 2) Patch parameters.hdf5 so -load uses EXTRA_EPOCHS as the new range(epochs)
    #    (also let the loaded lr be overridden if requested)
    python -u <<PYEOF
import h5py, sys, os
folder = "${NEW_FOLDER%/}"
extra = ${EXTRA_EPOCHS}
lr_override = "${LR}".strip()
with h5py.File(folder + "/parameters.hdf5", "r+") as f:
    print("  before: epochs=", int(f["epochs"][...]), " lr=", float(f["lr"][...]))
    del f["epochs"]
    f.create_dataset("epochs", data=extra)
    if lr_override:
        del f["lr"]
        f.create_dataset("lr", data=float(lr_override))
    print("  after:  epochs=", int(f["epochs"][...]), " lr=", float(f["lr"][...]))
PYEOF
fi

# 3) Build training command
CMD="python -u main.py -load -folder $NEW_FOLDER -cuda 0 \
    -skipHMC -dataDriven -dataPath $HS_PT -noDeq -symmetry \
    -gradClip 5.0"

if [ "$COSINE" = "1" ]; then
    CMD="$CMD -cosineAnneal"
    [ -n "$ETA_MIN" ] && CMD="$CMD -cosineEtaMin $ETA_MIN"
fi

echo "CMD: $CMD"
echo "----"
eval $CMD
echo "Done."
