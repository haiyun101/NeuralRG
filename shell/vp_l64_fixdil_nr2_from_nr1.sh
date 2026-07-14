#!/bin/bash -l
#SBATCH --job-name=vpL64_fixdil_nr2_fN1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=./logs/vpL64_fixdil_nr2_fN1_%j.out
#SBATCH --error=./logs/vpL64_fixdil_nr2_fN1_%j.err

# L=64 fixdil nr=2 + VP-1e-3, WARM-STARTED from the nr=1 champion
# (fixdil+VP-1e-3 nr=1, Best-200 = 7658.61).
#
# Uses analyzers/convert_nr1_to_nr2_saving.py to double the layerList:
#   - nr=1 layers 0..11 → nr=2 rep-0 slots (0,1,4,5,8,9,...)
#   - nr=2 rep-1 slots (2,3,6,7,...) initialized to identity
#     (zero final-Linear + ScalableTanh scale in every t/s MLP)
# → nr=2 forward output at ep 0 is IDENTICAL to nr=1's forward output.
#
# Then training continues from that state with fresh Adam. The fresh rep-1
# blocks are free to move; the pre-trained rep-0 blocks preserve the
# champion basin. VP-1e-3 continues on the doubled model.
#
# Env vars:
#   SRC_EPOCH   (default: latest = ep 15000)
#   VP_LAMBDA   (default: 1e-3, matches nr=1 champion)
#   EPOCHS      (default: 15000)

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T="2.269185314213022"
N="200000"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
VP_LAMBDA="${VP_LAMBDA:-1e-3}"
SRC_EPOCH="${SRC_EPOCH:-}"
EPOCHS="${EPOCHS:-15000}"
BATCH="${BATCH:-16}"

SRC_FOLDER="./data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp${VP_LAMBDA}_nr1_b16"
if [ ! -d "$SRC_FOLDER/savings" ]; then
    echo "ERROR: nr=1 source folder missing: $SRC_FOLDER"
    exit 1
fi

DST_FOLDER="./data/${L}Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp${VP_LAMBDA}_fromnr1_nr2_b${BATCH}"

# ── STEP 1: Convert nr=1 checkpoint → nr=2 identity-init checkpoint ────
if [ ! -f "$DST_FOLDER/savings/"*.saving 2>/dev/null ] && [ ! -e "$DST_FOLDER/parameters.hdf5" ]; then
    echo "===== convert nr=1 → nr=2 (identity-init rep-1) ====="
    EPOCH_ARG=""
    [ -n "$SRC_EPOCH" ] && EPOCH_ARG="--epoch $SRC_EPOCH"
    python3 analyzers/convert_nr1_to_nr2_saving.py \
        --src "$SRC_FOLDER" \
        --dst-folder "$DST_FOLDER" \
        $EPOCH_ARG \
        --overwrite
    if [ $? -ne 0 ]; then
        echo "ERROR: converter failed"
        exit 1
    fi
else
    echo "===== dst folder exists — skipping convert step ====="
fi

# ── STEP 2: Launch training from the converted checkpoint ──────────────
# main.py's -load restores nrepeat=2 (converter wrote it into HDF5) and
# every other arg from parameters.hdf5. BUT: volumePreservingWeight is
# saved to HDF5 but not restored by -load (main.py:90-166), so we pass
# it explicitly on the CLI to override the argparse default (0.0).
echo "=========================================="
echo "L=$L fixdil nr=2 + VP-${VP_LAMBDA}, warm-start from nr=1 champion"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  source (nr=1):     $SRC_FOLDER"
echo "  target (nr=2):     $DST_FOLDER"
echo "=========================================="

python -u main.py \
    -load -folder "$DST_FOLDER" \
    -cuda 0 \
    -symmetry \
    -volumePreservingWeight "$VP_LAMBDA"

echo "Done."
