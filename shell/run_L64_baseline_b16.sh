#!/bin/bash -l
#SBATCH --job-name=L64_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=./logs/L64_base_%j.out
#SBATCH --error=./logs/L64_base_%j.err

# L=64 hs_bignet baseline at batch=16 — control for the L=64 III.1
# scaleLoss ablation. batch=16 (down from the b=32 used in
# run_L64_hs_bignet.sh) so the matched-batch comparison against iii1
# is exact (iii1 needs the smaller batch to fit the extra forward graph
# for the scale loss). Same architecture / epochs / symmetry otherwise.
#
# Tunables via env:
#   BATCH=16
#   NLAYERS=16     (RNVP block depth)
#   NHIDDEN=128    (MLP hidden channels — set 192 for megabignet)
#   LR=1e-3        (learning rate; lower to 5e-4 for megabignet bigger param space)
#   GRADCLIP=0     (set to 5.0 to bound noisy gradients in megabignet)
#   EPOCHS=20000
#   FOLDER_SUFFIX  (default _b${BATCH}; auto-suffixes for non-default nlayers/nhidden/lr/gradClip)

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
N="${N:-200000}"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-16}"
NLAYERS="${NLAYERS:-16}"
NHIDDEN="${NHIDDEN:-128}"
LR="${LR:-1e-3}"
GRADCLIP="${GRADCLIP:-0}"
# Build folder suffix from any non-default tunables (don't bake lr if it's still 1e-3, etc.)
SUFFIX=""
[ "$NLAYERS" != "16" ] || [ "$NHIDDEN" != "128" ] && SUFFIX="${SUFFIX}_l${NLAYERS}h${NHIDDEN}"
[ "$LR" != "1e-3" ] && SUFFIX="${SUFFIX}_lr${LR}"
[ "$GRADCLIP" != "0" ] && [ "$GRADCLIP" != "0.0" ] && SUFFIX="${SUFFIX}_gc${GRADCLIP}"
[ "$N" != "200000" ] && SUFFIX="${SUFFIX}_N${N}"
SUFFIX="${SUFFIX}_b${BATCH}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-$SUFFIX}"
FOLDER="./data/${L}Ising_T2.269_hsBignet_baseline${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet baseline (batch=$BATCH, no scaleLoss)"
echo "  nlayers=$NLAYERS  nhidden=$NHIDDEN  lr=$LR  gradClip=$GRADCLIP  epochs=$EPOCHS"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch "$BATCH" \
    -nlayers "$NLAYERS" -nmlp 3 -nhidden "$NHIDDEN" -nrepeat 1 \
    -lr "$LR" \
    -gradClip "$GRADCLIP" \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
