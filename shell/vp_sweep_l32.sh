#!/bin/bash -l
#SBATCH --job-name=vp_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/vp_sweep_%j.out
#SBATCH --error=./logs/vp_sweep_%j.err

# Volume-preserving penalty sweep on L=32 shared HCG (best baseline).
# Tests whether forcing log|det J_MERA| toward 0 rescues σ calibration
# without collapsing total LOSS. One λ per job — submit multiple.
#
# Env vars:
#   VP_LAMBDA (required, e.g. 1e-4, 1e-3)
#   EPOCHS (default 10000)

module load miniforge
source activate neuralrg

mkdir -p logs

L=32
T="${T:-2.269185314213022}"
N="200000"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
VP_LAMBDA="${VP_LAMBDA:?VP_LAMBDA env var required}"
EPOCHS="${EPOCHS:-10000}"
HCG_SHARED="${HCG_SHARED:-1}"
HCG_DILATED="${HCG_DILATED:-1}"
NREPEAT="${NREPEAT:-1}"
BATCH="${BATCH:-64}"
GRADACCUM="${GRADACCUM:-1}"

# Folder tag: encodes shared vs fixdil
if [ "$HCG_SHARED" = "1" ]; then
    VARIANT_TAG="shared"
elif [ "$HCG_DILATED" = "1" ]; then
    VARIANT_TAG="perscale_fixdil"
else
    VARIANT_TAG="perscale_nodilate"
fi

# Add nr tag only if nr != 1 (backward-compatible folder names)
if [ "$NREPEAT" = "1" ]; then
    NR_TAG=""
else
    NR_TAG="_nr${NREPEAT}"
fi
# Include T in folder name so parallel T-sweeps don't collide on parameters.hdf5.
# T_c uses legacy "T2.269" prefix for backward compat.
if [ "$T" = "2.269185314213022" ]; then
    T_TAG="T2.269"
else
    T_TAG="T${T}"
fi
FOLDER="./data/${L}Ising_${T_TAG}_hsBignet_hcg_${VARIANT_TAG}_vp${VP_LAMBDA}${NR_TAG}_b64"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet HCG + volume-preserving penalty"
echo "  variant=${VARIANT_TAG}  HCG_SHARED=$HCG_SHARED  HCG_DILATED=$HCG_DILATED"
echo "  VP_LAMBDA=$VP_LAMBDA  EPOCHS=$EPOCHS"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch "$BATCH" \
    -gradAccum "$GRADACCUM" \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat "$NREPEAT" \
    -lr 1e-3 \
    -gradClip 5.0 \
    -savePeriod 500 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared "$HCG_SHARED" \
    -hcgHidden 32 \
    -hcgDilated "$HCG_DILATED" \
    -hcgCircular 1 \
    -volumePreservingWeight "$VP_LAMBDA"

echo "Done."
