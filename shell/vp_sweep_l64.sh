#!/bin/bash -l
#SBATCH --job-name=vp_sweep64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/vp_sweep64_%j.out
#SBATCH --error=./logs/vp_sweep64_%j.err

# Volume-preserving penalty sweep on L=64 HCG. Mirrors the L=32 sweep
# after finding that fixdil + VP λ=1e-3 beat D at L=32.
#
# Env vars:
#   VP_LAMBDA  (required)
#   NREPEAT    (1 or 2, default 1)
#   HCG_SHARED (0 = fixdil per-scale, 1 = shared; default 0)
#   EPOCHS     (default 15000 — larger network takes longer to converge)

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T="${T:-2.269185314213022}"
N="200000"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
VP_LAMBDA="${VP_LAMBDA:?VP_LAMBDA env var required}"
EPOCHS="${EPOCHS:-15000}"
NREPEAT="${NREPEAT:-1}"
HCG_SHARED="${HCG_SHARED:-0}"
HCG_DILATED="${HCG_DILATED:-1}"
BATCH="${BATCH:-16}"
GRADACCUM="${GRADACCUM:-1}"

if [ "$HCG_SHARED" = "1" ]; then
    VARIANT_TAG="shared"
elif [ "$HCG_DILATED" = "1" ]; then
    VARIANT_TAG="perscale_fixdil"
else
    VARIANT_TAG="perscale_nodilate"
fi

# Include T in folder name so parallel T-sweeps don't collide.
if [ "$T" = "2.269185314213022" ]; then
    T_TAG="T2.269"
else
    T_TAG="T${T}"
fi
FOLDER="./data/${L}Ising_${T_TAG}_hsBignet_hcg_${VARIANT_TAG}_vp${VP_LAMBDA}_nr${NREPEAT}_b16"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet HCG + volume-preserving penalty"
echo "  variant=${VARIANT_TAG}  NREPEAT=$NREPEAT  VP_LAMBDA=$VP_LAMBDA  EPOCHS=$EPOCHS"
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
