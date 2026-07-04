#!/bin/bash -l
#SBATCH --job-name=L32_hcg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L32_hcg_%j.out
#SBATCH --error=./logs/L32_hcg_%j.err

# L=32 hs_bignet + Hierarchical Conditional Gaussian prior (Path C of
# prior_offload_analysis_zh.md). Replaces single-CNN i2 with multi-scale
# hierarchy [16, 8, 4, 2, 1] of conditional Gaussians, each scored by a
# CNN. Scale-shared CNN (default) enforces scale-invariant conditional
# whitening as a direct architectural RG prior.
#
# Same MERA arch as D bignet (nlayers=16, nhidden=128, nrepeat=1) so
# results compare fairly to Phase-2 D32 (17.7 nat).
#
# Tunables via env:
#   HCG_SHARED    1 (default, scale-shared CNN) | 0 (per-level CNN)
#   HCG_HIDDEN    32
#   HCG_DILATED   1
#   HCG_CIRCULAR  1
#   BATCH=64  NREPEAT=1  LR=1e-3  GRADCLIP=0  EPOCHS=20000

module load miniforge
source activate neuralrg

mkdir -p logs

L=32
T="${T:-2.269185314213022}"        # override with env: T=2.4 for off-critical
N="${N:-200000}"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-64}"
LOAD="${LOAD:-0}"
NREPEAT="${NREPEAT:-1}"
NHIDDEN="${NHIDDEN:-128}"
LR="${LR:-1e-3}"
GRADCLIP="${GRADCLIP:-0}"
HCG_SHARED="${HCG_SHARED:-1}"
HCG_HIDDEN="${HCG_HIDDEN:-32}"
HCG_DILATED="${HCG_DILATED:-1}"
HCG_CIRCULAR="${HCG_CIRCULAR:-1}"

# Folder suffix encodes non-default knobs.
SUFFIX=""
if [ "$HCG_SHARED" = "1" ]; then
    SUFFIX="${SUFFIX}_shared"
else
    SUFFIX="${SUFFIX}_perscale"
fi
[ "$HCG_HIDDEN" != "32" ] && SUFFIX="${SUFFIX}_hcgh${HCG_HIDDEN}"
[ "$HCG_DILATED" != "1" ] && SUFFIX="${SUFFIX}_nodilate"
[ "$HCG_CIRCULAR" != "1" ] && SUFFIX="${SUFFIX}_zeropad"
[ "$NHIDDEN" != "128" ] && SUFFIX="${SUFFIX}_nh${NHIDDEN}"
[ "$NREPEAT" != "1" ] && SUFFIX="${SUFFIX}_nr${NREPEAT}"
[ "$LR" != "1e-3" ] && SUFFIX="${SUFFIX}_lr${LR}"
[ "$GRADCLIP" != "0" ] && [ "$GRADCLIP" != "0.0" ] && SUFFIX="${SUFFIX}_gc${GRADCLIP}"
SUFFIX="${SUFFIX}_b${BATCH}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-$SUFFIX}"
# Fold non-critical T into folder name (T_c uses legacy "T2.269" prefix)
if [ "$T" = "2.269185314213022" ]; then
    T_TAG="T2.269"
else
    T_TAG="T${T}"
fi
FOLDER="./data/${L}Ising_${T_TAG}_hsBignet_hcg${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet + HCG prior"
echo "  hcgScaleShared=$HCG_SHARED  hcgHidden=$HCG_HIDDEN"
echo "  hcgDilated=$HCG_DILATED  hcgCircular=$HCG_CIRCULAR"
echo "  nlayers=16  nhidden=$NHIDDEN  nrepeat=$NREPEAT"
echo "  batch=$BATCH  epochs=$EPOCHS  lr=$LR  gradClip=$GRADCLIP"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

LOAD_FLAG=""
if [ "$LOAD" = "1" ]; then
    LOAD_FLAG="-load"
    echo "  RESUMING from latest saving in $FOLDER (-load)"
fi

python -u main.py \
    $LOAD_FLAG \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch "$BATCH" \
    -nlayers 16 -nmlp 3 -nhidden "$NHIDDEN" -nrepeat "$NREPEAT" \
    -lr "$LR" \
    -gradClip "$GRADCLIP" \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared "$HCG_SHARED" \
    -hcgHidden "$HCG_HIDDEN" \
    -hcgDilated "$HCG_DILATED" \
    -hcgCircular "$HCG_CIRCULAR"

echo "Done."
