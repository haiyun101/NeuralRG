#!/bin/bash -l
#SBATCH --job-name=L64_hcg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=./logs/L64_hcg_%j.out
#SBATCH --error=./logs/L64_hcg_%j.err

# L=64 hs_bignet + Hierarchical Conditional Gaussian prior (Path C of
# prior_offload_analysis_zh.md). 6-level hierarchy [32, 16, 8, 4, 2, 1].
#
# Same MERA arch as D64 bignet (nlayers=16, nhidden=128, nrepeat=1) so
# results compare fairly to Phase-2 D64 (~51 KL_fwd nat, 41% below
# baseline plateau). 36h walltime is safer than 24h for L=64.
#
# Tunables via env:
#   HCG_SHARED    1 (default, scale-shared) | 0 (per-level CNN)
#   HCG_HIDDEN    32
#   HCG_DILATED   1
#   HCG_CIRCULAR  1
#   BATCH=16  NREPEAT=1  LR=1e-3  GRADCLIP=0  EPOCHS=20000

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T="2.269185314213022"
N="${N:-200000}"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-16}"
NLAYERS="${NLAYERS:-16}"
NHIDDEN="${NHIDDEN:-128}"
NREPEAT="${NREPEAT:-1}"
LR="${LR:-1e-3}"
GRADCLIP="${GRADCLIP:-0}"
GRADACCUM="${GRADACCUM:-1}"
LOAD="${LOAD:-0}"            # 1 = resume from latest saving in $FOLDER (adds -load)
HCG_SHARED="${HCG_SHARED:-1}"
HCG_HIDDEN="${HCG_HIDDEN:-32}"
HCG_DILATED="${HCG_DILATED:-1}"
HCG_CIRCULAR="${HCG_CIRCULAR:-1}"
HCG_SHARED_DIL="${HCG_SHARED_DIL:-}"    # e.g. "1,2,4" for progressive shared-HCG dilation
HCG_INIT_SHARED="${HCG_INIT_SHARED:-}"  # path to shared-HCG folder for per-scale FULL init copy

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
[ -n "$HCG_SHARED_DIL" ] && SUFFIX="${SUFFIX}_progdil${HCG_SHARED_DIL//,/-}"
[ -n "$HCG_INIT_SHARED" ] && SUFFIX="${SUFFIX}_initshared"
[ "$NLAYERS" != "16" ] || [ "$NHIDDEN" != "128" ] && SUFFIX="${SUFFIX}_l${NLAYERS}h${NHIDDEN}"
[ "$NREPEAT" != "1" ] && SUFFIX="${SUFFIX}_nr${NREPEAT}"
[ "$LR" != "1e-3" ] && SUFFIX="${SUFFIX}_lr${LR}"
[ "$GRADCLIP" != "0" ] && [ "$GRADCLIP" != "0.0" ] && SUFFIX="${SUFFIX}_gc${GRADCLIP}"
[ "$GRADACCUM" != "1" ] && SUFFIX="${SUFFIX}_ga${GRADACCUM}"
SUFFIX="${SUFFIX}_b${BATCH}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-$SUFFIX}"
FOLDER="./data/${L}Ising_T2.269_hsBignet_hcg${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet + HCG prior"
echo "  hcgScaleShared=$HCG_SHARED  hcgHidden=$HCG_HIDDEN"
echo "  hcgDilated=$HCG_DILATED  hcgCircular=$HCG_CIRCULAR"
echo "  nlayers=$NLAYERS  nhidden=$NHIDDEN  nrepeat=$NREPEAT"
echo "  batch=$BATCH  epochs=$EPOCHS  lr=$LR  gradClip=$GRADCLIP  gradAccum=$GRADACCUM"
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
    -nlayers "$NLAYERS" -nmlp 3 -nhidden "$NHIDDEN" -nrepeat "$NREPEAT" \
    -lr "$LR" \
    -gradClip "$GRADCLIP" \
    -gradAccum "$GRADACCUM" \
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
    -hcgCircular "$HCG_CIRCULAR" \
    -hcgSharedDilations "$HCG_SHARED_DIL" \
    -hcgInitFromShared "$HCG_INIT_SHARED"

echo "Done."
