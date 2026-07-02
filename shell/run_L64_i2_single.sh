#!/bin/bash -l
#SBATCH --job-name=L64_i2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=./logs/L64_i2_%j.out
#SBATCH --error=./logs/L64_i2_%j.err

# L=64 hs_bignet baseline + I.2 conditional Gaussian prior (scheme A).
# Same flow as run_L32_i2_single.sh.
#
# slow_stride default 16 -> 4x4 slow grid for L=64 (consistent ratio
# with L=32 stride=8 -> 4x4). batch=16 to match the matched-batch
# baseline.

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
SLOW_STRIDE="${SLOW_STRIDE:-16}"
COND_HIDDEN="${COND_HIDDEN:-32}"
NREPEAT="${NREPEAT:-1}"
NHIDDEN="${NHIDDEN:-128}"
LR="${LR:-1e-3}"
GRADCLIP="${GRADCLIP:-0}"
GRADACCUM="${GRADACCUM:-1}"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-16}"
LOAD="${LOAD:-0}"            # 1 = resume from latest checkpoint in $FOLDER (adds -load)
SUFFIX="_stride${SLOW_STRIDE}h${COND_HIDDEN}"
[ "$NHIDDEN" != "128" ] && SUFFIX="${SUFFIX}_nh${NHIDDEN}"
[ "$NREPEAT" != "1" ] && SUFFIX="${SUFFIX}_nr${NREPEAT}"
[ "$LR" != "1e-3" ] && SUFFIX="${SUFFIX}_lr${LR}"
[ "$GRADCLIP" != "0" ] && [ "$GRADCLIP" != "0.0" ] && SUFFIX="${SUFFIX}_gc${GRADCLIP}"
[ "$GRADACCUM" != "1" ] && SUFFIX="${SUFFIX}_ga${GRADACCUM}"
SUFFIX="${SUFFIX}_b${BATCH}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-$SUFFIX}"
FOLDER="./data/${L}Ising_T2.269_hsBignet_i2${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet + I.2 conditional Gaussian prior"
echo "  slow_stride=$SLOW_STRIDE  cnn_hidden=$COND_HIDDEN  nrepeat=$NREPEAT"
echo "  nhidden=$NHIDDEN  lr=$LR  gradClip=$GRADCLIP"
echo "  batch=$BATCH  epochs=$EPOCHS"
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
    -gradAccum "$GRADACCUM" \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -priorType conditional_gaussian \
    -condPriorSlowStride "$SLOW_STRIDE" \
    -condPriorHidden "$COND_HIDDEN"

echo "Done."
