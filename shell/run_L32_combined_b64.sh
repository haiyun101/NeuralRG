#!/bin/bash -l
#SBATCH --job-name=L32_comb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L32_comb_%j.out
#SBATCH --error=./logs/L32_comb_%j.err

# L=32 hs_bignet baseline + III.1 multi-scale loss + I.2 conditional
# Gaussian prior STACKED. Fills the (1,1) cell of the 2x2 ablation
# matrix laid out in concise_report_L32 / improvements_zh.md:
#
#                  | -scaleLoss          | +scaleLoss=1.0
#   ---------------+---------------------+---------------------
#   Gaussian prior | baseline_b64        | iii1_lam1.0_b64
#   Cond Gaussian  | i2_stride8h32       | this run
#
# Interaction effect = (this) - (iii1_lam1.0_b64) - (i2_stride8h32) + baseline.
# Sub-additive => the two interventions overlap; super-additive => synergy.
#
# Same flags as iii1+i2 individually (batch=64 because the scale-loss
# extra forward graph forced the same drop in iii1).

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
LAMBDA_SCALE="${LAMBDA_SCALE:-1.0}"
SLOW_STRIDE="${SLOW_STRIDE:-8}"
COND_HIDDEN="${COND_HIDDEN:-32}"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-64}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-_lam${LAMBDA_SCALE}_stride${SLOW_STRIDE}h${COND_HIDDEN}_b${BATCH}}"
FOLDER="./data/32Ising_T2.269_hsBignet_combined${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 hs_bignet + III.1 scaleLoss + I.2 conditional prior  (combined)"
echo "  lambda_scale=$LAMBDA_SCALE  slow_stride=$SLOW_STRIDE  cnn_hidden=$COND_HIDDEN"
echo "  batch=$BATCH  epochs=$EPOCHS"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch "$BATCH" \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -scaleLoss "$LAMBDA_SCALE" \
    -priorType conditional_gaussian \
    -condPriorSlowStride "$SLOW_STRIDE" \
    -condPriorHidden "$COND_HIDDEN"

echo "Done."
