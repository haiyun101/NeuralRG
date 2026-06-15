#!/bin/bash -l
#SBATCH --job-name=L32_i2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L32_i2_%j.out
#SBATCH --error=./logs/L32_i2_%j.err

# L=32 hs_bignet baseline + I.2 conditional Gaussian prior (Scheme A).
# Scheme I.2 from analyzers/rg_fixed_point/improvements_zh.md:
#   P(z) = P(z_slow) * P(z_fast | z_slow)
# z_slow lives on the slow-grid sub-lattice (stride condPriorSlowStride),
# z_fast lives everywhere else with mean/log-std produced by a small CNN
# conditioned on z_slow. Zero-init at the final conv reproduces the
# isotropic N(0,I) baseline exactly, so any divergence is attributable
# to learned conditional structure.
#
# Tunables via env:
#   SLOW_STRIDE=8        (slow-grid stride; L=32 stride 8 -> 4x4 slow)
#   COND_HIDDEN=32       (CNN hidden channels)
#   EPOCHS=20000
#   BATCH=128            (lower to 64 if scaleLoss/combined adds forward graph)
#   FOLDER_SUFFIX=       (defaults to _stride${SLOW_STRIDE}h${COND_HIDDEN}[_b${BATCH}-if-not-128])
#
# This is a SINGLE-VARIABLE run: same as hs_bignet except for the
# -priorType conditional_gaussian flag, so the diff vs baseline isolates
# the conditional-prior contribution to V3/V4/V5 metrics.

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
SLOW_STRIDE="${SLOW_STRIDE:-8}"
COND_HIDDEN="${COND_HIDDEN:-32}"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-128}"
if [ "$BATCH" = "128" ]; then
    FOLDER_SUFFIX="${FOLDER_SUFFIX:-_stride${SLOW_STRIDE}h${COND_HIDDEN}}"
else
    FOLDER_SUFFIX="${FOLDER_SUFFIX:-_stride${SLOW_STRIDE}h${COND_HIDDEN}_b${BATCH}}"
fi
FOLDER="./data/32Ising_T2.269_hsBignet_i2${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 hs_bignet + I.2 conditional Gaussian prior"
echo "  slow_stride=$SLOW_STRIDE  cnn_hidden=$COND_HIDDEN  epochs=$EPOCHS"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python main.py \
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
    -priorType conditional_gaussian \
    -condPriorSlowStride "$SLOW_STRIDE" \
    -condPriorHidden "$COND_HIDDEN"

echo "Done."
