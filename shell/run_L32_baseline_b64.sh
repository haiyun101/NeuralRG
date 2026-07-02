#!/bin/bash -l
#SBATCH --job-name=L32_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L32_base_%j.out
#SBATCH --error=./logs/L32_base_%j.err

# L=32 hs_bignet baseline at batch=64 — control for the III.1 scaleLoss
# ablation, which had to drop batch from 128 -> 64 to fit the extra
# forward graph (see run_L32_iii1_single.sh comment). Without this
# matched-batch baseline, iii1 vs hs_bignet@batch=128 would conflate
# scaleLoss with batch-size effects.
#
# Same command line as hs_bignet (run_L32_hsBignet_bridge.sh) minus
# bridgeWeight / bridgeThresh and with -batch 64.

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-64}"
NREPEAT="${NREPEAT:-1}"
LR="${LR:-1e-3}"
GRADCLIP="${GRADCLIP:-0}"
SUFFIX=""
[ "$NREPEAT" != "1" ] && SUFFIX="${SUFFIX}_nr${NREPEAT}"
[ "$LR" != "1e-3" ] && SUFFIX="${SUFFIX}_lr${LR}"
[ "$GRADCLIP" != "0" ] && [ "$GRADCLIP" != "0.0" ] && SUFFIX="${SUFFIX}_gc${GRADCLIP}"
SUFFIX="${SUFFIX}_b${BATCH}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-$SUFFIX}"
FOLDER="./data/32Ising_T2.269_hsBignet_baseline${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 hs_bignet baseline (batch=$BATCH, no scaleLoss, no bridge)"
echo "  epochs=$EPOCHS  nrepeat=$NREPEAT  lr=$LR  gradClip=$GRADCLIP"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch "$BATCH" \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat "$NREPEAT" \
    -lr "$LR" \
    -gradClip "$GRADCLIP" \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
