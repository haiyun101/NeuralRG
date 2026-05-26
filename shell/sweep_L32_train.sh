#!/bin/bash
#SBATCH --job-name=nrg_L32_sweep
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/L32_sweep_%x_%j.out
#SBATCH --error=./logs/L32_sweep_%x_%j.err

# L=32 hs_dataDriven architecture sweep. One arm per invocation:
#   sbatch --job-name=L32sw_<arm> shell/sweep_L32_train.sh <arm>
#
# Arms (all forward-KL / MLE on the same HS continuous samples, -noDeq):
#   bignet      - deeper + wider net  (nlayers 16, nhidden 128)
#   haarPrior   - baseline net + Haar majority-vote prior (targets the
#                 magnetisation / slow mode that the baseline drops)
#   weightTying - baseline net + scale-invariant weight sharing (exact at T_c)
#
# Baseline for comparison = the existing data/32Ising_T2.269_hs_dataDriven
# run (nlayers 10, nhidden 64, -symmetry only). All arms converge well
# before EPOCHS; the diagnostic keys off the last checkpoint.

module load miniforge
source activate neuralrg

mkdir -p logs

ARM="$1"
T=2.269185314213022
L=32
EPOCHS=10000
HS_PT="data/mcmc_data/hs_L32_T${T}_N200000.pt"

case "$ARM" in
    bignet)
        FOLDER="./data/32Ising_T2.269_hs_bignet"
        ARCH="-nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 -symmetry"
        ;;
    haarPrior)
        FOLDER="./data/32Ising_T2.269_hs_haarPrior"
        ARCH="-nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 -symmetry -haarPrior"
        ;;
    weightTying)
        FOLDER="./data/32Ising_T2.269_hs_weightTying"
        ARCH="-nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 -symmetry -weightTying"
        ;;
    *)
        echo "ERROR: unknown arm '$ARM' (expected: bignet | haarPrior | weightTying)"
        exit 1
        ;;
esac

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data not found: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=32 hs_dataDriven sweep  |  arm: $ARM"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "arch:   $ARCH"
echo "epochs: $EPOCHS"
echo "=========================================="

python main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -savePeriod 500 \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    $ARCH

echo "Training done: arm=$ARM"
