#!/bin/bash -l
#SBATCH --job-name=d_L32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/d_L32_%j.out
#SBATCH --error=./logs/d_L32_%j.err

# D baseline (i2 single-CNN, nr=2) at L=32 across temperatures.
# Reproduces the L=32 D that scored F=1896.16 at T_c.

module load miniforge
source activate neuralrg
mkdir -p logs

L=32
T="${T:-2.269185314213022}"
N="200000"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
EPOCHS="${EPOCHS:-10000}"

if [ "$T" = "2.269185314213022" ]; then T_TAG="T2.269"; else T_TAG="T${T}"; fi
FOLDER="./data/${L}Ising_${T_TAG}_hsBignet_i2_stride8h32_nr2_b64"
mkdir -p "$FOLDER"

echo "L=$L D baseline (i2 nr=2)  T=$T  folder: $FOLDER"

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 -epochs "$EPOCHS" \
    -batch 64 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 2 \
    -lr 1e-3 -gradClip 5.0 -savePeriod 500 \
    -symmetry -skipHMC -dataDriven -dataPath "$HS_PT" -noDeq \
    -priorType conditional_gaussian \
    -condPriorSlowStride 8 -condPriorHidden 32

echo "Done."
