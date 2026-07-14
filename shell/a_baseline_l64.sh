#!/bin/bash -l
#SBATCH --job-name=a_L64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/a_L64_%j.out
#SBATCH --error=./logs/a_L64_%j.err

# A baseline (isotropic Gaussian prior, no CNN) at L=64 across temperatures.
# Third of three L=64 T-sweeps (fixdil+VP champion / D=i2 nr=2 / A=Gaussian nr=2).
# Reproduces the L=64 A nr=2 that scored F=7579.28 at T_c
# (from data/64Ising_T2.269_hsBignet_baseline_nr2_b16).

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T="${T:-2.269185314213022}"
N="200000"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
EPOCHS="${EPOCHS:-15000}"

if [ "$T" = "2.269185314213022" ]; then
    T_TAG="T2.269"
else
    T_TAG="T${T}"
fi
FOLDER="./data/${L}Ising_${T_TAG}_hsBignet_baseline_nr2_b16"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L A baseline (Gaussian nr=2)  T=$T"
echo "  folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch 16 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 2 \
    -lr 1e-3 \
    -gradClip 5.0 \
    -savePeriod 500 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -priorType gaussian

echo "Done."
