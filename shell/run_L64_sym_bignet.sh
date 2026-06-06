#!/bin/bash -l
#SBATCH --job-name=L64_sym_bignet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_sym_%j.out
#SBATCH --error=./logs/L64_sym_%j.err

# L=64 T_c reverse-KL with the standard score-function (REINFORCE-style)
# estimator, bignet. The non-STL companion to L64_pathgrad_bignet, so we
# can repeat the L=32 STL-vs-sym comparison at one more L.
# No training data needed (samples from the flow).

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
FOLDER="./data/${L}Ising_T2.269_sym_bignet"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L sym_bignet rev-KL (score-function, b=64, A100)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 20000 \
    -batch 16 \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC

echo "Done."
