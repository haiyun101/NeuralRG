#!/bin/bash
#SBATCH --job-name=pg_L8_9800
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/pg_L8_9800_%j.out
#SBATCH --error=./logs/pg_L8_9800_%j.err

# L=8 path-gradient (STL) run, matched in epoch count to the standing
# reverse-KL baseline data/8Ising_T2.269_sym (9800 epochs, default arch).
# Replaces the 5000-ep long pair: instead of training a matched baseline
# we compare directly against the existing sym run.
#
# Matched flags vs sym: same arch (nlayers=10, nhidden=64), same -symmetry
# -skipHMC, same batch=128, same savePeriod=100. Only -pathGrad differs.

module load miniforge
source activate neuralrg

mkdir -p logs

T=2.269
L=8
EPOCHS=9800

FOLDER="./data/8Ising_T${T}_long9800_pathgrad"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=8 STL run, 9800 epochs (sym-matched)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 100 \
    -symmetry -skipHMC \
    -pathGrad

echo "Done."
