#!/bin/bash
#SBATCH --job-name=nrg_L32_train
#SBATCH --time=2-00:00:00           # Adjust time as needed (D-HH:MM:SS)
#SBATCH --partition=gpu             # Specify your cluster's GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           # CPU cores for data loading
#SBATCH --mem=32G                   # Memory (adjust based on dataset size)
#SBATCH --output=./data/32Ising_T2.269_sym_dataDriven_skipHMC/train_L32_%j.out
#SBATCH --error=l./data/32Ising_T2.269_sym_dataDriven_skipHMC/train_L32_%j.err

# 1. Load modules and activate environment
# (Uncomment and adjust these based on your HPC environment setup)
# module load anaconda/2023.09
# conda activate neuralrg_env

# Ensure the logs directory exists
mkdir -p logs

echo "=========================================="
echo "Starting NeuralRG Data-Driven Training"
echo "Lattice: 32 | Temp: 2.3"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# 2. Run the training script
# The -dataDriven flag will automatically search for mcmc_wolff_L32_T2.2680_N*.pt
python main.py \
    -L 32 \
    -T 2.269 \
    -folder "./data/32Ising_T2.269_sym_dataDriven_skipHMC_epoch500000" \
    -dataDriven \
    -cuda 0 \
    -epochs 500000 \
    -batch 128 \
    -nlayers 10 \
    -nmlp 3 \
    -nhidden 10 \
    -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC

echo "=========================================="
echo "Training completed."