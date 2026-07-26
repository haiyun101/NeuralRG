#!/bin/bash -l
#SBATCH --job-name=L128_baseline_fresh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/L128_baseline_fresh_%j.out
#SBATCH --error=./logs/L128_baseline_fresh_%j.err

# L=128 GAUSSIAN-PRIOR baseline (no HCG, no CNN). Same arch as champion
# family (nlayers=16, nhidden=128) but priorType=gaussian.
# Serves as: (a) pure MERA baseline at L=128; (b) target for pure-MERA
# transfer from L=64 baseline (see companion script _from_L64.sh).

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

python -u main.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_baseline_freshInit \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L128_T2.269185314213022_N100000.pt \
    -epochs 15000 -batch 8 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry -alpha 0 \
    -priorType gaussian \
    -savePeriod 200 -cuda 0

date
