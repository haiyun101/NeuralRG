#!/bin/bash
#SBATCH --job-name=nrg_flow_diag
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/flow_diag_%j.out
#SBATCH --error=./logs/flow_diag_%j.err

module load miniforge
source activate neuralrg

# All hs_dataDriven, sym, nsym runs at T=2.269 for L=8 and L=16
FOLDERS=(
    data/8Ising_T2.269_hs_dataDriven
    data/8Ising_T2.269_sym
    data/8Ising_T2.269_nsym
    data/16Ising_T2.269_hs_dataDriven
    data/16Ising_T2.269_sym
)

python analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 8000 -b 512
