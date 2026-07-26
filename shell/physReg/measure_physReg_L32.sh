#!/bin/bash -l
#SBATCH --job-name=phys_measure
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/phys_measure_%j.out
#SBATCH --error=./logs/phys_measure_%j.err

# Measure |M|, χ, U4 from L=32 physReg sweep cells vs plain champion.
# Answers: does physReg actually improve the physics observables it targets?
# L=32 GT: |M|=0.6544, χ=31.61, U4=0.6110

module load miniforge
source activate neuralrg
mkdir -p logs

CHAMPION=data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64
PHYS_C1=data/32Ising_T2.269_physReg_chi0.01_u40.01
PHYS_C2=data/32Ising_T2.269_physReg_chi0.1_u40.1
PHYS_C3=data/32Ising_T2.269_physReg_chi1.0_u41.0

python -u analyzers/physReg/measure_physReg_effect.py \
    --N 2000 --batch 64 --device cpu \
    --cells \
        "plain_champion:$CHAMPION:9500" \
        "physReg_λ0.01:$PHYS_C1:800" \
        "physReg_λ0.1:$PHYS_C2:800" \
        "physReg_λ1.0:$PHYS_C3:800"

echo
echo "L=32 GT reference:  |M|=0.6544  χ=31.61  U4=0.6110"
date
