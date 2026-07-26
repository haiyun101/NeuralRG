#!/bin/bash -l
#SBATCH --job-name=L128_revKL_from_L64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L128_revKL_from_L64_%j.out
#SBATCH --error=./logs/L128_revKL_from_L64_%j.err

# L=128 REVERSE-KL warm-start from L=64 champion.
#
# Rationale: reverse KL doesn't need HS data — samples from the flow itself
# and evaluates against Ising Hamiltonian. Can start immediately, no data
# dependency. Complements the forward-KL job (waits on L=128 HS data gen).
#
# Warm-start: same as forward-KL job — stride-aligned CNN transfer +
# scale-index MERA transfer from L=64 champion. The L=128-only extra
# blocks (coarsest MERA scale + stride-64 HCG CNN) stay at fresh init.
#
# Risks:
#  - Reverse KL is mode-seeking → risk of Z2 mode collapse (memory
#    project_rg_probe_objective_fingerprint documents L=32 sym_bignet
#    collapsing to degenerate identity)
#  - Symmetrized wrapper (-symmetry) partially mitigates via q(x)=q(-x)
#  - If the L=64 forward-KL warm-start's bimodal structure gets destroyed
#    → this run may still fail even with symmetrized
#
# We run this AS A HEDGE alongside the forward-KL job. Whichever converges
# to better physics (Best-200 or χ, U₄) wins as the L=128 champion.
#
# Note: NO -dataDriven flag → reverse-KL mode. Uses Ising energy directly.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_champion_revKL_from_L64 \
    -skipHMC \
    -epochs 15000 -batch 8 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -loadFromSmallerL data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -savePeriod 200 -cuda 0

date
