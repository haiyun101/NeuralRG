#!/bin/bash -l
#SBATCH --job-name=gen_L64_mcmc
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=./logs/gen_L64_mcmc_%j.out
#SBATCH --error=./logs/gen_L64_mcmc_%j.err

# Generate Wolff-cluster MCMC dataset at L=64 T=T_c for downstream
# forward-KL training. CPU-only (Numba JIT). Then convert discrete
# samples to the HS continuous-field representation.
# N=200000 matches the L=32 protocol; budget 4-6h on a fast node.

module load miniforge
source activate neuralrg

mkdir -p logs data/mcmc_data

T=2.269185314213022
L=64
N=200000
DISC_PT="data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N}.pt"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

echo "=========================================="
echo "L=$L Wolff MCMC + HS-field conversion (N=$N)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

if [ -f "$DISC_PT" ]; then
    echo "Discrete dataset already exists: $DISC_PT — skipping."
else
    echo "--- Phase 1: Wolff MCMC ---"
    python generate_mcmc_data.py -L $L -T $T -N $N
fi

if [ -f "$HS_PT" ]; then
    echo "HS dataset already exists: $HS_PT — skipping."
else
    echo "--- Phase 2: discrete -> HS continuous ---"
    python generate_hs_samples.py -L $L -T $T \
        --in_path $DISC_PT \
        --out_dir data/mcmc_data
fi

echo "Done."
