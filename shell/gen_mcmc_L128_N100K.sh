#!/bin/bash -l
#SBATCH --job-name=gen_L128_N100K
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/gen_L128_N100K_%j.out
#SBATCH --error=./logs/gen_L128_N100K_%j.err

# Generate N=100K Wolff MCMC + HS at L=128 T_c.
# L=128 lattice is 4× L=64 area, so each Wolff step is ~4× cost.
# Starting with N=100K (5× fewer samples than L=64's 500K) to fit walltime.
# Can extend later if training needs more data.

module load miniforge
source activate neuralrg

mkdir -p logs data/mcmc_data

T=2.269185314213022
L=128
N=100000
DISC_PT="data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N}.pt"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

echo "=========================================="
echo "L=$L Wolff MCMC + HS conversion (N=$N)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

if [ -f "$DISC_PT" ]; then
    echo "Discrete dataset already exists: $DISC_PT — skipping."
else
    echo "--- Phase 1: Wolff MCMC (N=$N at L=$L) ---"
    python -u generate_mcmc_data.py -L $L -T $T -N $N
fi

if [ -f "$HS_PT" ]; then
    echo "HS dataset already exists: $HS_PT — skipping."
else
    echo "--- Phase 2: discrete -> HS continuous ---"
    python -u generate_hs_samples.py -L $L -T $T \
        --in_path $DISC_PT \
        --out_dir data/mcmc_data
fi

echo
echo "Done."
date
