#!/bin/bash -l
#SBATCH --job-name=gen_L128_N5K
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/gen_L128_N5K_%j.out
#SBATCH --error=./logs/gen_L128_N5K_%j.err

# Fast N=5K L=128 HS data generation to unblock L=128 training earlier.
# Runs in parallel with the N=100K gen (job 1796152). Training jobs use
# whichever dataset file exists at their launch time; main.py's data
# loader auto-selects the largest N.
#
# Expected wall: ~1.5h (Wolff at L=128 ~ 1 sec/sample).
# Chinchilla-style ratio for N=5K vs L=128 (16K dims): ~0.3 — undertrained
# regime, but useful for early convergence signal. Users can restart with
# N=100K when 1796152 completes.

module load miniforge
source activate neuralrg
mkdir -p logs data/mcmc_data

T=2.269185314213022
L=128
N=5000
DISC_PT="data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N}.pt"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

echo "=========================================="
echo "L=$L Wolff MCMC + HS conversion (N=$N, FAST)"
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
