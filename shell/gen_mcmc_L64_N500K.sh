#!/bin/bash -l
#SBATCH --job-name=gen_L64_N500K
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=./logs/gen_L64_N500K_%j.out
#SBATCH --error=./logs/gen_L64_N500K_%j.err

# Generate N=500K Wolff MCMC dataset at L=64 T=T_c, then HS-field convert.
# Built to support megabignet (and future NSF) resume training on the
# expanded dataset — addresses Phase-2 verdict limit ① + ②: bigger models
# need more data than bignet's 200K (Chinchilla).
#
# Storage: ~7.7G discrete + ~7.7G HS ≈ 15.4G (405G free, OK).
# Time: Wolff at L=64 is single-thread Numba; budget 16h walltime safety.

module load miniforge
source activate neuralrg

mkdir -p logs data/mcmc_data

T=2.269185314213022
L=64
N=500000
DISC_PT="data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N}.pt"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

echo "=========================================="
echo "L=$L Wolff MCMC + HS conversion (N=$N)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

if [ -f "$DISC_PT" ]; then
    echo "Discrete dataset already exists: $DISC_PT — skipping."
else
    echo "--- Phase 1: Wolff MCMC (N=$N) ---"
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

echo "Done."
