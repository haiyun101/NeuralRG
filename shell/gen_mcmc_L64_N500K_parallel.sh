#!/bin/bash -l
#SBATCH --job-name=gen_L64_par
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --array=0-9
#SBATCH --output=./logs/gen_L64_par_%A_%a.out
#SBATCH --error=./logs/gen_L64_par_%A_%a.err

# Parallel-chain Wolff MCMC: 10 chunks × N=50K = N=500K total at L=64 T_c.
# Each array task uses a different RNG seed; chains are i.i.d. after the
# per-chain thermalisation (1000 Wolff steps). Concatenation + HS field
# conversion happens in a separate dependent job (concat_and_hs_L64_N500K.sh).
#
# Each task writes: data/mcmc_data/chunks/mcmc_wolff_L64_T<T>_N50000_seed{S}.pt

module load miniforge
source activate neuralrg

mkdir -p logs data/mcmc_data/chunks

T=2.269185314213022
L=64
N_PER_CHUNK=50000
SEED=$((1000 + SLURM_ARRAY_TASK_ID))
OUT="data/mcmc_data/chunks/mcmc_wolff_L${L}_T${T}_N${N_PER_CHUNK}_seed${SEED}.pt"

echo "=========================================="
echo "Parallel Wolff chunk ${SLURM_ARRAY_TASK_ID}/9"
echo "  L=$L T=$T N=$N_PER_CHUNK seed=$SEED"
echo "  Job ${SLURM_JOB_ID} on $SLURMD_NODENAME"
echo "  out: $OUT"
echo "=========================================="

if [ -f "$OUT" ]; then
    echo "Chunk already exists, skipping."
else
    python -u generate_mcmc_data.py -L $L -T $T -N $N_PER_CHUNK \
        -seed $SEED -outPath "$OUT"
fi

echo "Done."
