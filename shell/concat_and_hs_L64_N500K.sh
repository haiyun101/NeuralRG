#!/bin/bash -l
#SBATCH --job-name=concat_hs_L64
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/concat_hs_L64_%j.out
#SBATCH --error=./logs/concat_hs_L64_%j.err

# Concatenate 10 parallel chunks into N=500K, then run HS conversion.
# Designed to be submitted with --dependency=afterok:<array_job_id>.

module load miniforge
source activate neuralrg

mkdir -p logs data/mcmc_data

T=2.269185314213022
L=64
N=500000
DISC_PT="data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N}.pt"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

echo "=========================================="
echo "Concat 10×50K -> N=500K, then HS convert"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

if [ -f "$DISC_PT" ]; then
    echo "Concatenated discrete already exists: $DISC_PT — skipping concat."
else
    python -u concat_mcmc_chunks.py \
        --pattern "data/mcmc_data/chunks/mcmc_wolff_L${L}_T${T}_N50000_seed*.pt" \
        --out "$DISC_PT" \
        --expected-N $N
fi

if [ -f "$HS_PT" ]; then
    echo "HS already exists: $HS_PT — skipping conversion."
else
    echo "--- HS conversion ---"
    python -u generate_hs_samples.py -L $L -T $T \
        --in_path "$DISC_PT" \
        --out_dir data/mcmc_data
fi

echo "Done."
