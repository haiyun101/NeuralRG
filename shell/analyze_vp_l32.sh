#!/bin/bash
#SBATCH --job-name=analyze_vp_L32
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/analyze_vp_L32_%j.out
#SBATCH --error=./logs/analyze_vp_L32_%j.err

# Diagnostic on L=32 VP-sweep runs. Env var GROUP selects the arm to
# analyze so we can parallelize.
#   GROUP=shared    → shared VP variants (4 folders)
#   GROUP=fixdil    → fixdil VP nr=1 (4 folders)

module load miniforge
source activate neuralrg

mkdir -p logs

GROUP="${GROUP:-fixdil}"

if [ "$GROUP" = "shared" ]; then
    FOLDERS=(
        data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-5_b64
        data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-4_b64
        data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-3_b64
        data/32Ising_T2.269_hsBignet_hcg_shared_vp1e-2_b64
    )
else
    FOLDERS=(
        data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-5_b64
        data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_b64
        data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64
        data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_b64
    )
fi

echo "GROUP=$GROUP"
python -u analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 8000 -b 512

echo "Done."
