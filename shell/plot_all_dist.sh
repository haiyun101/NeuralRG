#!/bin/bash

# Loop through all subdirectories in the data folder
for dir in data/*/ 
do
    # Remove trailing slash and get just the folder name for the job name
    folder=${dir%/}
    folder_name=$(basename "$folder")
    
    echo "Submitting plot job for: $folder_name"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=plot_${folder_name}
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=${folder}/plot_output_%j.log

module load miniforge
source activate neuralrg

# Run the plotting script
# We add "|| true" so that if one folder fails, it doesn't affect others
python analyzers/plot_dist_hdf5.py "$folder" || true
EOT
done