#!/bin/bash

#SBATCH --job-name=higgs_test
#SBATCH --partition=boost_usr_prod                  # The Booster partition name
#SBATCH --qos=normal                                # Quality of service
#SBATCH --time=12:00:00                             # Walltime
#SBATCH --output=higgs_test_print.log               # Output file
#SBATCH --error=higgs_test_err.log                  # Error file
#SBATCH --mail-user=giordano.vitale@edu.unige.it    # Mail address for notifications
#SBATCH --mail-type=BEGIN,END,FAIL                  # Events to notify via mail

# -- Node Configuration --
#SBATCH --nodes=1                           # Request 1 physical node
#SBATCH --mem=0                             # All memory (RAM) per node
#SBATCH --gres=gpu:4                        # All 4 A100 GPUs per node
#SBATCH --cpus-per-gpu=8                    # All 32 GPUs

# -- Environment --
source /leonardo/home/userexternal/gvitale0/miniconda3/etc/profile.d/conda.sh
conda activate falkon_original_env310

# -- Execution --
current_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "The experiment began on: $current_time"

python3 balkon_test_higgs.py

current_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "The experiment ended on: $current_time"