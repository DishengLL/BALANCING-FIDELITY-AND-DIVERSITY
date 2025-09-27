#!/bin/bash
#SBATCH --job-name=Index_selection
#SBATCH -A aiscii
#SBATCH -p aiscii
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=80:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -w gput[060-075],aisciit[01-05],aisct[01-04]
# Load required modules
. ~/.bashrc

# Job info
# print user info
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
conda activate diffusers_env

nvidia-smi
echo "Running job: $SLURM_JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"

PYTHON=~/.conda/envs/tiny/bin/python

$PYTHON ./index_selection.py
echo "Job completed: $SLURM_JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"