#!/bin/bash

#SBATCH --job-name=Test
#SBATCH --output=/central/groups/CS156b/2024/BroadBahnMi/%j.out
#SBATCH --error=/central/groups/CS156b/2024/BroadBahnMi/%j.err
#SBATCH -A CS156b
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres gpu:1
#SBATCH --mail-user=oxu@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source /home/oxu/my_venv/bin/activate

cd /central/groups/CS156b/2024/BroadBahnMi/

python pathology_script.py