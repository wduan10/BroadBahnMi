#!/bin/bash

#SBATCH --job-name=CNN-1000
#SBATCH --output=/central/groups/CS156b/2024/BroadBahnMi/%j-1000.out
#SBATCH --error=/central/groups/CS156b/2024/BroadBahnMi/%j-1000.err
#SBATCH -A CS156b
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres gpu:2
#SBATCH --mail-user=wduan@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source /home/wduan/venvs/CS156b/bin/activate

cd /central/groups/CS156b/2024/BroadBahnMi/

python CNN_script.py hpc
