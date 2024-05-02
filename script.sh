#!/bin/bash

#SBATCH --job-name=Test
#SBATCH --output=/home/oxu/BroadBahnMi/out/%j.out
#SBATCH --error=/home/oxu/BroadBahnMi/out/%j.err
#SBATCH -A CS156b
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres gpu:4
#SBATCH --mail-user=oxu@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source /home/oxu/my_venv/bin/activate

cd /home/oxu/BroadBahnMi

python pathology_script.py