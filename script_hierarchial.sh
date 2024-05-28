#!/bin/bash

#SBATCH --job-name=Hierarchical
#SBATCH --output=/home/mbanik/BroadBahnMi/%j.out
#SBATCH --error=/home/mbanik/BroadBahnMi/%j.err
#SBATCH -A CS156b
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres gpu:4
#SBATCH --mail-user=mbanik@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source /home/wduan/venvs/CS156b/bin/activate

cd /central/home/mbanik/BroadBahnMi/

python hierarchy_clustering.py
