#!/bin/bash

#SBATCH --job-name=ResNet
#SBATCH --output=/central/groups/CS156b/2024/BroadBahnMi/%j.out
#SBATCH --error=/central/groups/CS156b/2024/BroadBahnMi/%j.err
#SBATCH -A CS156b
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres gpu:4
#SBATCH --mail-user=wduan@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source /home/wduan/venvs/CS156b/bin/activate

cd /central/groups/CS156b/2024/BroadBahnMi/Wilson/ResNet

# python ResNet_final.py hpc 'No Finding'
python ResNet_final.py hpc 'Enlarged Cardiomediastinum'
# python ResNet_final.py hpc 'Cardiomegaly'
# python ResNet_final.py hpc 'Lung Opacity'
# python ResNet_final.py hpc 'Pneumonia'
# python ResNet_final.py hpc 'Pleural Effusion'
# python ResNet_final.py hpc 'Pleural Other'
# python ResNet_final.py hpc 'Fracture'
# python ResNet_final.py hpc 'Support Devices'