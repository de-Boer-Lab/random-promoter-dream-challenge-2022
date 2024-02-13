#!/bin/bash
#SBATCH --array=0-49
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --job-name=unlockPos
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0-48:0:0
#SBATCH --account=rrg-cdeboer
#SBATCH --output=/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/DeepSTARR_torch/outputs/%N-%j.out
#SBATCH --error=/home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/DeepSTARR_torch/errors/%N-%j.err

module load cuda cudnn
source /home/rafi11/projects/rrg-cdeboer/rafi11/rafi_torch/bin/activate
python /home/rafi11/projects/rrg-cdeboer/rafi11/DREAMNet/DeepSTARR_torch/train_unlockdna_pos.py $SLURM_ARRAY_TASK_ID
