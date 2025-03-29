#!/bin/bash
#SBATCH --job-name=train_nn
#SBATCH --open-mode=append
#SBATCH --output=slurm/%j_%a_%x.out
#SBATCH --error=slurm/%j_%a_%x.err
#SBATCH --export=ALL
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:L40S:1
#SBATCH --mail-type=END
#SBATCH --mail-user=rshar@cs.cmu.edu


source /home/rshar/.bashrc
conda init
conda activate code-uncert
cd /home/rshar/school/ml-in-prac

python3 nn_similarity.py

