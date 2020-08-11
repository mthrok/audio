#!/bin/bash

#SBATCH --job-name=source_separation

#SBATCH --output=/checkpoint/%u/jobs/%x/%j.out

#SBATCH --error=/checkpoint/%u/jobs/%x/%j.err

#SBATCH --partition=dev

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=8

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=16G

#SBATCH --gpus-per-task=1

#SBATCH --time=24:00:00

#srun env
srun tools/slurm/train.sh $@
