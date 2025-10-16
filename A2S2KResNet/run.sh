#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpu1
##
#SBATCH --job-name=hyperimg
#SBATCH -o ../log/SLURM.%N.%j.out
#SBATCH -e ../log/SLURM.%N.%j.err
##
#SBATCH --gres=gpu:rtx3090:2

hostname
date

module add ANACONDA/2021.05
module add CUDA/11.3.0

python3 A2S2KResNet.py -d KSC -e 20 -i 1 -p 5 -vs 0.97 -o adam

date