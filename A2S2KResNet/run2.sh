#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu6
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
##
#SBATCH --job-name=a2s2k
#SBATCH -o ../log/SLURM.%N.%j.out
#SBATCH -e ../log/SLURM.%N.%j.err
##

hostname
date

free -h
module add cuda/12.2.1
python3 A2S2KResNet.py -d BS -e 100 -i 2 -p 4 -vs 0.97 -o adam

date
