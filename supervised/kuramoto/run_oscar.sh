#!/bin/bash

#SBATCH -J Kura_ODE_100_old
#SBATCH -p gpu --gres=gpu:8
#SBATCH --mem=20G
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -e log_error6.out
#SBATCH -o log_output6.out
#SBATCH --account=carney-tserre-condo

module load anaconda/3-5.2.0
module load gcc/7.2
module load cuda/10.0.130
source activate osci2

# Run program
python3 run.py --name ODE_100_old_MNIST

