#!/bin/bash

#SBATCH -J Kura_ODE_100
#SBATCH -p gpu --gres=gpu:8
#SBATCH --mem=5G
#SBATCH -n 8
#SBATCH -t 10:00:00
#SBATCH -e log_error.out
#SBATCH -o log_output.out

module load anaconda/3-5.2.0
module load cuda/9.2.148
module load gcc/7.2
source activate osci2

# Run program
python3 run.py --ODE_100_1

