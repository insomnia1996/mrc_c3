#!/bin/bash
#SBATCH --job-name=c3
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node 1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
ulimit -s unlimited
ulimit -l unlimited
module load \
    cuda/10.2.89-gcc-8.3.0 \
    intel-mkl \
    gcc/8.3.0-gcc-4.8.5
source activate torch
bash run_mrc_c3.sh
