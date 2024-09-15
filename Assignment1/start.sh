#!/bin/sh
#SBATCH --job-name=encoding # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=12:00:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gpus-per-node=1
srun python3 run.py --mode=$1
