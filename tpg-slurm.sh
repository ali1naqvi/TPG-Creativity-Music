#!/bin/bash 
#SBATCH --account=def-skelly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=512M
#SBATCH --time=0-36:00  # time (DD-HH:MM)
#SBATCH --error=error_file.txt

seed=$1

module load python/3.10

python /models/generator_TPG.PY -s $seed  --num_proc 32