#!/bin/bash 
#SBATCH --account=def-skelly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=10G
#SBATCH --time=3-10:00  # time (DD-HH:MM)
#SBATCH --error=error_file.txt

seed=$1

module load python/3.10

python models/generator_TPG.PY -s $seed  --num_proc 64
