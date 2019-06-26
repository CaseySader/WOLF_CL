#!/bin/bash -l
#SBATCH --job-name=output/logs/Result
#SBATCH -N 1 -n 1 --mem=20g --time=72:00:00
#SBATCH --mail-user=xxx@xx.xx.xx
#SBATCH --mail-type=FAIL
#SBATCH --workdir=/nfs/users/csader/Research/MastersProject/WOLF_CL
#SBATCH -e output/logs/Result.err
#SBATCH -o output/logs/Result.out
#SBATCH --array=0-0 

module load scikit-learn/0.18.0-Python-2.7.12

python DatabaseResults.py -i diabetes_2495.arff