#!/bin/bash -l
#SBATCH --job-name=output/logs/SplitData
#SBATCH -N 1 -n 1 --mem=20g --time=72:00:00
#SBATCH --mail-user=xxx@xx.xx.xx
#SBATCH --mail-type=FAIL
#SBATCH --workdir=/nfs/users/csader/Research/MastersProject/WOLF_CL
#SBATCH -e output/logs/SplitData.err
#SBATCH -o output/logs/SplitData.out
#SBATCH --array=0-0 

module load scikit-learn/0.18.0-Python-2.7.12

python Splittingdata.py -i diabetes.arff -o output/Splittingdata1 -d diabetes_2495.arff -k 5 -r 1 -l class
