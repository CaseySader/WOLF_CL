#!/bin/bash -l
#SBATCH --job-name=predict
#SBATCH -N 1 -n 1 --mem=20g --time=72:00:00
#SBATCH --mail-user=xxx@xx.xx.xx
#SBATCH --mail-type=FAIL
#SBATCH --workdir=/nfs/users/csader/Research/MastersProject/WOLF_CL
#SBATCH -e /nfs/users/csader/Research/MastersProject/WOLF_CL/predictions/predict.err
#SBATCH -o /nfs/users/csader/Research/MastersProject/WOLF_CL/predictions/predict.out
#SBATCH --array=0-0 

module load scikit-learn/0.18.0-Python-2.7.12
module load TensorFlow/1.11-cp27-gpu 

python MakePrediction.py -i diabetes.arff -o predictions.csv -m example_output/Splittingdata1/LinearDiscriminantAnalysis_result3/models/LinearDiscriminantAnalysisModel3.pkl -l class -s True