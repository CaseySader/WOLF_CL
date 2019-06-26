#!/bin/bash -l
#SBATCH -p gpu --gres="gpu:1" 
#SBATCH --job-name=output/logs/Algo
#SBATCH --mail-user=xxx@xx.xx.xx
#SBATCH --mail-type=FAIL
#SBATCH --workdir=/nfs/users/csader/Research/MastersProject/WOLF_CL
#SBATCH -e output/logs/Algo.err
#SBATCH -o output/logs/Algo.out
#SBATCH --array=1 

module load scikit-learn/0.18.0-Python-2.7.12
module load caffe/rc3-Python-2.7.12-CUDA-7.5.18-cuDNN-4.0 
module load TensorFlow/1.11-cp27-gpu 

python TFNeuralNetwork.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/TFNeuralNetwork_result1 -h 0.0 -l [10,10,10]
python TFNeuralNetwork.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/TFNeuralNetwork_result2 -h 0.0 -l [15,20,10]
python TFNeuralNetwork.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/TFNeuralNetwork_result3 -h 0.5 -l [10,10,10]
python TFNeuralNetwork.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/TFNeuralNetwork_result4 -h 0.5 -l [15,20,10]
