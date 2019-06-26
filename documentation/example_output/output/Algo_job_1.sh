#!/bin/bash -l
#SBATCH --job-name=output/logs/Algo
#SBATCH -N 8 -n 8 --mem=20g --time=72:00:00
#SBATCH --mail-user=xxx@xx.xx.xx
#SBATCH --mail-type=FAIL
#SBATCH --workdir=/nfs/users/csader/Research/MastersProject/WOLF_CL
#SBATCH -e output/logs/Algo.err
#SBATCH -o output/logs/Algo.out
#SBATCH --array=0-28 

module load scikit-learn/0.18.0-Python-2.7.12

python CSupportVectorClassification.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/CSupportVectorClassification_result1 -k rbf
python CSupportVectorClassification.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/CSupportVectorClassification_result2 -k linear
python LinearDiscriminantAnalysis.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/LinearDiscriminantAnalysis_result1 -s svd -v auto -n 1 -t 0.0001
python LinearDiscriminantAnalysis.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/LinearDiscriminantAnalysis_result2 -s lsqr -v auto -n 1 -t 0.0001
python LinearDiscriminantAnalysis.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/LinearDiscriminantAnalysis_result3 -s eigen -v auto -n 1 -t 0.0001
python LogisticRegression.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/LogisticRegression_result1 -p l1
python LogisticRegression.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/LogisticRegression_result2 -p l2
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result1 -d 5 -t 100
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result2 -d 5 -t 150
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result3 -d 6 -t 100
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result4 -d 6 -t 150
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result5 -d 7 -t 100
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result6 -d 7 -t 150
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result7 -d 8 -t 100
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result8 -d 8 -t 150
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result9 -d 9 -t 100
python RandomForest.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/RandomForest_result10 -d 9 -t 150
python GaussianNaiveBayes.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/GaussianNaiveBayes_result
python DecisionTree.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/DecisionTree_result1 -d 4 -p True
python DecisionTree.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/DecisionTree_result2 -d 4 -p False
python DecisionTree.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/DecisionTree_result3 -d 8 -p True
python DecisionTree.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/DecisionTree_result4 -d 8 -p False
python BernoulliNaiveBayes.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/BernoulliNaiveBayes_result1 -a 0
python BernoulliNaiveBayes.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/BernoulliNaiveBayes_result2 -a 1.0
python QuadraticDiscriminantAnalysis.py -i output/Splittingdata1/splitdatafiles.yaml -o  output/Splittingdata1/QuadraticDiscriminantAnalysis_result
