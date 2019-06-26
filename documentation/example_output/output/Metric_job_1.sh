#!/bin/bash -l
#SBATCH --job-name=output/logs/Metric
#SBATCH -N 8 -n 8 --mem=20g --time=72:00:00
#SBATCH --mail-user=xxx@xx.xx.xx
#SBATCH --mail-type=FAIL
#SBATCH --workdir=/nfs/users/csader/Research/MastersProject/WOLF_CL
#SBATCH -e output/logs/Metric.err
#SBATCH -o output/logs/Metric.out
#SBATCH --array=0-0 

module load scikit-learn/0.18.0-Python-2.7.12

python MetricCollection.py -i output/Splittingdata1/CSupportVectorClassification_result1/results.yaml -o output/Splittingdata1/CSupportVectorClassification_result1
python MetricCollection.py -i output/Splittingdata1/CSupportVectorClassification_result2/results.yaml -o output/Splittingdata1/CSupportVectorClassification_result2
python MetricCollection.py -i output/Splittingdata1/LinearDiscriminantAnalysis_result1/results.yaml -o output/Splittingdata1/LinearDiscriminantAnalysis_result1
python MetricCollection.py -i output/Splittingdata1/LinearDiscriminantAnalysis_result2/results.yaml -o output/Splittingdata1/LinearDiscriminantAnalysis_result2
python MetricCollection.py -i output/Splittingdata1/LinearDiscriminantAnalysis_result3/results.yaml -o output/Splittingdata1/LinearDiscriminantAnalysis_result3
python MetricCollection.py -i output/Splittingdata1/LogisticRegression_result1/results.yaml -o output/Splittingdata1/LogisticRegression_result1
python MetricCollection.py -i output/Splittingdata1/LogisticRegression_result2/results.yaml -o output/Splittingdata1/LogisticRegression_result2
python MetricCollection.py -i output/Splittingdata1/RandomForest_result1/results.yaml -o output/Splittingdata1/RandomForest_result1
python MetricCollection.py -i output/Splittingdata1/RandomForest_result2/results.yaml -o output/Splittingdata1/RandomForest_result2
python MetricCollection.py -i output/Splittingdata1/RandomForest_result3/results.yaml -o output/Splittingdata1/RandomForest_result3
python MetricCollection.py -i output/Splittingdata1/RandomForest_result4/results.yaml -o output/Splittingdata1/RandomForest_result4
python MetricCollection.py -i output/Splittingdata1/RandomForest_result5/results.yaml -o output/Splittingdata1/RandomForest_result5
python MetricCollection.py -i output/Splittingdata1/RandomForest_result6/results.yaml -o output/Splittingdata1/RandomForest_result6
python MetricCollection.py -i output/Splittingdata1/RandomForest_result7/results.yaml -o output/Splittingdata1/RandomForest_result7
python MetricCollection.py -i output/Splittingdata1/RandomForest_result8/results.yaml -o output/Splittingdata1/RandomForest_result8
python MetricCollection.py -i output/Splittingdata1/RandomForest_result9/results.yaml -o output/Splittingdata1/RandomForest_result9
python MetricCollection.py -i output/Splittingdata1/RandomForest_result10/results.yaml -o output/Splittingdata1/RandomForest_result10
python MetricCollection.py -i output/Splittingdata1/GaussianNaiveBayes_result/results.yaml -o output/Splittingdata1/GaussianNaiveBayes_result
python MetricCollection.py -i output/Splittingdata1/DecisionTree_result1/results.yaml -o output/Splittingdata1/DecisionTree_result1
python MetricCollection.py -i output/Splittingdata1/DecisionTree_result2/results.yaml -o output/Splittingdata1/DecisionTree_result2
python MetricCollection.py -i output/Splittingdata1/DecisionTree_result3/results.yaml -o output/Splittingdata1/DecisionTree_result3
python MetricCollection.py -i output/Splittingdata1/DecisionTree_result4/results.yaml -o output/Splittingdata1/DecisionTree_result4
python MetricCollection.py -i output/Splittingdata1/BernoulliNaiveBayes_result1/results.yaml -o output/Splittingdata1/BernoulliNaiveBayes_result1
python MetricCollection.py -i output/Splittingdata1/BernoulliNaiveBayes_result2/results.yaml -o output/Splittingdata1/BernoulliNaiveBayes_result2
python MetricCollection.py -i output/Splittingdata1/QuadraticDiscriminantAnalysis_result/results.yaml -o output/Splittingdata1/QuadraticDiscriminantAnalysis_result
python MetricCollection.py -i output/Splittingdata1/TFNeuralNetwork_result1/results.yaml -o output/Splittingdata1/TFNeuralNetwork_result1
python MetricCollection.py -i output/Splittingdata1/TFNeuralNetwork_result2/results.yaml -o output/Splittingdata1/TFNeuralNetwork_result2
python MetricCollection.py -i output/Splittingdata1/TFNeuralNetwork_result3/results.yaml -o output/Splittingdata1/TFNeuralNetwork_result3
python MetricCollection.py -i output/Splittingdata1/TFNeuralNetwork_result4/results.yaml -o output/Splittingdata1/TFNeuralNetwork_result4
