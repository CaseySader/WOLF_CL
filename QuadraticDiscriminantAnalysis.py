__author__ = 'Kristopher Kaleb Goering, Casey Sader'

import os
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
from array import array
from scipy.io.arff import loadarff
from sklearn.qda import QDA
import errno
import csv
import pickle

def calculateFeatureImportance(train_features,model,importancefolder,i):
	# sort these values and write to separate files
	feature_names = train_features.columns.tolist()

	# QDA doesn't have feature importance or coef_ attribute
	feature_importance = ['N/A'] * len(feature_names)

	feature_rows = zip(feature_names, feature_importance)
	importanceFile = importancefolder + "QuadraticDiscriminantAnalysisFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def quadraticDiscriminantAnalysis(dataFile, outputFolder, regParam,parameters):
	inputData = yaml.load(open(dataFile))
	trainingSet = inputData['training']
	testingSet = inputData['testing']
	inputFile = inputData['inputFile']
	label = inputData['label']
	resultSet = []
	modelset = []
	importanceset = []
	
	if not os.path.exists(outputFolder):
		try:
			os.makedirs(outputFolder)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise exc
			pass

	modelsfolder = outputFolder + "/models/"
	if not os.path.exists(modelsfolder):
		try:
			os.makedirs(modelsfolder)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise exc
			pass

	importancefolder = outputFolder + "/FeatureImportance/"
	if not os.path.exists(importancefolder):
		try:
			os.makedirs(importancefolder)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise exc
			pass


	for i in range(len(trainingSet)):
		train_df = pd.read_csv(trainingSet[i])
		train_labels = train_df[label]
		train_features = train_df.drop(label,axis=1)
		test_df = pd.read_csv(testingSet[i])
		test_predictions = pd.DataFrame(test_df[label])
		test_features = test_df.drop(label,axis=1)

		qda = QDA(reg_param=regParam)
		qda.fit(train_features, train_labels)

		modelFile = modelsfolder + "QuadraticDiscriminantAnalysisModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(qda, fd)
		modelset.append(modelFile)

		fd.close()

		importanceFile = calculateFeatureImportance(train_features,qda,importancefolder,i)
		importanceset.append(importanceFile)

		test_predictions['predictions'] = qda.predict(test_features)

		resultFile = outputFolder + '/result' + str(i + 1) + '.csv'

		test_predictions.to_csv(resultFile,index=False)
		resultSet.append(resultFile)

	resultDict = dict()

	resultDict['results'] = resultSet
	resultDict['models'] = modelset
	resultDict['featureimportance'] = importanceset
	resultDict['label'] = label

	if not parameters:
		parameters['parameter']='default'
	resultDict['algo_params'] = parameters
	resultDict['split_params'] = inputData['split_params']
	if 'feature_selection_parameters' in inputData:
		resultDict['feature_selection_parameters'] = inputData['feature_selection_parameters']
		resultDict['feature_selection_algorithm'] = inputData['feature_selection_algorithm']
	if 'feature_extraction_parameters' in inputData:
		resultDict['feature_extraction_parameters'] = inputData['feature_extraction_parameters']
		resultDict['feature_extraction_algorithm'] = inputData['feature_extraction_algorithm']
	if 'preprocessing_params' in inputData:
		resultDict['preprocessing_params'] = inputData['preprocessing_params']
	resultDict['inputFile'] = inputFile
	resultDict['algorithm'] = "QuadraticDiscriminantAnalysis"
	yaml.dump(resultDict, open(outputFolder + '/results.yaml', 'w'))

def main(args):
	inputFile = ''
	outputFolder = ''
	parameters=dict()
	regParam = 0.0 #float; regularizes the covariance estimate as [(1-reg_param)*Sigma + reg_param*np.eye(n_features)]
	try:
		opts,args = getopt.getopt(args, "i:o:p:", [])
	except getopt.GetoptError:
		print 'QuadraticDiscriminantAnalysis.py -i <inputFile> -o <outputFolder> -p <regParam>'
		sys.exit(2)
	for opt,arg in opts:
		if opt == '-i':
			inputFile = arg
		elif opt == '-o':
			outputFolder = arg
		elif opt == '-p':
			regParam = float(arg)
			parameters['parameter.p']=arg
	quadraticDiscriminantAnalysis(inputFile, outputFolder, regParam,parameters)
if __name__ == "__main__":
   main(sys.argv[1:])

###############################################
#Other parameters
#priors : array, optional, shape (n_classes,)
#Class priors.
