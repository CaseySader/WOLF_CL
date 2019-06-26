__author__ = 'Kristopher Kaleb Goering,Pranav_Bahl,Casey Sader'

import os
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
from array import array
from scipy.io.arff import loadarff
from sklearn.lda import LDA
import errno
import csv
import pickle

def calculateFeatureImportance(train_features,model,importancefolder,i):
	# calculate feature importance using coef
	coefs = model.coef_
	
	pos_coefs = map(abs, coefs[0])
	
	# normalize the values between 0 and 1
	feature_importance = [float(coef_index)/sum(pos_coefs) for coef_index in pos_coefs]
	
	# sort these values and write to separate files
	feature_names = train_features.columns.tolist()

	feature_names_sorted = [x for _,x in sorted(zip(feature_importance,feature_names), reverse=True)]
	feature_importance_sorted = sorted(feature_importance, reverse=True)

	feature_rows = zip(feature_names_sorted, feature_importance_sorted)
	importanceFile = importancefolder + "LinearDiscriminantAnalysisFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def linearDiscriminantAnalysis(dataFile, outputFolder, solverType, shrinkageValue, numOfComponents, tolerance,parameters):
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

		lda = LDA(solver=solverType, shrinkage=shrinkageValue, n_components=numOfComponents, tol=tolerance)
		lda.fit(train_features, train_labels)

		modelFile = modelsfolder + "LinearDiscriminantAnalysisModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(lda, fd)
		modelset.append(modelFile)

		fd.close()

		importanceFile = calculateFeatureImportance(train_features,lda,importancefolder,i)
		importanceset.append(importanceFile)

		test_predictions['predictions'] = lda.predict(test_features)

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
	resultDict['algorithm'] = "LinearDiscriminantAnalysis"
	yaml.dump(resultDict, open(outputFolder + '/results.yaml', 'w'))

def main(args):
	inputFile = ''
	outputFolder = ''
	parameters=dict()
	solverType = 'svd' #string; default = 'svd', 'lsqr', 'eigen'
	shrinkageValue = None #string or float; 'auto', 0-1, or None; only works with lsqr or eigen solvers
	numOfComponents = None #int; n < n_classes - 1
	tolerance = 0.0001 #float; threshold for rank estim in SVD solver
	try:
		opts,args = getopt.getopt(args, "i:o:s:v:n:t:", [])
	except getopt.GetoptError:
		print 'LinearDiscriminantAnalysis.py -i <inputFile> -o <outputFolder> -s <solverType> -v <shrinkageValue> -n <numOfComponents> -t <tolerance>'
		sys.exit(2)
	for opt,arg in opts:
		if opt == '-i':
			inputFile = arg
		elif opt == '-o':
			outputFolder = arg
		elif opt == '-s':
			solverType = arg
			parameters['parameter.s']=arg
		elif opt == '-v':
			try:
				shrinkageValue = float(arg)
			except:
				shrinkageValue = arg
			parameters['parameter.v']=arg
		elif opt == '-n':
			numOfComponents = int(arg)
			parameters['parameter.n']=arg
		elif opt == '-t':
			tolerance = float(arg)
			parameters['parameter.t']=arg
	if solverType == 'svd':
		shrinkageValue = None
	else:
		tolerance = None
	linearDiscriminantAnalysis(inputFile, outputFolder, solverType, shrinkageValue, numOfComponents, tolerance,parameters)
if __name__ == "__main__":
   main(sys.argv[1:])

###############################################
#Other parameters
#priors : array, optional, shape (n_classes,)
#Class priors.
