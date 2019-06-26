__author__ = 'Kristopher Kaleb Goering, Pranav Bahl, Casey Sader'

import os
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
from array import array
from scipy.io.arff import loadarff
from sklearn.svm import LinearSVC
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
	importanceFile = importancefolder + "LinearSVClassificationFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def linearSupportVectorClassification(dataFile, outputFolder, c, lossType, penaltyType, isDual, tolerance, multiClass, isFit, interceptScaling, classWeight, isVerbose, maxIter, randomState,parameters):
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

		svm = LinearSVC(C=c, loss=lossType,  penalty=penaltyType, dual=isDual, tol=tolerance, multi_class=multiClass, fit_intercept=isFit, intercept_scaling=interceptScaling, class_weight=classWeight, verbose=isVerbose, max_iter=maxIter, random_state=randomState)
		svm.fit(train_features, train_labels)

		modelFile = modelsfolder + "LinearSVClassificationModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(svm, fd)
		modelset.append(modelFile)

		fd.close()
		
		importanceFile = calculateFeatureImportance(train_features,svm,importancefolder,i)
		importanceset.append(importanceFile)

		test_predictions['predictions'] = svm.predict(test_features)

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
	resultDict['algorithm'] = "LinearSupportVectorClassification"
	yaml.dump(resultDict, open(outputFolder + '/results.yaml', 'w'))

def main(args):
	inputFile = ''
	outputFolder = ''
	parameters=dict()
	c = 1.0 #float;
	lossType = 'squared_hinge' #string; default = 'squared_hinge', 'hinge' is default for SVC class
	penaltyType = 'l2' #string; default = 'l2', 'l1'
	isDual = True #boolean
	tolerance = 0.001 #float;
	multiClass = 'ovr' #string; default = 'ovr', 'crammer_singer'
	isFit = True #boolean;
	interceptScaling = 1 #float;
	classWeight = None # {dict, 'balanced'}
	isVerbose = 0 #int;
	maxIter = 1000 #int; -1 for no limit
	randomState = None #int, RandomState instance or None

	try:
		opts,args = getopt.getopt(args, "i:o:c:l:p:d:t:m:f:s:w:v:x:r:", [])
	except getopt.GetoptError:
		print 'LinearSupportVectorClassification.py -i <inputFile> -o <outputFolder> -c <c> -l <lossType> -p <penaltyType> -d <isDual> -t <tolerance> -m <multiClass> -f <isFit> -s <interceptScaling> -w <classWeight> -v <isVerbose> - <maxIter> -r <randomState>'
		sys.exit(2)
	for opt,arg in opts:
		if opt == '-i':
			inputFile = arg
		elif opt == '-o':
			outputFolder = arg
		elif opt == '-c':
			c = float(arg)
			parameters['parameter.c']=arg
		elif opt == '-l':
			lossType = arg
			parameters['parameter.l']=arg
		elif opt == '-p':
			penaltyType = arg
			parameters['parameter.p']=arg
		elif opt == '-d':
			if arg == "True":
				isDual = True
			elif arg == "False":
				isDual = False
			else:
				raise ValueError
			parameters['parameter.d']=arg
		elif opt == '-t':
			tolerance = float(arg)
			parameters['parameter.t']=arg
		elif opt == '-m':
			multiClass = arg
			parameters['parameter.m']=arg
		elif opt == '-f':
			if arg == "True":
				isFit = True
			elif arg == "False":
				isFit = False
			else:
				raise ValueError
			parameters['parameter.f']=arg
		elif opt == '-s':
			interceptScaling = float(arg)
			parameters['parameter.s']=arg
		elif opt == '-w':
			classWeight = arg
			parameters['parameter.w']=arg
		elif opt == '-v':
			isVerbose = int(arg)
			parameters['parameter.v']=arg
		elif opt == '-x':
			maxIter = int(arg)
			parameters['parameter.x']=arg
		elif opt == '-r':
			if arg != None:
				randomState = int(arg)
			parameters['parameter.r']=arg
	linearSupportVectorClassification(inputFile, outputFolder, c, lossType, penaltyType, isDual, tolerance, multiClass, isFit, interceptScaling, classWeight, isVerbose, maxIter, randomState,parameters)
if __name__ == "__main__":
   main(sys.argv[1:])
