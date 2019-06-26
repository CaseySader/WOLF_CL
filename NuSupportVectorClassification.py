__author__ = 'Kristopher Kaleb Goering, Pranav Bahl, Casey Sader'

import os
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
from array import array
from scipy.io.arff import loadarff
from sklearn.svm import NuSVC
import errno
import csv
import pickle

def calculateFeatureImportance(train_features,model,importancefolder,i,kernelType):
	feature_names = train_features.columns.tolist()
	if kernelType == "linear":
		# calculate feature importance using coef
		coefs = model.coef_
		
		pos_coefs = map(abs, coefs[0])
		
		# normalize the values between 0 and 1
		feature_importance = [float(coef_index)/sum(pos_coefs) for coef_index in pos_coefs]
		
		# sort these values and write to separate files
		feature_names_sorted = [x for _,x in sorted(zip(feature_importance,feature_names), reverse=True)]
		feature_importance_sorted = sorted(feature_importance, reverse=True)

		feature_rows = zip(feature_names_sorted, feature_importance_sorted)
	else:
		# nonlinear kernel doesn't have feature importance attribute and doesn't work with eli5 due to a lack of coef_
		print "Can only calculate feature importances using coef if linear kernel"
		feature_importance = ['N/A'] * len(feature_names)

		feature_rows = zip(feature_names, feature_importance)

	importanceFile = importancefolder + "NuSVClassificationFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def nuSupportVectorClassification(dataFile, outputFolder, NU, kernelType, numDegree, numGamma, coef, isShrink, isProb, tolerance, cacheSize, classWeight, isVerbose, maxIter, decisionShape, randomState,parameters):
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

		svm = NuSVC(nu=NU, kernel=kernelType, degree=numDegree, gamma=numGamma, coef0=coef, shrinking=isShrink, probability=isProb, tol=tolerance, cache_size=cacheSize, class_weight=classWeight, verbose=isVerbose, max_iter=maxIter, decision_function_shape=decisionShape, random_state=randomState)
		svm.fit(train_features, train_labels)

		modelFile = modelsfolder + "NuSVClassificationModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(svm, fd)
		modelset.append(modelFile)

		fd.close()

		importanceFile = calculateFeatureImportance(train_features,svm,importancefolder,i,kernelType)
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
	resultDict['algorithm'] = "NuSupportVectorClassification"
	yaml.dump(resultDict, open(outputFolder + '/results.yaml', 'w'))

def main(args):
	inputFile = ''
	outputFolder = ''
	parameters=dict()
	NU = 0.5 #float;
	kernelType = 'rbf' #string; default = 'rbf', 'linear', 'poly', 'sigmoid', 'precomputed', callable; if callable, data must be [n,n]
	numDegree = 3 #int; only for poly, ignored by others
	numGamma = 'auto' #float; for rbf, poly, sigmoid; if auto, 1/n_features
	coef = 0.0 #float; only significant in poly or sigmoid
	isShrink = True #boolean;
	isProb = False #boolean;
	tolerance = 0.001 #float;
	cacheSize = 200 #float; in MB
	classWeight = None # {dict, 'balanced'}
	isVerbose = False #boolean
	maxIter = -1 #int; -1 for no limit
	decisionShape = None #string; 'ovo', 'ovr', or None
	randomState = None #int, RandomState instance or None

	try:
		opts,args = getopt.getopt(args, "i:o:n:k:d:g:f:s:p:t:a:w:v:m:e:r:", [])
	except getopt.GetoptError:
		print 'NuSupportVectorClassification.py -i <inputFile> -o <outputFolder> -n <NU> -k <kernelType> -d <numDegree> -g <numGamma> -f <coef> -s <isShrink> -p <isProb> -t <tolerance> -a <cacheSize> -w <classWeight> -v <isVerbose> -m <maxIter> -e <decisionFunctionShape> -r <randomState>'
		sys.exit(2)
	for opt,arg in opts:
		if opt == '-i':
			inputFile = arg
		elif opt == '-o':
			outputFolder = arg
		elif opt == '-n':
			NU = float(arg)
			parameters['parameter.n']=arg
		elif opt == '-k':
			kernelType = arg
			parameters['parameter.k']=arg
		elif opt == '-d':
			numDegree = int(arg)
			parameters['parameter.d']=arg
		elif opt == '-g':
			try:
				numGamma = float(arg)
			except:
				numGamma = arg
			parameters['parameter.g']=arg
		elif opt == '-f':
			coef = float(arg)
			parameters['parameter.f']=arg
		elif opt == '-s':
			if arg == "True":
				isShrink = True
			else:
				isShrink = False
			parameters['parameter.s']=arg
		elif opt == '-p':
			if arg == "True":
				isProb = True
			elif arg == "False":
				isProb = False
			else:
				raise ValueError
			parameters['parameter.p']=arg
		elif opt == '-t':
			tolerance = float(arg)
			parameters['parameter.t']=arg
		elif opt == '-a':
			cacheSize = float(arg)
			parameters['parameter.a']=arg
		elif opt == '-w':
			classWeight = arg
			parameters['parameter.w']=arg
		elif opt == '-v':
			if arg == "True":
				isVerbose = True
			elif arg == "False":
				isVerbose = False
			else:
				raise ValueError
			parameters['parameter.v']=arg
		elif opt == '-m':
			maxIter = int(arg)
			parameters['parameter.m']=arg
		elif opt == '-e':
			decisionShape = arg
			parameters['parameter.e']=arg
		elif opt == '-r':
			randomState = int(arg)
			parameters['parameter.r']=arg
	nuSupportVectorClassification(inputFile, outputFolder, NU, kernelType, numDegree, numGamma, coef, isShrink, isProb, tolerance, cacheSize, classWeight, isVerbose, maxIter, decisionShape, randomState,parameters)
if __name__ == "__main__":
   main(sys.argv[1:])
