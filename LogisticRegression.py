__author__ = 'Lecheng Zheng, Pranav Bahl, Casey Sader'

import os
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
from array import array
from scipy.io.arff import loadarff
from sklearn.linear_model import LogisticRegression
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
	importanceFile = importancefolder + "LogisticRegressionFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def LogesticRegression(dataFile,outputfolder,C,Penalty,parameters):
	inputData = yaml.load(open(dataFile))
	trainingSet = inputData['training']
	testingSet = inputData['testing']
	inputFile = inputData['inputFile']
	label = inputData['label']
	resultset = []
	modelset = []
	importanceset = []

	if not os.path.exists(outputfolder):
		try:
			os.makedirs(outputfolder)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise exc
			pass

	modelsfolder = outputfolder + "/models/"
	if not os.path.exists(modelsfolder):
		try:
			os.makedirs(modelsfolder)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise exc
			pass

	importancefolder = outputfolder + "/FeatureImportance/"
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

		lr = LogisticRegression(penalty=Penalty,C=C)
		lr.fit(train_features,train_labels)

		modelFile = modelsfolder + "LogisticRegressionModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(lr, fd)
		modelset.append(modelFile)

		fd.close()

		importanceFile = calculateFeatureImportance(train_features,lr,importancefolder,i)
		importanceset.append(importanceFile)

		test_predictions['predictions'] = lr.predict(test_features)

		resultFile = outputfolder+'/result'+str(i+1)+'.csv'

		test_predictions.to_csv(resultFile,index=False)
		resultset.append(resultFile)

	resultDict = dict()

	resultDict['results'] = resultset
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
	resultDict['algorithm'] = "LogisticRegression"
	yaml.dump(resultDict,open(outputfolder+'/results.yaml','w'))


def main(args):
	dataFile=''
	outputfolder=''
	parameters=dict()
	C=1.0
	penalty='l2'
	try:
		opts,args=getopt.getopt(args,"i:o:c:p:",[])
	except getopt.GetoptError:
		print 'datasplit.py -i <inputfile> -o <outputfolder> -c <regularization> -p <penalty>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			dataFile=arg
		elif opt=='-o':
			outputfolder=arg
		elif opt=='-c':
			C=float(arg)
			parameters['parameter.c']=arg
		elif opt=='-p':
			penalty=arg
			parameters['parameter.p']=arg
	LogesticRegression(dataFile,outputfolder,C,penalty,parameters)

if __name__ == "__main__":
   main(sys.argv[1:])
