__author__ = 'Lecheng Zheng, Pranav Bahl, Casey Sader'

import os
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
from array import array
from scipy.io.arff import loadarff
from sklearn.naive_bayes import GaussianNB
import errno
import csv
import pickle

def calculateFeatureImportance(train_features,model,importancefolder,i):
	# sort these values and write to separate files
	feature_names = train_features.columns.tolist()

	# GNB doesn't have feature importance or coef_ attribute
	feature_importance = ['N/A'] * len(feature_names)

	feature_rows = zip(feature_names, feature_importance)
	importanceFile = importancefolder + "GaussianNaiveBayesFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def GaussianNaiveBayes(dataFile,outputfolder,Alpha,parameters):
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

		clf = GaussianNB()
		clf.fit(train_features,train_labels)

		modelFile = modelsfolder + "GaussianNaiveBayesModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(clf, fd)
		modelset.append(modelFile)

		fd.close()
		
		importanceFile = calculateFeatureImportance(train_features,clf,importancefolder,i)
		importanceset.append(importanceFile)

		test_predictions['predictions'] = clf.predict(test_features)

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
	resultDict['inputFile'] = inputFile
	resultDict['algorithm'] = "GaussianNaiveBayes"
	resultDict['split_params'] = inputData['split_params']
	if 'feature_selection_parameters' in inputData:
		resultDict['feature_selection_parameters'] = inputData['feature_selection_parameters']
		resultDict['feature_selection_algorithm'] = inputData['feature_selection_algorithm']
	if 'feature_extraction_parameters' in inputData:
		resultDict['feature_extraction_parameters'] = inputData['feature_extraction_parameters']
		resultDict['feature_extraction_algorithm'] = inputData['feature_extraction_algorithm']
	yaml.dump(resultDict,open(outputfolder+'/results.yaml','w'))


def main(args):
	dataFile=''
	outputfolder=''
	parameters=dict()
	alpha=1.0
	try:
		opts,args=getopt.getopt(args,"i:o:a:",[])
	except getopt.GetoptError:
		print 'datasplit.py -i <inputfile> -o <outputfolder> -a <alpha>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			dataFile=arg
		elif opt=='-o':
			outputfolder=arg
		elif opt=='-a':
			alpha==float(arg)
			parameters['parameter.a']=arg
	GaussianNaiveBayes(dataFile,outputfolder,alpha,parameters)
if __name__ == "__main__":
	main(sys.argv[1:])
