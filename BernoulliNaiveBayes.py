__author__ = 'Lecheng Zheng, Pranav Bahl, Casey Sader'

import os
import errno
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
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
	importanceFile = importancefolder + "BernoulliNaiveBayesFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def BernoulliNaiveBayes(dataFile,outputfolder,Alpha,parameters):
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

		bnb = BernoulliNB(alpha=Alpha)
		bnb.fit(train_features, train_labels)

		modelFile = modelsfolder + "BernoulliNaiveBayesModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(bnb, fd)
		modelset.append(modelFile)

		fd.close()
		
		importanceFile = calculateFeatureImportance(train_features,bnb,importancefolder,i)
		importanceset.append(importanceFile)

		test_predictions['predictions'] = bnb.predict(test_features)
		resultFile = outputfolder+'/result'+str(i+1)+'.csv'

		test_predictions.to_csv(resultFile,index=False)
		resultset.append(resultFile)
	resultDict = dict()
	resultDict['results'] = resultset
	resultDict['models'] = modelset
	resultDict['featureimportance'] = importanceset
	resultDict['label'] = label
	if not parameters:
		parameters['parameter'] = "default"
	resultDict['algo_params'] = parameters
	resultDict['inputFile'] = inputFile
	resultDict['algorithm'] = "BernoulliNaiveBayes"
	resultDict['split_params'] = inputData['split_params']
	if 'feature_selection_parameters' in inputData:
		resultDict['feature_selection_parameters'] = inputData['feature_selection_parameters']
		resultDict['feature_selection_algorithm'] = inputData['feature_selection_algorithm']
	if 'feature_extraction_parameters' in inputData:
		resultDict['feature_extraction_parameters'] = inputData['feature_extraction_parameters']
		resultDict['feature_extraction_algorithm'] = inputData['feature_extraction_algorithm']
	if 'preprocessing_params' in inputData:
		resultDict['preprocessing_params'] = inputData['preprocessing_params']
	yaml.dump(resultDict,open(outputfolder+'/results.yaml','w'))


def main(args):
	dataFile=''
	outputfolder=''
	parameters=dict()
	alpha = 1.0
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
			alpha=float(arg)
			parameters['parameter.a']=arg
	BernoulliNaiveBayes(dataFile,outputfolder,alpha,parameters)

if __name__ == "__main__":
	main(sys.argv[1:])
