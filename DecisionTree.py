__author__ = 'Lecheng Zheng,Pranav Bahl,Casey Sader'

import os
import arff
import yaml
import csv
import sys, getopt
import numpy as np
import pandas as pd
import pickle
from scipy.io.arff import loadarff
from sklearn import tree
import errno

def calculateFeatureImportance(train_features,model,importancefolder,i):
	# sort these values and write to separate files
	feature_names = train_features.columns.tolist()
	feature_importance = model.feature_importances_

	feature_names_sorted = [x for _,x in sorted(zip(feature_importance,feature_names), reverse=True)]
	feature_importance_sorted = sorted(feature_importance, reverse=True)

	feature_rows = zip(feature_names_sorted, feature_importance_sorted)
	importanceFile = importancefolder + "DecisionTreeFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def DecissionTree(dataFile,outputfolder,criterion,max_features,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_leaf_nodes,presort,parameters):
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

		dt = tree.DecisionTreeClassifier(criterion=criterion,max_features=max_features,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_leaf_nodes=max_leaf_nodes,presort=presort,class_weight='balanced')
		dt.fit(train_features,train_labels)

		# Save the model to a file
		modelFile = modelsfolder + "DecisionTreeModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(dt, fd)
		modelset.append(modelFile)

		fd.close()

		importanceFile = calculateFeatureImportance(train_features,dt,importancefolder,i)
		importanceset.append(importanceFile)

		test_predictions['predictions'] = dt.predict(test_features)
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
	resultDict['algorithm'] = "DecisionTree"
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
	criterion = 'gini'
	max_features = 'auto'
	max_depth = None
	min_samples_split=2
	min_samples_leaf=1
	min_weight_fraction_leaf=0.
	max_leaf_nodes=None
	presort=False
	try:
		opts,args=getopt.getopt(args,"i:o:c:f:d:s:l:w:n:p:",[])
	except getopt.GetoptError:
		print 'datasplit.py -i <inputfile> -o <outputfolder> -c <criterion> -f <max_features> -d <max_depth> -s <min_samples_split> -l <min_samples_leaf> -w <min_weight_fraction_leaf> -n <max_leaf_nodes> -p <presort>'
		print 'option for criterion: gini or entropy'
		print 'option for max_features: auto, sqrt, log2'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			dataFile=arg
		elif opt=='-o':
			outputfolder=arg
		elif opt=='-c':
			criterion=arg
			parameters['parameter.c']=arg
		elif opt=='-f':
			max_features=arg
			parameters['parameter.f']=arg
		elif opt=='-d':
			if arg != None:
				max_depth=int(arg)
			parameters['parameter.d']=arg
		elif opt=='-s':
			try:
				min_samples_split=int(arg)
			except:
				min_samples_split=float(arg)
			parameters['parameter.s']=arg
		elif opt=='-l':
			try:
				min_samples_leaf=int(arg)
			except:
				min_samples_leaf=float(arg)
			parameters['parameter.l']=arg
		elif opt=='-w':
			min_weight_fraction_leaf=float(arg)
			parameters['parameter.w']=arg
		elif opt=='-n':
			if arg != None:
				max_leaf_nodes=int(arg)
			parameters['parameter.n']=arg
		elif opt=='-p':
			if arg == 'True':
				presort = True
			elif arg == 'False':
				presort = False
			else:
				raise ValueError
			parameters['parameter.p']=arg
	DecissionTree(dataFile,outputfolder,criterion,max_features,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_leaf_nodes,presort,parameters)
if __name__ == "__main__":
   main(sys.argv[1:])
