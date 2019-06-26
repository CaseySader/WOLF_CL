__author__ = 'Sohaib Kiani, Casey Sader'

import os
import arff
import yaml
import sys, getopt
import numpy as np
import pandas as pd
import errno
import csv
import pickle
from scipy.io.arff import loadarff
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def calculateFeatureImportance(train_features,model,importancefolder,i):
	# sort these values and write to separate files
	feature_names = train_features.columns.tolist()

	try:
		# works for decision tree and random forest
		feature_importance = model.feature_importances_
	except:
		try:
			# calculate feature importance using coef
			coefs = model.coef_
			pos_coefs = map(abs, coefs[0])
			
			# normalize the values between 0 and 1
			feature_importance = [float(coef_index)/sum(pos_coefs) for coef_index in pos_coefs]
		except:
			# doesn't have coef either
			print "Problem with base estimator for .feature_importances_ and .coef_\nMight be that base estimator doesn't have these attributes"
			feature_importance = ['N/A'] * len(feature_names)

	feature_names_sorted = [x for _,x in sorted(zip(feature_importance,feature_names), reverse=True)]
	feature_importance_sorted = sorted(feature_importance, reverse=True)

	feature_rows = zip(feature_names_sorted, feature_importance_sorted)
	importanceFile = importancefolder + "AdaBoostFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile

def adaBoostClassifier(dataFile,outputfolder,base_estimator_name, n_estimators, learning_rate, algorithm,parameters):
	inputData = yaml.load(open(dataFile))	
	trainingSet = inputData['training']
	testingSet = inputData['testing']
	inputFile = inputData['inputFile']
	label = inputData['label']
	resultset = []
	modelset = []
	importanceset = []
	
	names = ["NearestNeighbors", "LinearSVM", "RBFSVM", "GaussianProcess",
         "DecisionTree", "RandomForest", "NeuralNet", 
         "NaiveBayes", "QuadraticDiscriminantAnalysis"]

	classifiers = [
	    	KNeighborsClassifier(3),
	    SVC(kernel="linear", C=0.025),
	    SVC(gamma=2, C=1),
	    GaussianProcessClassifier(1.0 * RBF(1.0)),
	    DecisionTreeClassifier(max_depth=5),
	    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	    MLPClassifier(alpha=1),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis()]

	try:
		if (base_estimator_name is None) or (base_estimator_name.lower() == "none"):
			base_estimator = None
		else:
			idx = names.index(base_estimator_name)
			base_estimator = classifiers[idx]
	except:
		print "The base_estimator: ", base_estimator_name, " is not supported"
		raise
		sys.exit(2)

	if not os.path.isdir(outputfolder):
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
		testpredictions = pd.DataFrame(test_df[label])
		test_features = test_df.drop(label,axis=1)
				

		abc = AdaBoostClassifier(base_estimator = base_estimator, n_estimators = n_estimators , learning_rate = learning_rate, algorithm = algorithm)
		
		abc.fit(train_features,train_labels)

		modelFile = modelsfolder + "AdaBoostModel" + str(i+1) + ".pkl"
		with open(modelFile, 'wb') as fd:
			pickle.dump(abc, fd)
		modelset.append(modelFile)

		fd.close()

		importanceFile = calculateFeatureImportance(train_features,abc,importancefolder,i)
		importanceset.append(importanceFile)

		testpredictions['predictions'] = abc.predict(test_features)
		resultFile = outputfolder+'/result'+str(i+1)+'.csv'

		testpredictions.to_csv(resultFile,index=False)
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
	resultDict['algorithm'] = "AdaBoostClassifier"
	yaml.dump(resultDict,open(outputfolder+'/results.yaml','w'))


def main(args):
	datafile=''
	outputfolder=''
	parameters=dict()
	base_estimator=None
	n_estimators=50
	learning_rate=1.0
	algorithm='SAMME'
	
	
	try:
		opts,args=getopt.getopt(args,"i:o:e:n:r:a:",[])
	except getopt.GetoptError:
		print 'datasplit.py -i <inputfile> -o <outputfolder> -e <base_estimator> -n<number_of_estimators> -r<learning_rate> -a<algorithm>'
		sys.exit(2)
	
	for opt,arg in opts:
		if opt=='-i':
			dataFile=arg
		elif opt=='-o':
			outputfolder=arg
		elif opt=='-e':
			base_estimator=arg
			parameters['parameter.e']=arg
		elif opt=='-n':
			n_estimators=int(arg)
			parameters['parameter.n']=arg
		elif opt == "-r":
			learning_rate=float(arg)
			parameters['parameter.r']=arg
		elif opt=="-a":
			algorithm = arg
			parameters['parameter.a']=arg
	
	adaBoostClassifier(dataFile,outputfolder,base_estimator, n_estimators, learning_rate, algorithm,parameters)
if __name__ == "__main__":
   main(sys.argv[1:])
