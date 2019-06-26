__author__ = 'Casey Sader'

import os
import sys, getopt
import numpy as np
import pandas as pd
import yaml
import csv
import pickle
from scipy.io.arff import loadarff
import re
import time
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
import eli5
from eli5.sklearn import PermutationImportance

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
# https://www.heatonresearch.com/2017/07/22/keras-getting-started.html
def to_xy(df, target):
	result = []
	for x in df.columns:
		if x != target:
			result.append(x)
	dummies = pd.get_dummies(df[target])
	return df.as_matrix(result), dummies.as_matrix().astype(np.int32)

	#
	# This is from the site and doesn't work/isn't needed, but could be helpful to still see
	#
	# find out the type of the target column.  Is it really this hard? :(
	# target_type = df[target].dtypes
	# target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

	# # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
	# if target_type in (np.int64, np.int32):
	# 	# Classification
	# 	dummies = pd.get_dummies(df[target])
	# 	return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.int32)
	# else:
	# 	# Regression
	# 	return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)

def calculateFeatureImportance(model,train_features,test_x,test_y,importancefolder,i):
	feature_names = train_features.columns.tolist()

	perm = PermutationImportance(model).fit(test_x, test_y)
	feat_imp = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names=feature_names)
	new_feat_imp = feat_imp.drop('std', axis=1)

	feature_rows = new_feat_imp.values
	importanceFile = importancefolder + "NeuralNetworkFI" + str(i+1) + ".csv"

	with open(importanceFile, "w") as f:
		writer = csv.writer(f)
		FI_header = ["name", "importance"]
		writer.writerow(FI_header)
		for row in feature_rows:
			writer.writerow(row)
	f.close()

	return importanceFile



def create_model(train_x, activation_function, layers, input_drop, hidden_drop, learning_rate, num_labels):
		model = Sequential()

		model.add(Dense(layers[0], activation = activation_function, input_dim = train_x.shape[1]))


		for j in range(len(layers) - 1):
			model.add(Dense(layers[j+1], activation = activation_function))

		model.add(Dense(num_labels,activation = 'softmax'))

		opt = optimizers.Adam(lr=learning_rate)
		model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

		return model

def tfNeuralNetwork(datafile, outputfolder, activation_function, layers, input_drop, hidden_drop, epochs, batch_size, learning_rate, parameters):
	inputData = yaml.load(open(datafile))
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
		train_labels = pd.DataFrame(train_df[label])
		train_features = train_df.drop(label,axis=1)
		test_df = pd.read_csv(testingSet[i])
		test_predictions = pd.DataFrame(test_df[label])

		train_x, train_y = to_xy(train_df, label)
		test_x, test_y = to_xy(test_df, label)

		num_labels = len(train_y[0])

		model = KerasClassifier(build_fn=create_model, train_x=train_x, activation_function=activation_function, layers=layers, input_drop=input_drop, hidden_drop=hidden_drop, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, num_labels=num_labels)
		model.fit(train_x, train_y)

		modelFile = modelsfolder + "NeuralNetworkModel" + str(i+1) + ".h5"
		model.model.save(modelFile)
		modelset.append(modelFile)

		importanceFile = calculateFeatureImportance(model,train_features,test_x,test_y,importancefolder,i)
		importanceset.append(importanceFile)

		possible_labels = sorted(pd.unique(pd.merge(train_labels[label],test_predictions[label])[label]))

		test_predictions['predictions'] = model.predict(test_x)

		try:
			# This is for a case where labels shifted, i.e. labels are [1,2,3] but would be predicting [0,1,2]
			# Also just works if labels are correct for something like binary [0,1]
			for predindex in range(len(test_predictions['predictions'])):
				test_predictions['predictions'][predindex] = possible_labels[test_predictions['predictions'][predindex]]
		except:
			# case if it ends up doing floating point predictions
			try:
				for predindex in range(len(test_predictions['predictions'])):
					for labindex in range(len(possible_labels)):
						if possible_labels[labindex] == possible_labels[-1]:
							test_predictions['predictions'][predindex] = possible_labels[labindex]
							break
						elif test_predictions['predictions'][predindex] < (possible_labels[labindex] + possible_labels[labindex+1])/2:
							test_predictions['predictions'][predindex] = possible_labels[labindex]
							break
			except:
				# something went wrong
				raise
				sys.exit(2)

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
	resultDict['algorithm'] = "TFNeuralNetwork"
	yaml.dump(resultDict,open(outputfolder+'/results.yaml','w'))	


def main(args):
	ActivationFunction="relu"   # softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, exponential, linear
	InputDropout = 0		# Percentage of nuerons to dropout during training phase (input layer) Value btw 0 and 1
	HiddenDropout =	0.5	# Percentage of nuerons to dropout during training phase (hidden layers) Value btw 0 and 1, Recommended 0.5
	LearningRate = 0.001
	layers=[100,100,100]
	epochs = 10
	BatchSize = None

	datafile=''
	outputfolder=''
	parameters=dict()
	try:
		opts,args=getopt.getopt(args,"i:o:a:l:d:h:r:",[])
	except getopt.GetoptError:
		print 'datasplit.py -i <inputfile> -o <outputfolder> -a <Activation Function> -l <Layers> -d <Input Dropout> -h <HiddenDropout> -e <epochs> -b <batch size> -r <learning rate>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			datafile=arg
		elif opt=='-o':
			outputfolder=arg
		elif opt=='-a':
			ActivationFunction=int(arg)
			parameters['parameter.a']=arg
		elif opt=='-l':
			#layers = map(int, arg.strip('[]').split(','))
			layers = map(int, re.split("\s*,\s*",arg.strip('[]')))
			parameters['parameter.l']=arg
		elif opt == "-d":
			InputDropout=float(arg)
			parameters['parameter.d']=arg
		elif opt=="-h":
			HiddenDropout=float(arg)
			parameters['parameter.h']=arg
		elif opt == "-e":
			epochs=int(arg)
			parameters['parameter.e']=arg
		elif opt == "-b":
			BatchSize=int(arg)
			parameters['parameter.b']=arg
		elif opt=="-r":
			LearningRate=float(arg)
			parameters['parameter.r']=arg

	tfNeuralNetwork(datafile, outputfolder, ActivationFunction, layers, InputDropout, HiddenDropout, epochs, BatchSize, LearningRate, parameters)


if __name__ == "__main__":
	main(sys.argv[1:])
