__author__ = 'Pranav Bahl'

import os
import arff
import yaml
import sys, getopt
from scipy.io.arff import loadarff
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import errno

def svm_rfe(data_file,output_folder,number_of_features,step,parameters):
	inputData = yaml.load(open(data_file))
	train_files = inputData['training']
	test_files = inputData['testing']
	inputFile = inputData['inputFile']
	label = inputData['label']
	metadata = dict()
	trainingSet = []
	testingSet = []

	if not os.path.exists(output_folder):
		try:
			os.makedirs(output_folder)
		except OSError as exc:
		    if exc.errno != errno.EEXIST:
		        raise exc
		    pass
	for i in range(len(train_files)):
		test_file ='test_'+str(i+1)+'.csv'
		train_file='train_'+str(i+1)+'.csv'
		trainingSet.append(output_folder+"/"+train_file)
		testingSet.append(output_folder+"/"+test_file)

		#loading training and testing data set
		train_df = pd.read_csv(train_files[i]).dropna()
		test_df = pd.read_csv(test_files[i]).dropna()

		#extraining train and test features & labels
		train_labels = train_df[label]
		train_features = train_df.drop(label,axis=1)
		test_labels = test_df[label]
		test_features = test_df.drop(label,axis=1)

		#initiating PCA
		if number_of_features == 0 or number_of_features > len(train_features.columns):
			number_of_components = None
		estimator = SVR(kernel="linear")
		selector = RFE(estimator, n_features_to_select=number_of_features, step=step)

		#fitting PCA to train data
		selector_fit = selector.fit(train_features,train_labels)

		#transforming test and train data
		train_features_transf = selector_fit.transform(train_features)
		test_features_transf = selector_fit.transform(test_features)

		#creating final dataframes
		train_svm_rfe_df = pd.DataFrame(train_features_transf)
		train_svm_rfe_df['class'] = train_labels
		test_svm_rfe_df = pd.DataFrame(test_features_transf)
		test_svm_rfe_df['class'] = test_labels

		#saving result to file
		train_svm_rfe_df.dropna().to_csv(output_folder+'/'+train_file,index=False)
		test_svm_rfe_df.dropna().to_csv(output_folder+'/'+test_file,index=False)

	metadata['training'] = trainingSet
	metadata['testing'] = testingSet
	metadata['label'] = label

	if not parameters:
		parameters['parameter']='default'
	metadata['feature_selection_parameters'] = parameters
	metadata['feature_selection_algorithm'] = "SVM_RFE"
	if 'feature_extraction_parameters' in inputData:
		metadata['feature_extraction_parameters'] = inputData['feature_extraction_parameters']
		metadata['feature_extraction_algorithm'] = inputData['feature_extraction_algorithm']
	if 'preprocessing_params' in inputData:
		metadata['preprocessing_params'] = inputData['preprocessing_params']
	metadata['split_params'] = inputData['split_params']
	metadata['inputFile'] = inputFile
	yaml.dump(metadata,open(output_folder+'/splitdatafiles.yaml','w'))

def main(args):
	data_file=''
	output_folder=''
	parameters=dict()
	number_of_features=None #The number of features to select. If None, half of the features are selected.
	step = 1 #If greater than or equal to 1,then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
	try:
		opts,args=getopt.getopt(args,"i:o:n:s:",[])
	except getopt.GetoptError:
		print 'SVM_RFE.py -i <inputfile> -o <output_folder> -n <number_of_features> -s <step>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			data_file=arg
		elif opt=='-o':
			output_folder=arg
		elif opt=='-n':
			number_of_features=int(arg)
			parameters['parameter.n']=arg
		elif opt=='-s':
			step= int(arg)
			parameters['parameter.s']=arg
	svm_rfe(data_file,output_folder,number_of_features,step,parameters)

if __name__ == "__main__":
   main(sys.argv[1:])
