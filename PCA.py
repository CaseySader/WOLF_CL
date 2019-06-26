__author__ = 'Pranav Bahl'

import os
import arff
import yaml
import sys, getopt
from scipy.io.arff import loadarff
from sklearn.decomposition import PCA
import pandas as pd
import errno

def principalComponentAnalysis(data_file,output_folder,number_of_components,copy,whiten,parameters):
	train_files = yaml.load(open(data_file))['training']
	test_files = yaml.load(open(data_file))['testing']
	inputFile = yaml.load(open(data_file))['inputFile']
	label = yaml.load(open(data_file))['label']
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
		if number_of_components == 0 or number_of_components == None or number_of_components > min(len(train_features),len(train_features.columns)):
			number_of_components = train_features.shape[1]
		pca = PCA(n_components=number_of_components,copy=copy,whiten=whiten)

		#fitting PCA to train data
		pca_fit = pca.fit(train_features)

		#transforming test and train data
		train_features_transf = pca_fit.transform(train_features)
		test_features_transf = pca_fit.transform(test_features)

		#creating final dataframes
		train_pca_df = pd.DataFrame(train_features_transf)#,columns=train_features.columns)
		train_pca_df['class'] = train_labels
		test_pca_df = pd.DataFrame(test_features_transf)#,columns=test_features.columns)
		test_pca_df['class'] = test_labels

		#saving result to file
		train_pca_df.dropna().to_csv(output_folder+'/'+train_file,index=False)
		test_pca_df.dropna().to_csv(output_folder+'/'+test_file,index=False)

	metadata['training'] = trainingSet
	metadata['testing'] = testingSet
	metadata['label'] = label
	
	if not parameters:
		parameters['parameter']='default'
	if 'preprocessing_params' in yaml.load(open(data_file)):
		metadata['preprocessing_params'] = yaml.load(open(data_file))['preprocessing_params']
	metadata['feature_extraction_parameters'] = parameters
	metadata['feature_extraction_algorithm'] = 'PCA'
	metadata['split_params'] = yaml.load(open(data_file))['split_params']
	metadata['inputFile'] = inputFile
	yaml.dump(metadata,open(output_folder+'/splitdatafiles.yaml','w'))

def main(args):
	data_file=''
	output_folder=''
	parameters=dict()
	number_of_components=None #Number of components to keep. if n_components is not set all components are kept
	copy = True #If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead.
	whiten = False #When True (False by default) the components_ vectors are divided by n_samples times singular values to ensure uncorrelated outputs with unit component-wise variances.
	try:
		opts,args=getopt.getopt(args,"i:o:n:c:w:",[])
	except getopt.GetoptError:
		print 'PCA.py -i <inputfile> -o <output_folder> -n <number_of_components> -c <copy> -w <whiten>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			data_file=arg
		elif opt=='-o':
			output_folder=arg
		elif opt=='-n':
			number_of_components=int(arg)
			parameters['parameter.n']=arg
		elif opt=='-c':
			if arg == 'True':
				copy=True
			elif arg=='False':
				copy=False
			else:
				raise ValueError
			parameters['parameter.c']=arg
		elif opt=='-w':
			if arg == 'True':
				whiten=True
			elif arg=='False':
				whiten=False
			else:
				raise ValueError
			parameters['parameter.w']=arg
	principalComponentAnalysis(data_file,output_folder,number_of_components,copy,whiten,parameters)

if __name__ == "__main__":
   main(sys.argv[1:])
