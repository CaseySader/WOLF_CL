__author__ = 'Pranav Bahl'

import os
import arff
import yaml
import random, math
import sys, getopt
import numpy as np
from array import array
from scipy.io.arff import loadarff
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import errno
import h5py

def splitdata(inputfile,outputfolder,folds,iterations,label,fileName,parameters):
	trainingSet = []
	testingSet = []
	data_files = dict()
	data=[]
	count = 0
	if not os.path.exists(outputfolder):
		try:
			os.makedirs(outputfolder)
		except OSError as exc:
		    if exc.errno != errno.EEXIST:
		        raise exc
		    pass
	if inputfile.split('.')[-1] == 'arff':
		data,meta = loadarff(inputfile)
		data = pd.DataFrame(data)
	else:
		data = pd.read_csv(inputfile)
	labels = data[label].values
		
	
	for x in xrange(iterations):
		skf = StratifiedKFold(labels,folds,shuffle=True)
		for train,test in skf:
			count = count+1
			testfile = outputfolder + '/test_'+str(count)+'.csv'
			trainfile= outputfolder + '/train_'+str(count)+'.csv'
			HDF5Train=outputfolder +"/train_"+str(count)+".h5"
        		HDF5Test=outputfolder +"/test_"+str(count)+".h5"
			
			tr_labels=[]
        		tr_features=[]
        		for j in train:
            			tr_labels.append(labels[j])
            			tr_features.append(data.iloc[j,0:8])
        		tst_labels=[]
        		tst_features=[]
        		for k in test:
            			tst_labels.append(labels[k])
            			tst_features.append(data.iloc[k,0:8])
            		with h5py.File(HDF5Train, 'w') as f:
				f['label'] =tr_labels
				f['data'] =tr_features
			
			with h5py.File(HDF5Test, 'w') as f:
				f['label'] =tst_labels
				f['data'] =tst_features

			data.iloc[train].to_csv(trainfile,index=False)
			data.iloc[test].to_csv(testfile,index=False)
			trainingSet.append(trainfile)
			testingSet.append(testfile)
	data_files['training'] = trainingSet
	data_files['testing'] = testingSet
	data_files['label'] = label

	pp_file = outputfolder[:outputfolder.rfind('/')]+'/PreProcessing.yaml'
	if os.path.exists(pp_file):
		data_files['preprocessing_params'] = yaml.load(open(pp_file))['preprocessing_params']
	if not parameters:
		parameters['parameter']='default'
	data_files['split_params'] = parameters
	data_files['inputFile'] = fileName
	yaml.dump(data_files,open(outputfolder+'/splitdatafiles.yaml','w'))

def main(args):
	inputfile=''
	outputfolder=''
	parameters=dict()
	folds=5
	iterations=1
	label=''
	fileName=''
	try:
		opts,args=getopt.getopt(args,"i:o:k:r:l:d:",[])
	except getopt.GetoptError:
		print 'datasplit.py -i <inputfile> -o <outputfile> -k <folds> -r <number_of_repitiotions> -l <class_label>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			inputfile=arg
		elif opt=='-o':
			outputfolder=arg
		elif opt=='-k':
			folds=int(arg)
			parameters['parameter.k']=arg
		elif opt=='-r':
			iterations=int(arg)
			parameters['parameter.r']=arg
		elif opt=='-l':
			label=arg
			parameters['parameter.l']=arg
		elif opt=='-d':
			fileName=arg
	if label is '':
		raise ValueError
	else:
		splitdata(inputfile,outputfolder,folds,iterations,label,fileName,parameters)

if __name__ == "__main__":
   main(sys.argv[1:])
