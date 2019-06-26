__author__ = 'Pranav Bahl'

from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error
from sklearn.metrics import cohen_kappa_score
import yaml
import os
import sys, getopt
from pymongo import MongoClient
import numpy as np
import pandas as pd
import errno

def metricCalculation(inputfile,outfile):
	uri = "mongodb://wolfAdmin:wo20lf@db2.local/wolf"
	client = MongoClient(uri)
	db = client.wolf
	inputData = yaml.load(open(inputfile))
	inputData['split_params']["file"]=inputData['inputFile']
	inputData['algo_params']["executable"] = inputData['algorithm']
	resultsfile = inputData['results']
	label = inputData['label']
	sd_id= -1
	algo_id = -1
	fs_id = -1
	fe_id = -1
	pp_id = -1
	for s in  db.SplitData.find(inputData['split_params'],{"_id":1}):
		sd_id = s['_id']
	for r in db.Algorithm.find(inputData['algo_params'],{"_id":1}):
		algo_id = r['_id']
	if 'preprocessing_params' in inputData:
		for r in db.PreProcessing.find(inputData['preprocessing_params'],{"_id":1}):
			pp_id = r['_id']
	if 'feature_extraction_algorithm' in inputData:
		inputData['feature_extraction_parameters']['executable']=inputData['feature_extraction_algorithm']
		for r in db.FeatureExtraction.find(inputData['feature_extraction_parameters'],{"_id":1}):
			fe_id = r['_id']
	if 'feature_selection_algorithm' in inputData:
		inputData['feature_selection_parameters']["executable"]=inputData['feature_selection_algorithm']
		for r in db.FeatureSelection.find(inputData['feature_selection_parameters'],{"_id":1}):
			fs_id = r['_id']
	for r,i in zip(resultsfile,range(len(resultsfile))):
		test_labels = pd.read_csv(r)
		y_true = test_labels[label]
		y_pred = test_labels['predictions']
	#	accuracy=float(accuracy_score(true_labels,predictions))
	#	precision = float(precision_score(true_labels,predictions,average="macro"))
	#	recall = float(recall_score(true_labels,predictions,average="macro"))
	#	f1_Score = float(f1_score(true_labels,predictions,average="macro"))
	#	cohen_kappa=float(cohen_kappa_score(true_labels, predictions))
	#	try:
	#		mcc = float(matthews_corrcoef(true_labels,predictions))
	#	except ValueError:
	#		mcc = 0.0
	#	try:
	#		std = np.std(predictions).astype(np.float)
	#	except :
	#		std = 0.0
	#	try:
	#		roc_auc = float(roc_auc_score(true_labels,predictions,average="macro"))
	#	except ValueError:
	#		roc_auc = 0.0
		EVS = float(explained_variance_score(y_true, y_pred))
		MAE = float(mean_absolute_error(y_true, y_pred))
		MSE = float (mean_squared_error(y_true, y_pred))
		MedAE = float(median_absolute_error(y_true, y_pred))
		objId = int(db.seqs.find_and_modify(query={ 'collection' : 'result' },update={'$inc': {'id': 1}},fields={'id': 1, '_id': 0},new=True).get('id'))
		db.Result.insert({"_id":objId,"pp_id":pp_id,"sd_id":sd_id,"file_id":i+1,"feature_extraction_id":fe_id,"feature_selection_id":fs_id,"algo_id":algo_id,"EVS":EVS,"MAE":MAE,"MSE":MSE,"MedAE":MedAE})
		metric_calc = dict()
		metric_calc['EVS'] = EVS
		metric_calc['MAE'] = MAE
		metric_calc['MSE'] = MSE
		metric_calc['MedAE'] = MedAE
		yaml.dump(metric_calc,open(outfile+'/metric'+str(i+1)+'.yaml','w'))


def main(args):
	inputfile=''
	outputfolder=''
	try:
		opts,args=getopt.getopt(args,"i:o:",[])
	except getopt.GetoptError:
		print 'MetricCollection_try.py -i <inputfile> -o <outputfolder>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			inputfile=arg
		elif opt=='-o':
			outputfolder=arg
	metricCalculation(inputfile,outputfolder)

if __name__ == "__main__":
   main(sys.argv[1:])
