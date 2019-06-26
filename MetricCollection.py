__author__ = 'Pranav Bahl, Casey Sader'

from sklearn.metrics import roc_auc_score,recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
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
	modelfile = inputData['models']
	importancefile = inputData['featureimportance']
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
	assert len(resultsfile) == len(modelfile), "results not equal model"
	assert len(resultsfile) == len(importancefile), "results not equal importance"
	for r,m,f,i in zip(resultsfile,modelfile,importancefile,range(len(resultsfile))):
		test_labels = pd.read_csv(r)
		true_labels = test_labels[label]
		predictions = test_labels['predictions']
		accuracy=float(accuracy_score(true_labels,predictions))
		precision = float(precision_score(true_labels,predictions))
		recall = float(recall_score(true_labels,predictions))
		f1_Score = float(f1_score(true_labels,predictions))
		try:
			mcc = float(matthews_corrcoef(true_labels,predictions))
		except ValueError:
			mcc = 0.0
		try:
			std = np.std(predictions).astype(np.float)
		except :
			std = 0.0
		try:
			roc_auc = float(roc_auc_score(true_labels,predictions))
		except ValueError:
			roc_auc = 0.0
		objId = int(db.seqs.find_and_modify(query={ 'collection' : 'result' },update={'$inc': {'id': 1}},fields={'id': 1, '_id': 0},new=True).get('id'))
		db.Result.insert({"_id":objId,"pp_id":pp_id,"sd_id":sd_id,"file_id":i+1,"feature_extraction_id":fe_id,"feature_selection_id":fs_id,"algo_id":algo_id,"accuracy":accuracy,"precision":precision,"recall":recall,"f1_score":f1_Score,"roc_auc":roc_auc,"mcc":mcc,"standard_deviation":std,"model_path":m,"feature_importance_path":f})
		metric_calc = dict()
		metric_calc['accuracy'] = accuracy
		metric_calc['precision'] = precision
		metric_calc['recall'] = recall
		metric_calc['f1_Score'] = f1_Score
		metric_calc['roc_auc'] = roc_auc
		yaml.dump(metric_calc,open(outfile+'/metric'+str(i+1)+'.yaml','w'))


def main(args):
	inputfile=''
	outputfolder=''
	try:
		opts,args=getopt.getopt(args,"i:o:",[])
	except getopt.GetoptError:
		print 'MetricCollection.py -i <inputfile> -o <outputfolder>'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			inputfile=arg
		elif opt=='-o':
			outputfolder=arg
	metricCalculation(inputfile,outputfolder)

if __name__ == "__main__":
   main(sys.argv[1:])
