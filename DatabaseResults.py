__author__ = "Pranav Bahl, Casey Sader"

from pymongo import MongoClient
import argparse
import csv
import os
import numpy as np
import pandas as pd
from cStringIO import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import errno

pd.set_option('display.max_columns', 30)

class DatabaseResults:

	def __init__(self,args):
		#self.uri = "mongodb://wolfAdmin:wo20lf@db2.acf.ku.edu/wolf"
		self.uri = "mongodb://wolfAdmin:wo20lf@db2.local/wolf"
		self.client = MongoClient(self.uri)
		self.db = self.client.wolf
		self.sd_file_name = args.i
		self.result_file = args.o
		self.processResult()

	def processResult(self):
		sd_ids=[]
		resfile = os.getcwd()+'/Results.xlsx'
		if self.result_file:
			resfile = self.result_file
		for sd_id in self.db.SplitData.find({"file":self.sd_file_name},{"_id":1}):
			sd_ids.append(sd_id['_id'])

		results = self.db.Result.aggregate([{'$match':{'sd_id':{'$in':sd_ids}}},{'$group': {'_id' : {'algo_id' : "$algo_id"},'recall' : {'$avg' : "$recall"}, 'precision' : {'$avg' : "$precision"}, 'accuracy' : {'$avg' : "$accuracy"},'f1_score':{'$avg' : "$f1_score"},'mcc':{'$avg' :"$mcc"},'roc_auc':{'$avg' :"$roc_auc"},'std':{'$avg' :"$standard_deviation"} }}])

		models = self.db.Result.aggregate([{'$match':{'sd_id':{'$in':sd_ids}}},{'$group': {'_id' : None, 'algo_id' : {'$push' : "$algo_id"},'model_path' : {'$push' : "$model_path"}, 'feat_imp' : {'$push' : "$feature_importance_path"},'roc_auc' : {'$push' :"$roc_auc"} }}])

		with open(str(resfile), 'w') as csvfile:
			fieldnames = ['algo_config', 'precision','recall','accuracy','f1_score','mcc','roc_auc','model_path']
			algo_configs = []
			recall = []
			precision = []
			accuracy = []
			f1_score = []
			algo_names = []
			algo_params = []
			mcc = []
			roc_auc = []

			id_roc_auc = {}
			for result in results:
				id_roc_auc[result['_id']['algo_id']] = result['roc_auc']
				recall.append(result['recall'])
				precision.append(result['precision'])
				accuracy.append(result['accuracy'])
				f1_score.append(result['f1_score'])
				mcc.append(result['mcc'])
				roc_auc.append(result['roc_auc'])

				algo_param = ""
				algo_name = ""
				for algo_config in self.db.Algorithm.find({"_id":result['_id']['algo_id']}):
					algo_param = algo_config['parameter']
					algo_name = algo_config['executable']

				algo_names.append(algo_name)
				if type(algo_param) is dict:
					ualgo_param={}
					for p,v in algo_param.iteritems():
						ualgo_param[str(p)]=str(v)
				else:
					ualgo_param = str(algo_param)
				algo_params.append(ualgo_param)
				algo_configs.append('Algorithm:'+str(algo_name)+" \n "+' parameter:'+str(algo_param))
			
			# Find model path and feature importance path for best configuration
			max_roc_auc_id =  max(id_roc_auc.iterkeys(), key=lambda k: id_roc_auc[k])  # id of maximum roc_auc value
			best_model = ""
			best_feature_importances = ""
			# this should just be one loop through as there is one container in models
			for model in models:
				highest_roc = 0.0
				for i in range(len(model['algo_id'])):
					if model['algo_id'][i] == max_roc_auc_id:
						if model['roc_auc'][i] > highest_roc:
							highest_roc = model['roc_auc'][i]
							best_model = model['model_path'][i]
							best_feature_importances = model['feat_imp'][i]

			recall.append(np.std(recall))
			precision.append(np.std(precision))
			accuracy.append(np.std(accuracy))
			algo_configs.append('std')
			algo_names.append('std')
			algo_params.append('')
			f1_score.append(np.std(f1_score))
			mcc.append(np.std(mcc))
			roc_auc.append(np.std(roc_auc))
			data = pd.DataFrame({'algo_config':algo_configs,'algo_name':algo_names,'algo_param':algo_params,'precision':precision,'recall':recall,'accuracy':accuracy,'f1_score':f1_score,'mcc':mcc,'roc_auc':roc_auc})
			writer = pd.ExcelWriter(str(resfile), engine='xlsxwriter')
			idx = data.groupby(['algo_name'])['roc_auc'].transform(max) == data['roc_auc']
			plot_df = data[idx].drop('algo_config',axis=1).reset_index(drop=True)
			plot_df.to_excel(writer,sheet_name="Results")
			for algo in data['algo_name'].unique():
				if algo != 'std':
					data[data['algo_name'] == algo].drop('algo_config',axis=1).to_excel(writer,sheet_name=algo[:31])
			data[data['roc_auc'] == max(data['roc_auc'])].drop('algo_config',axis=1).to_excel(writer,sheet_name="Best_configuration")
			
			model_path_out = pd.DataFrame(columns=['model_path'])
			model_path_out['model_path'] = [best_model]
			model_path_out.to_excel(writer,sheet_name="Best_configuration", startcol=9, index=False)

			feat_imp_out = pd.read_csv(best_feature_importances)
			feat_imp_out.to_excel(writer,sheet_name="Best_configuration", startrow=3, startcol=1, index=False)

			cols = [col for col in data.columns if col not in ['algo_config']]
			workbook  = writer.book
			worksheet = writer.sheets['Results']
			observations = len(data)
			count = 0
			for i in plot_df.columns:
				if i not in ['algo_config','algo_name','algo_param']:
					imgdata = StringIO()
					fig,ax = plt.subplots()
					plot_df[plot_df['algo_name'] != 'std'][i].plot.bar(ax=ax)
					ax.set_xticklabels(plot_df[plot_df['algo_name'] != 'std'].index,rotation = 0)
					ax.set_xlabel(i)
					ax.set_ylabel("score")
					fig.savefig(imgdata)
					worksheet.insert_image(observations+5+(35*count),1,"",{'image_data':imgdata})
					count += 1
					
			writer.save()
			writer.close()

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', help='Splitdata filename')
	parser.add_argument('-o', help='Result filename')
	args = parser.parse_args()

	dbRes = DatabaseResults(args)
