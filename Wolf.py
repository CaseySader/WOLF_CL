__author__ = 'Pranav Bahl and Casey Sader'

import yaml
import sys, getopt
import argparse
import itertools
import math
import os
import subprocess,shlex
import time
from pymongo import MongoClient
from collections import defaultdict
from subprocess import Popen, PIPE
import errno
import PreProcessing
import re
import numpy as np
class WolfJob:

	def __init__(self, args):
		#self.uri = "mongodb://wolfAdmin:wo20lf@db2.acf.ku.edu/wolf"
		self.uri = "mongodb://wolfAdmin:wo20lf@db2.local/wolf"
		self.client = MongoClient(self.uri)
		self.db = self.client.wolf

		self.maxNodes = 8
		self.maxCpu = 8*20
		wolfDir = os.path.realpath(__file__)
		self.workDir = wolfDir[:wolfDir.rfind("/")]
		self.resultFile = ""

		self.metricJobs = []
		self.output_folder = ""
		self.email = ""
		self.inputfile=args.i
		self.folder_list = []
		self.prev_folder = []
		self.transactions = set()
		self.job_list = defaultdict(list)
		self.createJobs()
		
		

	def createJobs(self):
		config = yaml.load(open(self.inputfile))
		try:
			self.email = config['cluster_config']['email']
		except:
			pass
		featureExtraction = ''
		featureSelection = ''
		preProcessing = ''

		justPredict = True
		if 'pre_processing' in config:
			preProcessing = config['pre_processing']
			self.generateCommand(preProcessing,'pre_processing')
			self.output_folder = preProcessing['output_folder']
			config['datasplit']['data_file'] = self.output_folder+'/DataFile.csv'
		
		if 'datasplit' in config:
			justPredict = False
			datasplit = config['datasplit']
			if self.output_folder =="":
				self.output_folder = datasplit['output_folder']
			self.generateCommand(datasplit,'datasplit')


		if 'feature_extraction' in config:
			featureExtraction = config['feature_extraction']
			for i in featureExtraction:
				if i != "no_of_files":
					self.generateCommand(featureExtraction[i],'feature_extraction')

		if 'feature_selection' in config:
			featureSelection = config['feature_selection']
			for i in featureSelection:
				if i != "no_of_files":
					self.generateCommand(featureSelection[i],'feature_selection')

		if 'algorithm' in config:
			algorithm = config['algorithm']
			for i in algorithm:
				if i != "no_of_files":
					self.generateCommand(algorithm[i],'algorithm')

			metricCalculation = config['metric_calculation']
			self.generateCommand(metricCalculation,'metric_calculation')
			print "Successfully submitted jobs on cluster"

		if 'model_prediction' in config:
			prediction = config['model_prediction']
			self.generateCommand(prediction, 'model_prediction')
			pbs_config_pred = "#!/bin/bash -l\n\
#SBATCH --job-name={transaction_name}\n\
#SBATCH -N {nodes} -n {nodes} --mem=20g --time=72:00:00\n\
#SBATCH --mail-user={email}\n\
#SBATCH --mail-type=FAIL\n\
#SBATCH --workdir={location}\n\
#SBATCH -e {output_path}.err\n\
#SBATCH -o {output_path}.out\n\
#SBATCH --array={job_array_size} \n\
\n\
module load scikit-learn/0.18.0-Python-2.7.12\n\
module load TensorFlow/1.11-cp27-gpu \n\
"
			pred_job_filename = self.workDir + "/predictions/prediction.sh"
			if not os.path.exists(self.workDir + "/predictions/"):
				try:
					os.makedirs(self.workDir + "/predictions/")
				except OSError as exc:
					if exc.errno != errno.EEXIST:
						raise exc
					pass
			pred_job_file = open(pred_job_filename,'w')
			pred_job_file.write(pbs_config_pred.format(nodes=1,number_of_processors=1,email=self.email,location=self.workDir,
			transaction_name = 'predict', output_path=self.workDir + "/predictions/" + "predict",job_array_size=str("0-")+str(0))+"\n")
			for j in self.job_list['model_prediction']:
				pred_job_file.write(j)
			pred_job_file.close()
			proc = Popen("sbatch "+ pred_job_filename, shell=True, stdout=PIPE)

		if not justPredict:
			self.writeJobScript(datasplit,algorithm,metricCalculation,featureExtraction,featureSelection,preProcessing)

	def generateCommand(self,component,function):
		splitcmd = 'python ' + component['executable'] + ' -i '
		paramset = defaultdict(set)
		filesCount = 1
		if (function != 'metric_calculation'):
			if 'parameters' in component.keys():
				if function not in self.transactions:
					self.prev_folder = self.folder_list
					self.folder_list = []
				list_of_lists = list(map(lambda x: self.getOptionsList(x), component['parameters']))
				option_strings = [list(tup) for tup in itertools.product(*list_of_lists)]
				for i,option in enumerate(option_strings):
					params = ''
					for row in option:
						params += ' ' + row[0] + ' '+str(row[1])
						if((function == 'datasplit') and (row[0] <> '-l')):
							paramset[row[0]].add(row[1])
					if (function == 'pre_processing'):
						self.insertParams(params,component['executable'],function)
						self.job_list[function].append(splitcmd + component['data_file'] + ' -o ' + component['output_folder'] + params)
					if (function == 'datasplit'):
						self.insertSplitParams(params,component['data_file'])
						datasplitFolder = component['output_folder']+'/'+component['executable'].split('.')[0]+str(i+1)
						self.folder_list.append(datasplitFolder)
						fileName = "".join(component['data_file'].split(".")[:-1])+"_"+str(os.getpid())+"."+component['data_file'].split(".")[-1]
						self.job_list[function].append(splitcmd + component['data_file'] + ' -o ' + datasplitFolder + ' -d ' + fileName + params)
					if (function == 'model_prediction'):
						self.insertParams(params,component['executable'],function)
						self.job_list[function].append(splitcmd + component['data_file'] + params)
					else:
						params = re.sub(",\s+", ",", params);
						for j in self.prev_folder:
							folder_name = j + '/' + component['executable'].split('.')[0]+'_result'+ str(i+1)
							self.folder_list.append(folder_name)
							self.job_list[function].append(splitcmd + j +'/splitdatafiles.yaml' + ' -o ' + ' ' + folder_name + params)
						self.insertParams(params,component['executable'],function)
				if (function == 'datasplit'):
					for x in paramset:
						filesCount *= max(paramset[x])
					self.insertFiles(filesCount)
			else:
				if function not in self.transactions:
					self.prev_folder = self.folder_list
					self.folder_list = []
				folder_name = ''
				if (function == 'datasplit'):
					self.insertSplitParams('',component['data_file'])
					self.insertFiles(5)
					folder_name = component['output_folder']+'/'+component['executable'].split('.')[0]
					self.folder_list.append(folder_name)
					fileName = "".join(component['data_file'].split(".")[:-1])+"_"+str(os.getpid())+"."+component['data_file'].split(".")[-1]
					self.job_list[function].append(splitcmd + component['data_file'] + ' -o ' + folder_name +' -d '+fileName)
				elif(function == 'pre_processing'):
						self.insertParams('',component['executable'],function)
						self.job_list[function].append(splitcmd + component['data_file'] + ' -o ' + component['output_folder'])
				else:
					self.insertParams('',component['executable'],function)
					for i,f_name in enumerate(self.prev_folder):
						folder_name = f_name + '/' +component['executable'].split('.')[0]+'_result'
						self.folder_list.append(folder_name)
						self.job_list[function].append(splitcmd + f_name +'/splitdatafiles.yaml' + ' -o ' + ' ' + folder_name)
			self.transactions.add(function)

		elif(function == 'metric_calculation'):
			if 'result_file' in component:
				self.resultFile = component['result_file']
			for i in self.folder_list:
				self.metricJobs.append(splitcmd + i + '/results.yaml' + ' -o ' + i )
		else:
			print "Wrong input"

	def insertSplitParams(self,params,dataFile):
		parameters = 'default'
		fileName = "".join(dataFile.split(".")[:-1])+"_"+str(os.getpid())+"."+dataFile.split(".")[-1]
		if params != '':
			param = params.replace('-','').split()
			parameters = dict(param[i:i+2] for i in range(0,len(param),2))
		if(0 == self.db.SplitData.find({"file":fileName,"parameter": parameters}).count()):
			objId = int(self.db.seqs.find_and_modify(query={ 'collection' : 'splitdata' },update={'$inc': {'id': 1}},fields={'id': 1, '_id': 0},new=True).get('id'))
			self.db.SplitData.insert({"_id":objId,"file":fileName,"parameter": parameters})

	def insertFiles(self,filesCount):
		for count in range(filesCount):
			if(0 == self.db.Files.find({"_id":count+1}).count()):
				self.db.Files.insert_one({"_id":count+1,"train_file":'train_'+str(count+1)+'.csv',"test_file":'test_'+str(count+1)+'.csv'})

	def insertParams(self,params,component,task_name):
		executable = component[0:component.find('.')]
		parameters = 'default'
		if params != '':
			p = re.compile('-([a-zA-Z]\s+.*?)((?=\s+-)|$)');
                	iterator = p.finditer(params);
			param=[]
                	for m in iterator:
				paramPairStr = m.group(1)
    				paramPair = re.split("\s+", paramPairStr);
    				param.extend(paramPair)
			#param = params.replace('-','').split()
			parameters = dict(param[i:i+2] for i in range(0,len(param),2))
		if task_name == 'pre_processing':
			if(0 == self.db.PreProcessing.find({"executable":executable,"parameter": parameters}).count()):
				pp_id = int(self.db.seqs.find_and_modify(query={ 'collection' : 'preProcessing' },update={'$inc': {'id': 1}},fields={'id': 1, '_id': 0},new=True).get('id'))
				self.db.PreProcessing.insert({"_id":pp_id,"executable":executable,"parameter": parameters})

		if task_name == 'algorithm':
			if(0 == self.db.Algorithm.find({"executable":executable,"parameter": parameters}).count()):
				algo_id = int(self.db.seqs.find_and_modify(query={ 'collection' : 'algorithm' },update={'$inc': {'id': 1}},fields={'id': 1, '_id': 0},new=True).get('id'))
				self.db.Algorithm.insert({"_id":algo_id,"executable":executable,"parameter": parameters})
		if task_name == 'feature_extraction':
			if(0 == self.db.FeatureExtraction.find({"executable":executable,"parameter": parameters}).count()):
				fe_id = int(self.db.seqs.find_and_modify(query={ 'collection' : 'featureExtraction' },update={'$inc': {'id': 1}},fields={'id': 1, '_id': 0},new=True).get('id'))
				self.db.FeatureExtraction.insert({"_id":fe_id,"executable":executable,"parameter": parameters})
		if task_name == 'feature_selection':
			if(0 == self.db.FeatureSelection.find({"executable":executable,"parameter": parameters}).count()):
				fs_id = int(self.db.seqs.find_and_modify(query={ 'collection' : 'featureSelection' },update={'$inc': {'id': 1}},fields={'id': 1, '_id': 0},new=True).get('id'))
				self.db.FeatureSelection.insert({"_id":fs_id,"executable":executable,"parameter": parameters})


	def getOptionsList(self,row):
        	if row[0] == 'single':
        	    return [(row[1], row[2])]

        	elif row[0] == 'collection':
			if row[1] == 'list':
				return list(map(lambda opt: (row[2], opt), row[3]))
			else:
				return list((row[2],a)for a in np.arange(row[3][0],row[3][1],row[3][2] if len(row[3])==3 else 1))


	def parseJobID(self, jobOutput):
		return jobOutput.split()[-1]

	def getAvailableNodes(self, jobList):
		return min(self.maxNodes, len(jobList))

	def getAvailableCpu(self, jobList):
		return min(self.maxCpu, len(jobList))

	def writeJobScript(self,datasplit,algorithm,metricCalculation,featureExtraction,featureSelection,preProcessing):
		job_files = defaultdict(list)
		outputFolder = datasplit['output_folder']
		log_folder = self.output_folder+"/logs"
		if not os.path.isdir(outputFolder):
			os.makedirs(outputFolder)
			os.makedirs(log_folder)
		split_file_name = outputFolder+"/Split_data_job_"
		algo_file_name = self.output_folder+"/Algo_job_"
		algo_file_name_deep = self.output_folder+"/Algo_job_deep"
		metric_file_name = self.output_folder+"/Metric_job_"
		result_file_name = outputFolder+"/Result_job.sh"
		no_of_split_jobs = int(math.ceil(len(self.job_list['datasplit'])/float(datasplit['no_of_files'])))
		no_of_algo_jobs = int(math.ceil(len(self.job_list['algorithm'])/float(algorithm['no_of_files'])))
		no_of_metric_jobs = int(math.ceil(len(self.metricJobs)/float(metricCalculation['no_of_files'])))
		pbs_config = "#!/bin/bash -l\n\
#SBATCH --job-name={transaction_name}\n\
#SBATCH -N {nodes} -n {nodes} --mem=20g --time=72:00:00\n\
#SBATCH --mail-user={email}\n\
#SBATCH --mail-type=FAIL\n\
#SBATCH --workdir={location}\n\
#SBATCH -e {transaction_name}.err\n\
#SBATCH -o {transaction_name}.out\n\
#SBATCH --array={job_array_size} \n\
\n\
module load scikit-learn/0.18.0-Python-2.7.12\n\
"
		pbs_config_deep = "#!/bin/bash -l\n\
#SBATCH -p gpu --gres=\"gpu:1\" \n\
#SBATCH --job-name={transaction_name}\n\
#SBATCH --mail-user={email}\n\
#SBATCH --mail-type=FAIL\n\
#SBATCH --workdir={location}\n\
#SBATCH -e {transaction_name}.err\n\
#SBATCH -o {transaction_name}.out\n\
#SBATCH --array={job_array_size} \n\
\n\
module load scikit-learn/0.18.0-Python-2.7.12\n\
module load caffe/rc3-Python-2.7.12-CUDA-7.5.18-cuDNN-4.0 \n\
module load TensorFlow/1.11-cp27-gpu \n\
\n\
"

		split_job_id = ''
		fe_job_id = ''
		fs_job_id = ''
		algo_job_id = ''
		mc_job_id = ''
		pp_job_id = ''

		if preProcessing != '':
			pre_processing_file_name = outputFolder+"/Pre_Processing.sh"
			job_file = open(pre_processing_file_name,'w')
			job_file.write(pbs_config.format(nodes=1,number_of_processors=1,email=self.email,location=self.workDir,transaction_name=log_folder+"/PreProceesing",job_array_size=str("0"))+"\n")
			for j in self.job_list['pre_processing']:
				job_file.write(j)
			job_file.close()
			proc = Popen("sbatch "+ pre_processing_file_name, shell=True, stdout=PIPE)
			pp_job_id =  self.parseJobID(proc.stdout.read())
			
		for i in range(datasplit['no_of_files']):
			f_name = split_file_name + str(i+1) + ".sh"
			job_files['datasplit_job_files'].append(f_name)
			job_file = open(f_name,'w')
			job_list = self.job_list['datasplit'][i*no_of_split_jobs : min((i*no_of_split_jobs)+no_of_split_jobs,len(self.job_list['datasplit']))]
			job_file.write(pbs_config.format(nodes=self.getAvailableNodes(job_list),number_of_processors=1,email=self.email,location=self.workDir,
			transaction_name=log_folder+"/SplitData",job_array_size=str("0-")+str(len(job_list)-1))+"\n")
			for j in job_list:
				job_file.write(j+"\n")
			job_file.close()
			if pp_job_id != '':
				proc = Popen("sbatch " + " --dependency=afterok:" + pp_job_id + " " + f_name, shell=True, stdout=PIPE)
			else:
				proc = Popen("sbatch "+ f_name, shell=True, stdout=PIPE)
			split_job_id =  self.parseJobID(proc.stdout.read())
			
			###################Uncomment below lines if running locally#####################################
			#command = 'sh ./'+f_name
			#arg = shlex.split(command)
			#process = subprocess.Popen(arg)
		#process.wait()

		if featureExtraction != '':
			feature_extraction_file_name = outputFolder+"/Feature_extraction_job_"
			no_of_fe_jobs = int(math.ceil(len(self.job_list['feature_extraction'])/float(featureExtraction['no_of_files'])))
			for i in range(featureExtraction['no_of_files']):
				f_name = feature_extraction_file_name + str(i+1) + ".sh"
				job_files['feature_extraction_job_files'].append(f_name)
				job_file = open(f_name,'w')
				job_list = self.job_list['feature_extraction'][i*no_of_fe_jobs : min((i*no_of_fe_jobs)+no_of_fe_jobs,len(self.job_list['feature_extraction']))]
				job_file.write(pbs_config.format(nodes=self.getAvailableNodes(job_list),number_of_processors=1,email=self.email,location=self.workDir,
				transaction_name=log_folder+"/FeatureExtraction",job_array_size=str("0-")+str(len(job_list)-1))+"\n")
				for j in job_list:
					job_file.write(j+"\n")
				job_file.close()
				proc = Popen("sbatch " + " --dependency=afterok:" + split_job_id + " " + f_name, shell=True, stdout=PIPE)
				
				fe_job_id =  self.parseJobID(proc.stdout.read())
				
				###################Uncomment below lines if running locally#####################################
				#command = 'sh ./'+f_name
				#arg = shlex.split(command)
				#process = subprocess.Popen(arg)
			#process.wait()
		if featureSelection != '':
			if fe_job_id == '':
				fe_job_id = split_job_id
			feature_selection_file_name = outputFolder+"/Feature_selection_job_"
			no_of_fs_jobs = int(math.ceil(len(self.job_list['feature_selection'])/float(featureSelection['no_of_files'])))
			for i in range(featureSelection['no_of_files']):
				f_name = feature_selection_file_name + str(i+1) + ".sh"
				job_files['feature_selection_job_files'].append(f_name)
				job_file = open(f_name,'w')
				job_list = self.job_list['feature_selection'][i*no_of_fs_jobs : min((i*no_of_fs_jobs)+no_of_fs_jobs,len(self.job_list['feature_selection']))]
				job_file.write(pbs_config.format(nodes=self.getAvailableNodes(job_list),number_of_processors=1,email=self.email,location=self.workDir,
				transaction_name=log_folder+"/FeatureSelection",job_array_size=str("0-")+str(len(job_list)-1))+"\n")
				for j in job_list:
					job_file.write(j+"\n")
				job_file.close()
				proc = Popen("sbatch " + " --dependency=afterok:" + fe_job_id  + " " + f_name , shell=True, stdout=PIPE)
				
				fs_job_id =  self.parseJobID(proc.stdout.read())
				
				###################Uncomment below lines if running locally#####################################
				#command = 'sh ./'+f_name
				#arg = shlex.split(command)
				#process = subprocess.Popen(arg)
			#process.wait()
		for i in range(algorithm['no_of_files']):
			if fs_job_id == '':
				if fe_job_id == '':
					fs_job_id = split_job_id
				else:
					fs_job_id = fe_job_id
			f_name = algo_file_name + str(i+1) + ".sh"
			f_name_deep = algo_file_name_deep + str(i+1) + ".sh"
			
			
			job_files['algo_job_files'].append(f_name)
			job_file = open(f_name,'w')
			job_file_deep = open(f_name_deep,'w')
			job_list = self.job_list['algorithm'][i*no_of_algo_jobs : min((i*no_of_algo_jobs)+no_of_algo_jobs,len(self.job_list['algorithm']))]
			job_file.write(pbs_config.format(nodes=self.getAvailableNodes(job_list),number_of_processors=1,email=self.email,location=self.workDir,
			transaction_name=log_folder+"/Algo",job_array_size=str("0-")+str(len(job_list)-1))+"\n")
			job_file_deep.write(pbs_config_deep.format(email=self.email,location=self.workDir,transaction_name=log_folder+"/Algo",job_array_size=1))
		
			run_deep = False
	
			for j in job_list:
				if j[7:11] == "Deep" or j[7:11] == "TFNe" :
					job_file_deep.write(j+"\n")
					run_deep = True
				else:
					job_file.write(j+"\n")
			job_file.close()
			job_file_deep.close()
			proc = Popen("sbatch " + " --dependency=afterok:" + fs_job_id  + " " + f_name, shell=True, stdout=PIPE)
			algo_job_id =  self.parseJobID(proc.stdout.read())
			######comment by xiaoli
			if run_deep:
				proc_deep = Popen("sbatch " + " --dependency=afterok:" + fs_job_id  + " " + f_name_deep, shell=True, stdout=PIPE)
				algo_job_deep_id =  self.parseJobID(proc_deep.stdout.read())
			
			###################Uncomment below lines if running locally#####################################
			#command = 'sh ./'+f_name
			#arg = shlex.split(command)
			#process = subprocess.Popen(arg)
		#process.wait()
		for i in range(metricCalculation['no_of_files']):
			f_name = metric_file_name + str(i+1) + ".sh"
			job_files['metric_job_files'].append(f_name)
			job_file = open(f_name,'w')
			job_list = self.metricJobs[i*no_of_metric_jobs : min((i*no_of_metric_jobs)+no_of_metric_jobs,len(self.metricJobs))]
			#job_file.write(pbs_config.format(nodes=self.getAvailableNodes(job_list),number_of_processors=1,email=self.email,location=self.workDir,
			#transaction_name=log_folder+"/Metric",job_array_size=str("0-")+str(len(job_list)-1))+"\n")
			job_file.write(pbs_config.format(nodes=self.getAvailableNodes(job_list),number_of_processors=1,email=self.email,location=self.workDir,
			transaction_name=log_folder+"/Metric",job_array_size=str("0-")+str(0))+"\n")
			for j in job_list:
				job_file.write(j+"\n")
			job_file.close()

			if run_deep:
				proc = Popen("sbatch " + " --dependency=afterok:" + algo_job_id +":"+ algo_job_deep_id + " " + f_name, shell=True, stdout=PIPE)
			else:
				proc = Popen("sbatch " + " --dependency=afterok:" + algo_job_id + " " + f_name, shell=True, stdout=PIPE)
			mc_job_id =  self.parseJobID(proc.stdout.read())
			###################Uncomment below lines if running locally#####################################
			#command = 'sh ./'+f_name
			#arg = shlex.split(command)
			#process = subprocess.Popen(arg)
		#process.wait()
		#yaml.dump(job_files,open("jobs_list.yaml",'w'))
		job_file = open(result_file_name,'w')
		job_file.write(pbs_config.format(nodes=1,number_of_processors=1,email=self.email,location=self.workDir,
		transaction_name=log_folder+"/Result",job_array_size=str("0-0"))+"\n")
		if self.resultFile=="":
                        job_file.write("python DatabaseResults.py -i " +("".join(datasplit['data_file'].split(".")[:-1])+"_"+str(os.getpid())+"."+datasplit['data_file'].split(".")[-1]))
                else:
                        job_file.write("python DatabaseResults.py -i " +("".join(datasplit['data_file'].split(".")[:-1])+"_"+str(os.getpid())+"."+datasplit['data_file'].split(".")[-1]) + " -o " +self.resultFile)
		job_file.close()
		proc = Popen("sbatch " + " --dependency=afterok:" + mc_job_id + " " + result_file_name , shell=True, stdout=PIPE)

	

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', help='WOLF job configuration file')
	args = parser.parse_args()

	wolf = WolfJob(args)
