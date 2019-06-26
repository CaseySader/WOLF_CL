from caffe import layers as L
from caffe import params as P

import numpy as np
import matplotlib.pyplot as plt


import os

import gc

import sys, getopt
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
import pandas as pd
import yaml
from scipy.io.arff import loadarff
import re


def solver_write(train_net_path, test_net_path,max_itr,learning_rate,snapshot):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and test networks.
    s.train_net = train_net_path
    s.test_net.append(test_net_path)

    s.test_interval = 2000  # Test after every 1000 training iterations.
    s.test_iter.append(300) # Test 250 "batches" each time we test.

    s.max_iter = max_itr      # # of times to update the net (training iterations)

    # Set the initial learning rate for stochastic gradient descent (SGD).
    s.base_lr = learning_rate        

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 9
    s.stepsize = 400

    # Set other optimization parameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 500

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- just once at the end of training.
    # For larger networks that take longer to train, you may want to set
    # snapshot < max_iter to save the network and training state to disk during
    # optimization, preventing disaster in case of machine crashes, etc.
    s.snapshot = snapshot
    s.snapshot_prefix = ''


    #for reproducable results
    s.random_seed=-1
    
    #Snap short format to h5
    s.snapshot_format=0
    
    # We'll train on the CPU for fair benchmarking against scikit-learn.
    # Changing to GPU should result in much faster training!
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    return s




def nonlinear_net(hdf5, batch_size,layers,deploy,act,input_dropout,hidden_dropout,L2,filler):
		n = caffe.NetSpec()
		data, label = L.HDF5Data(source=hdf5,batch_size=batch_size,ntop=2)
		n.data=data
		n.label=label
		#Add hidden layers
		n.top = n.data
		if(input_dropout!=0):
			n.top = L.Dropout(n.top, in_place=True, dropout_ratio = input_dropout)
		
		test = 0
		for x in range(0,len(layers)):
			if(L2):
				if(filler==1):
					n.top = L.InnerProduct(n.top, num_output=layers[x], weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'),param=[dict(decay_mult=1)])
				elif(filler==2):
					n.top = L.InnerProduct(n.top, num_output=layers[x], weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type='gaussian',std=0.01),param=[dict(decay_mult=1)])

			else:
				if(filler==1):
					n.top = L.InnerProduct(n.top, num_output=layers[x], weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'),param=[dict(decay_mult=0)])
				elif(filler==2):
					n.top = L.InnerProduct(n.top, num_output=layers[x], weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type='gaussian',std=0.01),param=[dict(decay_mult=0)])

	
			if(act == 1):
				n.top = L.ReLU(n.top,in_place=True)
			elif(act == 2):
				n.top = L.Sigmoid(n.top, in_place=True)
			elif(act == 3):
				n.top = L.TanH(n.top, in_place=True)
			else:
				print "Error, invalid activation function choice "
			if(hidden_dropout!=0):
				n.top = L.Dropout(n.top, in_place=True, dropout_ratio = hidden_dropout)
	
		#Add Output Layers
		if(filler==1):
			n.output = L.InnerProduct(n.top, num_output=2,weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'))
		elif(filler==2):
			n.output = L.InnerProduct(n.top, num_output=2,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type='gaussian',std=0.01))

		if(deploy == False):
			n.loss = L.SoftmaxWithLoss(n.output,n.label)
			n.accuracy=L.Accuracy(n.output,n.label)
				
		else:
			n.prob = L.Softmax(n.output)
			
    		return n.to_proto()
    
def main(args):
	
	#network parameters
	ActivationFunction=1   # 1 = ReLU, 2 = sigmoid, 3 = TanH
	Filler=1		# 1 = Xavier, 2 = Gaussian
	InputDropout= 0		# Percentage of nuerons to dropout during training phase (input layer) Value btw 0 and 1
	HiddenDropout=	0	# Percentage of nuerons to dropout during training phase (hidden layers) Value btw 0 and 1, Recommended 0.5
	LearningRate = 0.001
	layers=[100,100,100]
	L2Leguralization='false'
	
	Max_iter = 200
	batch_size = 100
	
	SnapShot=Max_iter
	
	datafile=''
	outputfolder=''
	parameters=dict()
	try:
		opts,args=getopt.getopt(args,"i:o:a:l:f:d:h:e:r:b:g",[])
	except getopt.GetoptError:
		print 'datasplit.py -i <inputfile> -o <outputfolder> -a <Activation Function> -l <Layers> -f<Filler> -d<Input Dropout> -h<HiddenDropout> -e<epochs> -b<batch size> -r<learning rate>'
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
		elif opt=='-f':
			Filler=int(arg)
			parameters['parameter.f']=arg
        	elif opt == "-d":
            		InputDropout=float(arg)
            		parameters['parameter.d']=arg
        	elif opt=="-h":
            		HiddenDropout=float(arg)
            		parameters['parameter.h']=arg
        	elif opt=="-e":
            		Max_iter=int(arg)
            		parameters['parameter.e']=arg   
            	elif opt=="-r":
            		LearningRate=float(arg)
            		parameters['parameter.r']=arg 	
            	elif opt=="-b":
            		batch_size=int(arg)
            		parameters['parameter.b']=arg	
            	elif opt=="-g":
            		print "Avoiding typecast error for last element"
  		     
	homefolder=os.getcwd()
	if not datafile.startswith('/'):
		datafile=homefolder+'/' + datafile
	if not outputfolder.startswith('/'):
		outputfolder=homefolder+'/' + outputfolder
	if not os.path.isdir(outputfolder+'/tmp'):
			os.makedirs(outputfolder+'/tmp')
	os.chdir(outputfolder+'/tmp')
	
	inputData = yaml.load(open(datafile))	
	trainingSet = inputData['training']
	testingSet = inputData['testing']
	#k = yaml.load(open(datafile))['k']
	#r = yaml.load(open(datafile))['r']
	inputFile = inputData['inputFile']
	label = inputData['label']
	resultset = []		
	
	for i in range(len(trainingSet)):
		with open("train.txt", "w") as text_file:
			if not trainingSet[i].startswith('/'):
	    			text_file.write(homefolder+'/'+trainingSet[i].split('.')[0]+'.h5')
			else:
				text_file.write(trainingSet[i].split('.')[0]+'.h5')
	    	with open("test.txt", "w") as text_file:
			if not testingSet[i].startswith('/'):
	    			text_file.write(homefolder+'/'+testingSet[i].split('.')[0]+'.h5')
			else:
				text_file.write(testingSet[i].split('.')[0]+'.h5')	
		train_net_path = 'nonlinear_auto_train.prototxt'
		with open(train_net_path, 'w') as f:
	    		f.write(str(nonlinear_net('train.txt', batch_size,layers,False,ActivationFunction,InputDropout,HiddenDropout,L2Leguralization,Filler)))

		test_net_path = 'nonlinear_auto_test.prototxt'
		with open(test_net_path, 'w') as f:
		    	f.write(str(nonlinear_net('test.txt', batch_size,layers,False,ActivationFunction,InputDropout,0,0,Filler)))
	
	
	
	
		    	
		solver_path = 'nonlinear_logreg_solver.prototxt'
		with open(solver_path, 'w') as f:
	    		f.write(str(solver_write(train_net_path, test_net_path,Max_iter,LearningRate,SnapShot)))
	    
		caffe.set_mode_gpu()
		solver = caffe.SGDSolver(solver_path)
		solver.solve()
	
		accuracy = 0
		batch_size = solver.test_nets[0].blobs['data'].num
		test_iters = int(384 / batch_size)
		for k in range(test_iters):
	    		solver.test_nets[0].forward()
	    		accuracy += solver.test_nets[0].blobs['accuracy'].data
	    		
		accuracy /= test_iters
		if not testingSet[i].startswith('/'):
			test_df = pd.read_csv(homefolder+'/'+testingSet[i])
		else:
			test_df = pd.read_csv(testingSet[i])
		test_predictions = pd.DataFrame(test_df[label])
	
		test_length=len(test_predictions)
	
		deploy_net_path = 'nonlinear_auto_deploy.prototxt'
		with open(deploy_net_path, 'w') as f:
		    	f.write(str(nonlinear_net('test.txt', test_length,layers,True,ActivationFunction,InputDropout,0,0,Filler)))
		net = caffe.Net(deploy_net_path, '_iter_'+str(Max_iter)+'.caffemodel.h5', caffe.TEST) 
	
		net.forward()
		
	
	
		predict_prob=net.blobs['prob'].data
		predict_label=[]
		for j in predict_prob:
			predict_label.append(j.argmax())
		#print predict_label
		resultFile = outputfolder+'/result'+str(i+1)+'.csv'
		test_predictions['predictions']=predict_label
		test_predictions.to_csv(resultFile,index=False)
		resultset.append(resultFile)
		print("Accuracy: {:.3f}".format(accuracy))
	resultDict = dict()
	
	resultDict['results'] = resultset
	resultDict['label'] = label
	"""parameters['parameter.t'] = str(numberOfTrees)
	parameters['parameter.d'] = str(depth)"""
	
	if not parameters:
		parameters['parameter']='default'
	resultDict['algo_params'] = parameters
	resultDict['split_params'] = inputData['split_params']


	resultDict['inputFile'] = inputFile
	resultDict['algorithm'] = "DeepLearning"
	yaml.dump(resultDict,open(outputfolder+'/results.yaml','w'))
	
				
	
	
	
if __name__ == "__main__":
   main(sys.argv[1:])
