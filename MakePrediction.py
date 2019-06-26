__author__ = 'Casey Sader'
import pickle
import os
import arff
import sys, getopt
import numpy as np
import pandas as pd
import pickle
import errno
from scipy.io.arff import loadarff
from sklearn.metrics import accuracy_score
import sklearn.ensemble
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from keras.models import load_model

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
# https://www.heatonresearch.com/2017/07/22/keras-getting-started.html
def to_xy(df):
	result = []
	for x in df.columns:
			result.append(x)
	return df.as_matrix(result)

def SavedModelPrediction(dataFile,outputfile,modelpath,label,metrics):

	try:
		dataframe = loadarff(dataFile)
	except:
		print 'Must provide a valid dataset in arff format'
		raise
		sys.exit(2)

	dataset = pd.DataFrame(dataframe[0])

	is_keras = False
	try:
		# pickled model
		loaded_model = pickle.load(open(modelpath, 'rb'))
	except:
		try:
			# Keras NN
			loaded_model = load_model(modelpath)
			is_keras = True
		except:
			print 'Must provide a valid trained model file'
			raise
			sys.exit(2)

	truth_array = np.empty
	if label is not None:
		try:
			truth_array = dataset['class'].values
			dataset = dataset.drop(['class'], axis=1)
			possible_preds = sorted(set(truth_array))
		except:
			print 'Problem with given label'
			raise

	if is_keras:
		dataset = to_xy(dataset)

	if not os.path.exists("predictions/"):
		try:
			os.makedirs("predictions/")
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise exc
			pass

	try:
		predictions = loaded_model.predict(dataset)

		if is_keras:
			# Need to convert class probabilities to labels e.g., [0.7, 0.3] -> 0
			predictions_keras = []
			for predindex in range(len(predictions)):
				pred_prob = max(predictions[predindex])
				label_index = np.where(predictions[predindex] == pred_prob)
				try:
					predictions_keras.append(possible_preds[label_index[0][0]])
				except:
					predictions_keras.append(label_index[0][0])
			# print predictions_keras
			np.savetxt("predictions/" + outputfile, predictions_keras, fmt='%s',delimiter=',')
		else:
			np.savetxt("predictions/" + outputfile, predictions, fmt='%s',delimiter=',')
	except:
		print 'Problem making predictions. Common problems could be label column being in dataset or dataset does not match the format from the trained set'
		raise
		sys.exit(2)

	if metrics and truth_array.size != 0:
		if is_keras:
			acc = accuracy_score(truth_array, predictions_keras)
		else:
			acc = accuracy_score(truth_array, predictions)
		print acc



def main(args):
	dataFile = ''
	modelpath = ''
	outputfile = 'predictions.csv'
	label = None
	metrics = False
	try:
		opts,args=getopt.getopt(args,"i:o:m:l:s:",[])
	except getopt.GetoptError:
		print 'MakePrediction.py -i <datafile> -o <outfile (optional)> -m <savedmodel> -l <label predicting (only needed if in dataset)> -s <boolean to calculate metrics (only possible if labeled data)>'
		raise
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-i':
			dataFile=arg
		elif opt=='-o':
			outputfile=arg
		elif opt=='-m':
			modelpath=arg
		elif opt=='-l':
			label=arg
		elif opt=='-s':
			if arg.lower() == 'true':
				metrics = True
			elif arg.lower() =='false':
				metrics = False
			else:
				raise ValueError

	SavedModelPrediction(dataFile,outputfile,modelpath,label,metrics)
if __name__ == "__main__":
	main(sys.argv[1:])