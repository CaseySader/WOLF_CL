__author__ = 'Pranav Bahl'

import os
import pandas as pd
import errno
import arff
import yaml
import argparse
import sys, getopt
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder

class PreProcessing:

    def __init__(self, dataFile,outputFolder,missingValue,labelEncoding,parameters):
        self.dataFile = dataFile
        self.outputFolder = outputFolder
        self.missingValue = missingValue
        self.labelEncoding = labelEncoding
        self.parameters = parameters
        self.preprocess()

    def preprocess(self):
        if not os.path.exists(self.outputFolder):
    		try:
    			os.makedirs(self.outputFolder)
    		except OSError as exc:
    		    if exc.errno != errno.EEXIST:
    		        raise exc
    		    pass
        metadata = dict()
        if not self.parameters:
    		self.parameters['parameter']='default'
        metadata['preprocessing_params'] = self.parameters
        yaml.dump(metadata,open(self.outputFolder+'/PreProcessing.yaml','w'))
    	if self.dataFile.split('.')[-1] == 'arff':
    		data,meta = loadarff(self.dataFile)
    		data = pd.DataFrame(data)
    	else:
    		data = pd.read_csv(self.dataFile)

        data = data.fillna(self.missingValue)

        if self.labelEncoding:
            data = self.labelEncode(data)

        data.to_csv(self.outputFolder+'/DataFile.csv',index=False)

    def labelEncode(self,data):
        le = LabelEncoder()
        for i in data.columns:
            if data[i].dtype == "object":
                le.fit(data[i])
                data[i] = le.transform(data[i])
        return data

if __name__=='__main__':
    dataFile=''
    outputFolder=''
    parameters=dict()
    missingValue = 0
    labelEncoding = True

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', help='WOLF job configuration file')
        parser.add_argument('-o', help='WOLF job configuration file')
        parser.add_argument('-m', help='WOLF job configuration file')
        parser.add_argument('-l', help='WOLF job configuration file')
        parser.add_argument('-f', help='WOLF job configuration file')
        args = parser.parse_args()

        dataFile=args.i
        outputFolder=args.o
        if args.m:
            missingValue=int(args.m)
            parameters['parameter.m']=args.m
        if args.l:
            if args.l == 'True':
                labelEncoding=True
            elif args.l=='False':
                labelEncoding=False
            else:
                raise ValueError
            parameters['parameter.l']=args.l
    except:
        raise
        print 'PreProcessing.py -i <inputfile> -o <outputFolder> -m <missingValue> -l <labelEncoding>'
        sys.exit(2)

    PreProcessing(dataFile,outputFolder,missingValue,labelEncoding,parameters)
