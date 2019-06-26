__author__ = 'Pranav Bahl'

import yaml
import sys, getopt
import argparse
import subprocess,shlex
import os
import time

class Run:

	def __init__(self,args):
		self.datasplit_job_files = yaml.load(open(args.i))['datasplit_job_files']
		self.algo_job_files = yaml.load(open(args.i))['algo_job_files']
		self.metric_job_files = yaml.load(open(args.i))['metric_job_files']
		self.execute_jobs()
	
	def execute_jobs(self):
		process = []
		for i in self.datasplit_job_files:
			command = 'sh ./'+i
			arg = shlex.split(command)
			process = subprocess.Popen(arg)
		process.wait()
		for i in self.algo_job_files:
			command = 'sh ./'+i
			arg = shlex.split(command)
			process = subprocess.Popen(arg)
		process.wait()
		for i in self.metric_job_files:
			command = 'sh ./'+i
			arg = shlex.split(command)
			process = subprocess.Popen(arg)
		process.wait()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', help='Job List file')
	args = parser.parse_args()
	#print "verifying check in"
	run = Run(args)
