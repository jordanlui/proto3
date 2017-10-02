# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 12:11:56 2017

@author: Jordan

Goal: Parse Text files of data from Proto device, read from bluetooth COM port

"""

from __future__ import division
import numpy as np
import json
from saveCSV import WriteProtoCSV
import glob, os
import fileinput
import re

path = '../Data/sept29/'
filename = 'move forward back.csv'

# Find all files
#os.chdir(path)
filelist = []
for file in glob.glob(path+'*.txt'):
	if os.path.basename(file) == 'readme.txt':
		# Do nothing, don't import this file
		print 'readme.txt detected, skipped'
	else:
		filelist.append(file)


def tryJSON(data):
	try:
		j = json.loads(data)
		return j
	except:
		print 'Invalid JSON'
		print data
		return False

def csv2JSON(path,filename):
	# Load a CSV file of JSON strings, load as proper JSON objects and save to file
	
	data = []
	
	# Load data and parse JSON
	file = open(path+filename,"r")
	for line in file:
		rawline = line.strip('\n\r')
		rawline = rawline.replace('""','"')
		rawline = rawline[1:-1]
		j = tryJSON(rawline)
		data.append(j)
	
	# Now save to file, using a save CSV function
	WriteProtoCSV(data,path,filename)

# Main Loop

# RegEx Searching
# Regex Searching attempt
query = "{[\s\S]+?}"
# Reading from https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file-using-python


file = filelist[0]
for file in filelist: # Loop through files

	filename = os.path.basename(file) # This is the actual filename (excluding the path)
	f = open(file,"r")
	filedata = f.read()
	f.close()
	
	m = re.findall(query,filedata) # Find all strings matching the query
	mFormat = []
	for item in m:
		item = item.replace('\n','')
		item = tryJSON(item)
		mFormat.append(item)
	WriteProtoCSV(mFormat,path,filename[:-4])