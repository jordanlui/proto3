# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 12:11:56 2017

@author: Jordan

Goal: Parse Text files of data from Proto device, read from bluetooth COM port
Text file format is the JSON style formatting output of proto device, 
as copied directly from the Sublime IDE. RegEx filtering to check for intact 
JSON strings and remove line returns


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
	# This was used on data from Proto device that is properly formatted (no stray line returns)
	
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
query = "{[\s\S]+?}"

file = filelist[0]
for file in filelist: # Loop through files

	filename = os.path.basename(file) # This is the actual filename (excluding the path)
	f = open(file,"r") # Read the file into an object
	filedata = f.read()
	f.close()
	
	m = re.findall(query,filedata) # Find all strings matching the query using RegEx. Note this can read across lines
	mFormat = []
	for item in m:
		item = item.replace('\n','') # Remove the stray and random line returns
		item = tryJSON(item) # Load as JSON object
		mFormat.append(item) # Add into new array
	WriteProtoCSV(mFormat,path,filename[:-4]) # Write to CSV