# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 12:11:56 2017

@author: Jordan

Goal: Parse Text files of data from Proto device, read from bluetooth COM port

"""

from __future__ import division
import numpy as np
import json
import pandas
from saveCSV import *

path = "../data/sept29/"
filename = 'move forward back.csv'

def tryJSON(data):
	try:
		j = json.loads(data)
		return j
	except:
		print 'Invalid JSON'
		print data
		return False

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



