# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:22:19 2016

Creates x matrix from CSV files.

Updating July 2017 
Note that motion tracking device as of June 2017 is labeled by the x-y coordinates, 
not the patient/class/trial info in the csv files.


@author: Jordan
"""
# Read all of the CSV files in the directory and extract the min, max, and average value per trial on two axis and outputs

# Libraries
from __future__ import division
import numpy as np
#from sklearn import neighbors, datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import glob, os, csv, random, re, sys
import operator
import datetime

# Import our custom functions
from plot_confusion_matrix import plot_confusion_matrix
from xmatrix import xmatrix

# File paths for accessing data

path ='../Data/june23/'

output_dir = '../Data/june23/analysis/'
output_file = 'analysis.csv'
output_path = os.path.join(output_dir,output_file)

#global ac, ac2, index, value, seed, accuracy, conf, confnorm, filelist
#global x_train, x_test, nancount

# Initial variables
#nancount = 0

# Functions
def genfilelist(path):
    # generates array of all csv files in the directory
    filelist = []    
    if isinstance(path,(str,unicode))  == True:
        # Then we hav a single path to generate files
        filelist = glob.glob(os.path.join(path,'*.csv'))
    else:
        # we should have a list or tuple of folder paths
        for path in paths:
            filelist.extend(glob.glob(os.path.join(path,'*.csv')))
    return filelist

# Main code
# Generate File list

filelist = genfilelist(path)#   Generate file list
numfiles = len(filelist)    #   Number of files

# Generate xmatrix with all files in filelist
x,patient,gesture,trial,nancount  = xmatrix(filelist)  

# Note that June / July 2017 data outputs have xy coordinates at end of table. For consistency, want them at front.
data_sensors = x[:,0:55]
data_coord = x[:,55:]
newx = np.hstack((data_coord,data_sensors))
x = newx
# Make the big x matrix
xx = np.hstack((patient,gesture,trial,x))

# Save to CSV File so we can use it later
np.savetxt(output_dir+"x.csv",x,delimiter=",")
np.savetxt(output_dir+"xx.csv",xx,delimiter=",")
np.savetxt(output_dir+"patient.csv",patient,delimiter=",")
np.savetxt(output_dir+"gesture.csv",gesture,delimiter=",")
np.savetxt(output_dir+"trial.csv",trial,delimiter=",")


