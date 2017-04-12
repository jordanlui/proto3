# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:22:19 2016

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
#path ='../Data/proto3_combined/'
path ='../Data/proto4/1'
paths = ['../Data/proto4/1','../Data/proto4/2','../Data/proto4/3','../Data/proto4/4','../Data/proto4/5',]
output_dir = '../Analysis/'
output_file = 'proto4_analysis.csv'
output_path = os.path.join(output_dir,output_file)

global ac, ac2, index, value, seed, accuracy, conf, confnorm, filelist
global x_train, x_test, nancount

# Initial variables
#nancount = 0

# Functions
def genfilelist(path):
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

filelist = genfilelist(paths)
numfiles = len(filelist)

# Import the file
x,nancount  = xmatrix(filelist)  
