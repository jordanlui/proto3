# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 12:21:18 2016

@author: Jordan
"""

# SVM analysis on results

# Libraries
from __future__ import division
import scipy.io as sio
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn import neighbors, datasets
from sklearn import svm
import glob, os
import csv
import random
import re

# Function parameters
cvalue = 2e-3

# Functions
def xmatrix(files):
    # Accepts a list of files and builds x matrix and t matrix
    # For loop
    x = []
    t = []    
    for file in files:
        # Load the data from csv file
        data = np.genfromtxt(file,delimiter=',')
        
        # Remove the header row
        data = data[1:,:]
#        print data.shape
        
        # Get the label data, (gestures)
        trial_info = re.findall('\[([0-9]{1,2})\]',file)
    #    patient = trial_info[0]
        gesture = trial_info[1]
    #    trialnumber = trial_info[2]    
        
    #    x_train = np.concatenate((x_train,data),axis=0)
        for row in data:
            x.append(row)
            t.append(gesture)
        
    x = np.asarray(x)
    t = np.asarray(t)
    t = np.reshape(t,(len(t),1))

    return x,t

path ='../Data/dec6_1'
output_path = '../Analysis/'
output_file = 'dec6.csv'

# Get a list of files
filelist = glob.glob(os.path.join(path,'*.csv'))

numfiles = len(filelist)

# Shuffle it and remove some for testing. Seed the shuffle.
random.seed(1)
filelist = random.sample(filelist,len(filelist))

segment = 0.15 # Let's train on 10% of the data
file_train = filelist[:numfiles-int(segment * numfiles)]
file_test = filelist[-int(segment * numfiles):]

# Generate x_train and x_test matrices
x_train,t_train = xmatrix(file_train)
x_test,t_test = xmatrix(file_test)

# Feed this into SVM Model

lin_clf = svm.LinearSVC(C=cvalue)
lin_clf.fit(x_train,t_train)

# Test Model
testdata = lin_clf.predict(x_test)
testdata = np.reshape(testdata,(len(testdata),1)) # reshape the data

# Compare to our test data
compare = testdata==t_test
numcorrect = np.sum(compare)
accuracy = numcorrect / len(testdata) * 100 

# Training Accuracy


# Testing Accuracy

