# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 12:21:18 2016

@author: Jordan
"""

# SVM analysis on results
# Data format
#dist1	dist2	omron8	omron8	omron8	omron8	omron8	omron8	omron8	omron8	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	omron16	fsr1	fsr2	fsr3	fsr4	fsr5	fsr6	orientation1	orientation2	orientation3


# Libraries
from __future__ import division
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
#from sklearn import neighbors, datasets
from sklearn import svm
from sklearn import preprocessing
import glob, os
import csv
import random
import re
import sys

# File paths for accessing data
path ='../Data/dec6_1'
output_path = '../Analysis/'
output_file = 'dec6.csv'

# Function parameters
cvalue = 2e-3

# Functions

def xmatrix(files):
    # Accepts a list of files and builds x matrix and t matrix
    x = []
    t = []
    # Check if we have a single file input
    if isinstance(files,str) == True:
        # Then just load the single file and output to x and t
        data = np.genfromtxt(files,delimiter=',')
        data = data[1:,:]
        trial_info = re.findall('\[([0-9]{1,2})\]',files)
        gesture = trial_info[1]
        for row in data:
            x.append(row)
            t.append(gesture)
    
    else:
        
        for file in files:
            # Load the data from csv file
            data = np.genfromtxt(file,delimiter=',')
            
            # Remove the header row
            data = data[1:,:]
            # Find the gesture number
            trial_info = re.findall('\[([0-9]{1,2})\]',file)
            gesture = trial_info[1]

            for row in data:
                x.append(row)
                t.append(gesture)
    # Reformat as arrays
    x = np.asarray(x)
    t = np.asarray(t)
    t = np.reshape(t,(len(t),1))

    return x,t

def normtraintest(train,test):
    # Normalizes a train and test matrix, in column by column fashion
    # Scales each column values from 0 to 1. 
    # Standardizes each column by subtracting mean and dividing by STDEV    
    # Currently you must put two inputs. Can fix this later.

    # Loop through our data, one column at a time
    for i in range(0,train.shape[1]):
        coltrain = train[:,i]
        coltest = test[:,i]
        colmax = np.max(coltrain)
        colmin = np.min(coltrain)
        mean = np.mean(coltrain)
        std = np.std(coltrain)
        
        # Standardize the data        
        coltrain = (coltrain - mean)/std
        coltest = (coltest - mean)/std        
        
        # Scale the data
        # For some reason the model runs even worse if we both standardize and scale data. figure this out later
#        coltrain = (coltrain - colmin)/(colmax - colmin)
#        coltest = (coltest - colmin)/(colmax - colmin)
        
        
        
        # Put the values back into the matrix
        
        train[:,i] = coltrain
        test[:,i] = coltest
        
    return train,test

# Get a list of files 
filelist = glob.glob(os.path.join(path,'*.csv'))
numfiles = len(filelist)

# Shuffle file list and remove some for testing. Seed the shuffle.
random.seed(1)
filelist = random.sample(filelist,len(filelist))

segment = 0.15 # Percentage of data that we test on
file_train = filelist[:numfiles-int(segment * numfiles)]
file_test = filelist[-int(segment * numfiles):]

# Generate x_train and x_test matrices
x_train,t_train = xmatrix(file_train)
x_test,t_test = xmatrix(file_test)

# Cut out two columns since we aren't using the FSR Data
#x_train = np.delete(x_train,[30,31],1)
#x_test = np.delete(x_test,[30,31],1)

# Data preprocessing
# Normalize the data, column-wise
x_train,x_test = normtraintest(x_train,x_test)


# Create SVM Model

lin_clf = svm.LinearSVC(C=cvalue)
lin_clf.fit(x_train,t_train)

# Overall Accuracy
testdata = lin_clf.predict(x_test)
testdata = np.reshape(testdata,(len(testdata),1)) # reshape the data

compare = testdata==t_test
numcorrect = np.sum(compare)
accuracy = numcorrect / len(testdata) * 100 
print 'overall accuracy is','{:04.2f}'.format(accuracy)

# Test Model Against each piece of testing data
# WIP - needs more work 
accuracy = []
for file in file_test:
    x1,t1 = xmatrix(file)
    
    # Temporary resizing of files. Remove
#    x1 = np.delete(x1,[30,31],1)

    testdata = lin_clf.predict(x1)
    testdata = np.reshape(testdata,(len(testdata),1)) # reshape the data

    # Compare to our test data
    
    compare = testdata==t1
    numcorrect = np.sum(compare)
    acc = numcorrect / len(testdata) * 100 
print 'summarize accuracy'
print accuracy
# Training Accuracy


# Testing Accuracy

# Data plots
plt.figure(1)
#plt.plot(x1[:,0:2],label='distance')
plt.plot(x1[:,-9:-4],label='fsr')
plt.legend()
plt.show()