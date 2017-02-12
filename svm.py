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
from sklearn.metrics import confusion_matrix
import glob, os
import csv
import random
import re
import sys
from plot_confusion_matrix import plot_confusion_matrix

# File paths for accessing data
path ='../Data/proto3_combined/'
output_dir = '../Analysis/'
output_file = 'proto3_analysis.csv'
output_path = os.path.join(output_dir,output_file)
class_names = ['nominal flexion','affected flexion','upward','noise']
# Function parameters
cvalue = 2e-3

# Functions

def xmatrix(files):
    # Accepts a list of files, extracts each row, builds x matrix and t matrix
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
    # Otherwise we have multiple files and need to iterate through them.
    else:
        
        for file in files:
            # Load the data from csv file
            data = np.genfromtxt(file,delimiter=',')
            
            # Remove the header row
            data = data[1:,:]
            # Use RegEx to get the gesture number
            trial_info = re.findall('\[([0-9]{1,2})\]',file)
            gesture = int(trial_info[1])

            for row in data:
                x.append(row)
                t.append(gesture)
    # Reformat as arrays
    x = np.asarray(x)
    t = np.asarray(t)
    t = np.reshape(t,(len(t),1))

    return x,t

def normtraintest(train,test):
    
    # Normalizes a train and test matrix, in column by column fashion, 
    # in relation to the mean and stdev of the training data
    # Scales each column values from 0 to 1. 
    # Standardizes each column by subtracting mean and dividing by STDEV    
    # Currently you must put two inputs. Can fix this later.

    # Loop through our data, one column at a time

    for i in range(0,train.shape[1]):
        # Extract a column of data from train and test data
        coltrain = train[:,i]
        coltest = test[:,i]
        # Computer max, min, mean, stdev from the training data
        colmax = np.max(coltrain)
        colmin = np.min(coltrain)
        mean = np.mean(coltrain)
        std = np.std(coltrain)
        
        # Standardize training and test data with training data mean, stdev
        # Data should be mean shifted and scaled to so all columns have equal stdev
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
def exportresults(filename,data):
#    file_exists = os.path.isfile(filename)
    with open(filename,'ab') as csvfile:
        logwriter = csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow(data)
#        filednames = ['']
#        logwriter = csv.DictWriter(csvfile,fieldnames=filednames)
#        if not file_exists:
#            logwriter.writeheader()
#        logwriter.writerow()
def confusion_matrix_normalize(conf):
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    return conf
def model(seed,segment):
    # Model that performs our analysis and generates confusion matrix
    random.seed(seed)
# Read file list from directory
filelist = glob.glob(os.path.join(path,'*.csv'))
numfiles = len(filelist)

# This is where we would begin our loop
# Parameters 
# move up in code to improve structure later
seedrange = 15 # Number of random seeds we will try
segment = 0.20 # Percentage of data that we test on
# Shuffle file list and remove some for testing. Seed the shuffle.
#for seed in range(0,seedrange):
seed=2 # test
#model(seed,segment)

# Shuffle the filelist according to the seed
filelist = random.sample(filelist,len(filelist))

# Segment into training and testing list
file_train = filelist[:numfiles-int(segment * numfiles)]
file_test = filelist[-int(segment * numfiles):]

# Generate x_train and x_test matrices
x_train,t_train = xmatrix(file_train)
x_test,t_test = xmatrix(file_test)
matrix_names = ['x_train','t_train','x_test','t_test']

# Cut out two columns since we aren't using the FSR 5,6 Data
x_train = np.delete(x_train,[30,31],1)
x_test = np.delete(x_test,[30,31],1)
# Note the following code option also worked
#x_train = np.hstack((x_train[:,0:30],x_train[:,32:]))
#x_test = np.hstack((x_test[:,0:30],x_test[:,32:]))

# Data preprocessing
# Normalize the data, column-wise according to mean, stdev of training data
x_train,x_test = normtraintest(x_train,x_test)

# Create SVM Model

#lin_clf = svm.SVC(kernel='linear')
lin_clf = svm.SVC(kernel='rbf')
#lin_clf = svm.SVC(kernel='poly',degree=2)
lin_clf.fit(x_train,t_train)

# Overall Accuracy
testdata = lin_clf.predict(x_test)
testdata = np.reshape(testdata,(len(testdata),1)) # reshape the data

compare = testdata==t_test # Compare predictions to the actual test values
numcorrect = np.sum(compare)
accuracy = numcorrect / len(testdata) * 100 
print 'overall test accuracy is','{:04.2f}'.format(accuracy)

# Confusion matrix
conf = confusion_matrix(t_test,testdata)
confnorm = confusion_matrix_normalize(conf)
plot_confusion_matrix(conf,classes=class_names,normalize=True)

# Save results to file
# The data we will save
data = [str(datetime.datetime.now()),accuracy,numcorrect,confnorm[0,0],confnorm[1,1],confnorm[2,2],confnorm[3,3],segment,seed,int(x_train.shape[1])]
# Function to write them to a row in a csv
exportresults(output_path,data)

# Try Feature Selection
#from sklearn.svm import LinearSVC
#from sklearn.feature_selection import SelectFromModel
#x_train.shape
#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train,t_train)
#model = SelectFromModel(lsvc, prefit=True)
#x_new = model.transform(x_train)
#x_new.shape


# Plotting of Distance and IMU through movements
#x1,t1 = xmatrix(file_test[-1])
#plt.figure(2)
#plt.plot(x1[:,0:2],label='distance')
##plt.plot(x1[:,-9:-4],label='fsr') # FSR Data # Distance sensors
#plt.plot(x1[:,-3:],label='IMU') #IMU Data
#plt.legend()
#plt.show()