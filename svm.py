    # -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 12:21:18 2016

@author: Jordan
SVM Code for analysis of IR Band research
Jordan Lui 2017, MENRVA Lab, SFU

June 2017:
PCA analysis of different sensor groups:
0 No FSR
1 Omron 8 close
2 Omron 16 close
3 Omron 8 Far
4 Omron 16 Far
5 Orientation RPY
6 IR Distance
7 Accelerometer
8 Omron 16-1, IR1&2, IMU
9 Omron 16-1, 16-2, IR1234, IMU

June / July 2017 data
Now we have x and y coord data from webcam
Format: # [patient, class, trial, x-coord, y-coord, DistIR, RPY, Omron Sensors]

"""
# Libraries
from __future__ import division
import numpy as np
#from sklearn import neighbors, datasets
from sklearn import svm
#from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import os, csv, random
import operator
import datetime

# Import our custom functions
from plot_confusion_matrix import plot_confusion_matrix
#from xmatrix import xmatrix

# File paths for accessing data
path ='../Analysis/Investigation Apr 2017/data/' # Input file path
output_dir = '../Analysis/'
output_file = 'proto4_analysis.csv' # File where we summarize run results
output_path = os.path.join(output_dir,output_file)
# Class names
class_names  = [15,20,25,30,35,40,45,50,55,60,'L1','L2','L3','R1','R2','R3','U1','U2','U3','D1','D2','D3']

# Names from the 9 static arm position test are below (Feb 2017)
#class_names = ['fwd 40','fwd 25','fwd 10','left 40','left 25','left 10','up 40','up 25','up 10']


# Declare globals here if we remove the main loop
global ac, ac2, index, value, seed, accuracy, conf, confnorm, filelist
global x_train, x_test, nancount

# Run Parameters
#cvalue = 2e-3 # Pretty sure this isn't even referenced, at least for RBF kernel
C = 1000
gamma = 0.0001
seedrange = 2 # Number of random seeds we will try
segment = 0.80 # Percentage of data that we train on. Train on 0.8 means test on 0.2.
plotbool=0 # Flag for plotting on or off
seed = 1
singlerun = 0 # Flag for signaling that we are doing a single randomized evaluation. 1 is a single run.
nancount = 0

# Functions

def normtraintest(train,test):
    
    # Normalizes a train and test matrix, in column by column fashion, 
    # in relation to the mean and stdev of the training data
    # Scales each column values from 0 to 1. 
    # Standardizes each column by subtracting mean and dividing by STDEV    
    # Currently you must put two inputs. Can fix this later.

    # Loop through our data, one column at a time
    print 'shape of train is',train.shape
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
                       
        # Put the values back into the matrix
        
        train[:,i] = coltrain
        test[:,i] = coltest
        
    return train,test
def exportresults(filename,data):
#    file_exists = os.path.isfile(filename)
    with open(filename,'ab') as csvfile:
        logwriter = csv.writer(csvfile,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        logwriter.writerow(data)

def confusion_matrix_normalize(conf):
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    return conf
def model(seed,segment,plotbool,x):
    # Model that performs our analysis and generates confusion matrix
    # Currently accepts augmented x matrix which also contains the labels
    # Latest test (May 2017) have 67 columns of data and 3 columns for labels
    # Get our global variables
    global filelist, numfiles, accuracy, conf, confnorm, x_train, x_test, nancount
    
    # New method
    # Load data into xmatrix

    # Find index at which we segment our data    
    segindex = int(len(x)*segment)

    # Segment our data    
    # Grab X values. 
    x_train = x[:segindex,3:]
    x_test = x[segindex:,3:]
    # Segment T values 
    t_train = x[:segindex,1]
    t_test = x[segindex:,1]
    # Reshape t to proper array
    t_train = np.reshape(t_train,(len(t_train),1))
    t_test = np.reshape(t_test,(len(t_test),1))
    
    
    #    matrix_names = ['x_train','t_train','x_test','t_test']
    
    # Choose the data features we examine
    # Currently we ignore the 6 FSR values as well as raw accelerometer and gyro data.
#    x_train = np.delete(x_train,range(4,16),1)
#    x_test = np.delete(x_test,range(4,16),1)
    
    # Normalize our data
    normmean = np.mean(x_train,axis=0)
    normstdev = np.std(x_train,axis=0)
    x_test = (x_test - normmean) / normstdev
    x_train = (x_train - normmean) / normstdev
    # x_test = (x_test - np.mean(x_train,axis=0)) / (np.max(x_train,axis=0) - np.mean(x_train,axis=0))
    # x_train = (x_train - np.mean(x_train,axis=0)) / (np.max(x_train,axis=0) - np.mean(x_train,axis=0))
    
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
#    print 'overall test accuracy is','{:04.2f}'.format(accuracy)
    
    # Create the Confusion matrix
    conf = confusion_matrix(t_test,testdata)
#    confnorm = confusion_matrix_normalize(conf)
    if plotbool==1: 
        confnorm = plot_confusion_matrix(conf,classes=class_names,normalize=True)
    else:
        confnorm = confusion_matrix_normalize(conf)
    # Save results to file
    # The data we will save
    outputdata=str(datetime.datetime.now()),accuracy,numcorrect,confnorm[0,0],confnorm[1,1],confnorm[2,2],confnorm[3,3],segment,seed,int(x_train.shape[1])
#    data = [str(datetime.datetime.now()),accuracy,numcorrect,confnorm[0,0],confnorm[1,1],confnorm[2,2],confnorm[3,3],segment,seed,int(x_train.shape[1])]
    data=outputdata
    # Function to write them to a row in a csv
    exportresults(output_path,data)
    return accuracy,confnorm

# End of Functions
# Begin main loop
#def main():

# This is where we would begin our loop
# Parameters 
# move up in code to improve structure later

#    Loop through several seed values and see our highest and average accuracies
#global ac, ac2, index, value, seed, accuracy, conf, confnorm, filelist
#   Old Code that is used to loop through different segment values and find the best combination. Not needed   
#    for segment in range(1,10):
#        segment = segment/10

# Version of code for looping through various seed values    
x = np.genfromtxt(os.path.join(path,"xx.csv"),delimiter=',')
xbackup = np.asarray(list(x))
print 'initial size of x is' , x.shape
# Masking and deletion of different columns for PCA

# Make empty list that we'll store all masks in
masks = []
# Anything True is kept. False will be deleted

# Basic mask to remove FSR data
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(7,13)]] = False # Remove FSR
masks.append(mask)

# Close 8x1 Omron only
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(3,22)]] = False
mask[30:]  = False
masks.append(mask)

# Close 16 Omron Only
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(3,30)]] = False
mask[46:]  = False
masks.append(mask)

# Far 8x1 Omron Only
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(3,46)]] = False
mask[54:]  = False
masks.append(mask)

# Far 16 Omron Only
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(3,54)]] = False
masks.append(mask)

# Orientation Only
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(3,19)]] = False
mask[22:] = False
masks.append(mask)

# IR sensors only
mask = np.ones(x.shape[1], dtype=bool)
mask[7:] = False
masks.append(mask)

# Accelerometer
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(3,13)]] = False
mask[16:] = False
masks.append(mask)

# Omron 16-2, IR Long1, Short1, IMU
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(5,19)]] = False
mask[[range(22,54)]] = False
masks.append(mask)

# Omron 16-1, 16-2, IR (all), IMU
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(7,19)]] = False
mask[[range(22,30)]] = False
mask[[range(46,54)]] = False
masks.append(mask)

# Omron 16-1 and IMU (no Active IR)
mask = np.ones(x.shape[1], dtype=bool)
mask[[range(3,19)]] = False
mask[[range(22,54)]] = False
masks.append(mask)


print 'resize x to', x.shape

# PCA Analysis loop
acc_pca = []
for singlemask in masks[-1:]: # Can decide which specific masks to run
    x = np.asarray(list(xbackup))
    x = x[:,singlemask]
    print x.shape

    #Segment variation
    #segmentrange = np.arange(0.001,0.01,0.001)
    #for segment in segmentrange:
    if singlerun == 1:
        # Execute a single run
        # Single run Version of code
    #    random.seed(a=seed)
    #    filelist = random.sample(filelist,numfiles)
        print 'seed is',seed
    #    x = xmatrix(filelist)  
        # Shuffle data
        random.seed(a=seed)
        x = np.asarray(random.sample(x,len(x)))
        accuracy, confnorm = model(seed,segment,plotbool,x)
        acc_pca.append(accuracy)
        
    #        output_string = 'In %d randomizations, %d from seed %d has highest overall accuracy. Mean %d, min%d, stdev %d Individual accuracies for this seed are %s. Highest individuals are %s. Mean is %s Segment %.2f' %(seedrange,value,index,ac_mean,ac_min,np.std(ac),str(ac2[index,:]),str(ac2_max),str(ac2_mean),segment)
        output_string = 'PCA. Accuracy %.2f. Seed %s. Data from %s' %(accuracy,str(seed),path)
        print output_string
    else:
        # Execute a looped run through many seed values
        ac = []
        ac2 = [] 
        
        
        for seed in range(0,seedrange+1):
    
            #        random.seed(a=seed)
            #        filelist = random.sample(filelist,numfiles)
            # Shuffle data
            random.seed(a=seed)
            x = np.asarray(random.sample(x,len(x)))
            accuracy, confnorm = model(seed,segment,plotbool,x)
            ac.append(accuracy)
            ac2.append(confnorm.diagonal())
            print 'Seed %d of %d, acc=%.2f'%(seed,seedrange,accuracy)
        
        # Stat Analysis of our results
        # Calculate main max, mean, min accuracy levels
        ac_max = np.nanmax(ac)
        ac_mean = np.nanmean(ac)
        ac_min = np.nanmin(ac)
        # Append this mean accuracy to a list for analysis after
        acc_pca.append(ac_mean)
        # ac2 objects are the diagonals of the confusion matrix
        ac2 = np.asarray(ac2)
        ac2_max = np.nanmax(ac2,axis=0)
        ac2_mean = np.nanmean(ac2,axis=0)
        ac2_min = np.nanmin(ac2,axis=0)
        
        # Give a meaningful result summary
        # Finx the trial that had the highest overall accuracy        
        index, value = max(enumerate(ac),key=operator.itemgetter(1))
        # Generate a statement to summarize our trials
        output_string = 'PCA. Data from %s. In %d randomizations, %d from seed %d has highest overall accuracy. Mean %d, min%d, stdev %d Individual accuracies for this seed are %s. Highest individuals are %s. Mean is %s Segment %.2f' %(path, seedrange,value,index,ac_mean,ac_min,np.std(ac),str(ac2[index,:]),str(ac2_max),str(ac2_mean),segment)
        print output_string
    
    # Print to a logfile
    text_file = open(os.path.join(output_dir,"log.txt"), "a")
    text_file.write(str(datetime.datetime.now())+" "+output_string+"\n")
    text_file.close()
    
    