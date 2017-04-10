    # -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 12:21:18 2016

@author: Jordan
"""
# Libraries
from __future__ import division
import numpy as np
#from sklearn import neighbors, datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import glob, os, csv, random, re, sys
from plot_confusion_matrix import plot_confusion_matrix
import operator
import datetime

# File paths for accessing data
#path ='../Data/proto3_combined/'
path ='../Data/proto4/1'
#paths = ['../Data/proto4/1','../Data/proto4/2','../Data/proto4/3','../Data/proto4/4','../Data/proto4/5',]
#path = '../Data/armruler_feb24'
output_dir = '../Analysis/'
output_file = 'proto4_analysis.csv'
output_path = os.path.join(output_dir,output_file)
# Class names
class_names  = [60,55,50,45,40,35,30,25,20]
#class_names = np.asarray(range(1,23))
# Names from the 9 static arm position test are below
#class_names = ['fwd 1.0','fwd 0.5','fwd 0','left 1.0','left 0.5','left 0','up 1.0','up 0.5','up 0']
# Names from the old dynamic movement classification test
#class_names = ['nominal flexion','affected flexion','upward','noise']

# Declare globals here if we remove the main loop
global ac, ac2, index, value, seed, accuracy, conf, confnorm, filelist
global x_train, x_test, nancount

# Run Parameters
cvalue = 2e-3
seedrange = 100 # Number of random seeds we will try
segment = 0.30 # Percentage of data that we test on
plotbool=1 # Flag for plotting on or off
seed = 1
singlerun = 1 # Flag for signaling that we are doing a single randomized evaluation. 1 is a single run.
nancount = 0
# Read file list from directory
filelist = glob.glob(os.path.join(path,'*.csv'))
numfiles = len(filelist)



# Functions

def xmatrix(files):
    # Accepts a list of files, extracts each row, builds x matrix and t matrix
    x = []
    t = []
    global nancount
    # Check if we have a single file input
    if isinstance(files,str) == True:
        # Then just load the single file and output to x and t
        data = np.genfromtxt(files,delimiter=',')
        data = data[1:,:] # Only consider data from row 1 downwards (due to headers)
        trial_info = re.findall('\[([0-9]{1,2})\]',files)
        gesture = int(trial_info[1])
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
                if np.isnan(row).any() == True:
                    print "nan found in file",file
                    nancount = nancount + 1
                else:
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
def model(seed,segment,plotbool):
    # Model that performs our analysis and generates confusion matrix
    # Get our global variables
    global filelist, numfiles, accuracy, conf, confnorm, x_train, x_test, nancount
#    random.seed(a=seed)
    # Shuffle the filelist according to the seed
    
    
    # Segment into training and testing list
    # Probably want to update and simplify the method for the data segmentation
#    file_train = filelist[:numfiles-int(segment * numfiles)]
#    file_test = filelist[-int(segment * numfiles):]
#    
#    # Generate x_train and x_test matrices
#    # This (archaic) method will read through the files and build out into arrays. Try to improve with a bulk load and shuffle
#    x_train,t_train = xmatrix(file_train)
#    x_test,t_test = xmatrix(file_test)
    
    # New method
    # Load data into xmatrix
    x,t = xmatrix(filelist)  
    # Find index at which we segment our data    
    testchunk = int(len(x)*segment)
    # Segment our data    
    x_train = x[:testchunk,:]
    x_test = x[testchunk:,:]
    t_train = t[:testchunk,:]
    t_test = t[testchunk:,:]
    
#    matrix_names = ['x_train','t_train','x_test','t_test']
    
    # Choose the data features we examine
    # Cut out two columns since we aren't using the FSR 5,6 Data
    x_train = np.delete(x_train,range(4,16),1)
    x_test = np.delete(x_test,range(4,16),1)

    # Data preprocessing
    # Normalize the data, column-wise according to mean, stdev of training data
    print "shape of train is", x_train.shape
    print "shape of test is", x_test.shape
    print "train max is %d and min is %d. Mean %d" %(np.max(x_train), np.min(x_train), np.mean(x_train))
    print 'number of nan found in train is', len(np.argwhere(np.isnan(x_train)))
    print 'number of nan found in train is', len(np.argwhere(np.isnan(x_test)))
    
    x_train,x_test = normtraintest(x_train,x_test)
    print "After we normalize, train max is %d and min is %d" %(np.max(x_train), np.min(x_train))
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
if singlerun == 1:
    # Execute a single run
    # Single run Version of code
    random.seed(a=seed)
    filelist = random.sample(filelist,numfiles)
    print 'seed is',seed
    accuracy, confnorm = model(seed,segment,plotbool)
    
#        output_string = 'In %d randomizations, %d from seed %d has highest overall accuracy. Mean %d, min%d, stdev %d Individual accuracies for this seed are %s. Highest individuals are %s. Mean is %s Segment %.2f' %(seedrange,value,index,ac_mean,ac_min,np.std(ac),str(ac2[index,:]),str(ac2_max),str(ac2_mean),segment)
    output_string = 'Accuracy %.2f. Seed %s. Data from %s' %(accuracy,str(seed),path)
    print output_string
else:
    # Execute a looped run through many seed values
    ac = []
    ac2 = []  
    for seed in range(0,seedrange+1):

        random.seed(a=seed)
        filelist = random.sample(filelist,numfiles)
        accuracy, confnorm = model(seed,segment,plotbool)
        ac.append(accuracy)
        ac2.append(confnorm.diagonal())
        print 'Seed %d of %d, acc=%.2f'%(seed,seedrange,accuracy)
    
    # Stat Analysis of our results
    ac_max = np.nanmax(ac)
    ac_mean = np.nanmean(ac)
    ac_min = np.nanmin(ac)
    ac2 = np.asarray(ac2)
    ac2_max = np.nanmax(ac2,axis=0)
    ac2_mean = np.nanmean(ac2,axis=0)
    ac2_min = np.nanmin(ac2,axis=0)
    
    # Give a meaningful result summary
    # Finx the trial that had the highest overall accuracy        
    index, value = max(enumerate(ac),key=operator.itemgetter(1))
    # Generate a statement to summarize our trials
    output_string = 'Data from %s. In %d randomizations, %d from seed %d has highest overall accuracy. Mean %d, min%d, stdev %d Individual accuracies for this seed are %s. Highest individuals are %s. Mean is %s Segment %.2f' %(path, seedrange,value,index,ac_mean,ac_min,np.std(ac),str(ac2[index,:]),str(ac2_max),str(ac2_mean),segment)
    print output_string

# Print to a logfile
text_file = open(os.path.join(output_dir,"log.txt"), "a")
text_file.write(str(datetime.datetime.now())+" "+output_string+"\n")
text_file.close()
    

    
#if __name__ == '__main__':
#    main()