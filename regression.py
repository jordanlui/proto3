# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 15:54:37 2017

@author: Jordan
July analysis of IR band results

Format of imported x matrix
[xcoord, ycoord, distance, sensor data]


"""
from __future__ import division
from  workspace_loader import load
from sklearn import datasets, linear_model
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

#%% Parameters
segment = 0.70 # Section of data that we train on
seed = 0
number_randomize = 2 # Number of times we want to random shuffle
seeds = range(0,number_randomize)
scale_table = 428/500 # Table scaling in pixels/mm. Note this calibration of active area square, but should be consistent across the entire camera frame.
table_width = 500 # Table width in mm (test table)

#%% Functions
def randomize_data(x,seed):
    # Randomizes the data based on a seed value
    random.seed(a=seed)
    x = np.asarray(random.sample(x,len(x)))
    return x
def split_xt(xin):
    # Split data into xmatrix and t columns
    t = xin[:,0:2] # First two columns. ignore distance for now
    x = xin[:,3:]
    return x,t
def segment_data(x,seg_index):
    # Segment data based on chosen train/test segmentation

    # t values based on distance calculation
    #t_train = np.reshape(x[:seg_index,5],(seg_index,1))
    #t_test =  np.reshape(x[seg_index:,5],(m-seg_index,1))

    # t value based on actual x y coord
    x,t = split_xt(x) # Split into the given columns x and t
    t_train = t[:seg_index,:]
    t_test = t[seg_index:,:]
    
    x_train = x[:seg_index,:]
    x_test =  x[seg_index:,:]
    return x_train, x_test, t_train, t_test
    
def prep_model(x,seg_index,seed):
    # Shuffle and split data as desired in a full mix scenario
    # Don't use this when doing LOO shuffle based on folders!

    x = randomize_data(x,seed) # Shuffle the data
    x_train, x_test, t_train, t_test = segment_data(x,seg_index) # Split into test and train
    return x_train, x_test, t_train, t_test
def model(x_train, x_test, t_train, t_test,seed):
    # Run the analysis
    # Inputs: x, segment, random seeds
    # Output: MSE Error Value, Variance score
#    seg_index = int(segment * len(x))
#    x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed) # Shuffle, segment data
    
    regr = linear_model.LinearRegression(normalize=True) # Build model
    regr.fit(x_train, t_train)  # Fit model
    
#    MSE = np.mean((regr.predict(x_test) - t_test) **2)
    
    # Do error as euclidean
    diff = np.sqrt(np.sum((regr.predict(x_test) - t_test) **2,axis=1)) # Euclidean error
    MSE = np.mean(diff) # Mean error (in pixels)
    variance = regr.score(x_test,t_test)
    return MSE, variance, diff
def model_multi(x_train,x_test,t_train,t_test,seed):
    # This function runs the model repeatedly based on number of random seeds and return the average MSE values and variances
    MSE = []
    variance = []
    for seed in seeds:
#        x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed)
        error, var, diff = model(x_train, x_test, t_train, t_test,seed)
        MSE.append(error)
        variance.append(var)

    MSE_mean = np.mean(MSE)
    variance_mean = np.mean(variance)
    # Coefficients
    #print('Coefficients: \n', regr.coef_)
    ## MSE
#    print 'Ran ',len(seeds),' randomizations'
#    print("MSE: %.2f" % MSE_mean)
#    print('Variance score: %.2f' %variance_mean)
    return MSE_mean, variance_mean
    
def LOOCV(path):
    # Runs cross validation routine
    # Accepts a list of files paths
    # Changed to output system error in mm

    error = []
    var = []
    
    for i in range(0,len(path)): # Iterate through the files
        x_train=[]
        x_test=[]
        
        single_path = path[i] # Single path
        rest_path = path[:i]+path[i+1:] # Rest of paths
        
        x_test = load(path=single_path) # Load into x matrix
        x_test,t_test = split_xt(x_test) # Split to x and t
    
        
        for apath in rest_path:
            xx_train = load(path=apath)
            x_train.append(xx_train)
      
        x_train = np.vstack(x_train)
        x_train,t_train = split_xt(x_train)

        
        # Run the model
        MSE_mean, variance_mean,diff = model(x_train,x_test,t_train,t_test,seed)
        error.append(MSE_mean/scale_table)
        var.append(variance_mean)
    return error,var


def singleRun(path):
    errors = [] # Error values in mm

    for path in path:
    #path = path[2]
        
        x = load(path)
        seg_index = int(segment*len(x))
        x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed)
        MSE, variance, diff = model(x_train,x_test,t_train,t_test,seed)
        
        # Histogram on error distribution
        diff_mm = diff/scale_table # This is the mm value error
        error_mean = np.mean(diff_mm)
        errors.append(error_mean)
    
        print 'Results: Mean error (mm)', error_mean, 'min error (mm) is',np.min(diff_mm),'max error',np.max(diff_mm),'median',np.median(diff_mm)
        
        # Plots
        plt.figure()
        plt.hist(diff_mm,bins='auto')
        plt.title('Histogram of error (mm)')
        plt.ylabel('Occurrences')
        plt.xlabel('Error value (mm)')
        plt.show()
        
        # 3D plot of error
        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(t_test[:,0], t_test[:,1], diff)
        
        plt.title('Error with position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Error')
    
    numfiles = len(path)
    print 'mean error across %d files was %f mm'%(numfiles,np.mean(errors))
    print 'Number of random shuffles:',len(seeds)
    print 'Number of files',numfiles
    return 0


#%% Results



#%% Prepare
#path = ['../Data/june23/1/','../Data/june23/2/','../Data/june23/3/','../Data/june23/4/','../Data/june23/5/','../Data/june23/6/','../Data/june23/7/']
path = ['../Data/july17/A - Neutral Hand Position/', '../Data/july17/B - Pronation, 45 Degrees/', '../Data/july17/C - Supination, 45 Degrees/']
allfiles = []
#apath=path[1]

for apath in path:

    allfiles =  allfiles + (glob.glob(apath+'*.csv'))
    
#%% Single Run Analysis
#singlefile = filelist[0]
#singleRun(filelist)
#test = load(singlefile)

#%% Model LOOCV
## Run model in LOOCV config
error,var = LOOCV(allfiles)


#%% LOOCV Result values

#print 'Variances: ', var
#print 'MSE:', error
print 'Mean error was %.3f mm, Error relative to table dimensions is %.3f ' % (np.mean(error), np.mean(error)/(table_width*scale_table))
