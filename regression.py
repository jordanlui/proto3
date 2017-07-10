# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 15:54:37 2017

@author: Jordan
July analysis

Scratch pad for analysis
"""

from  workspace_loader import load
from sklearn import datasets, linear_model
import numpy as np
import random
import matplotlib.pyplot as plt

#%% Parameters
segment = 0.80 # Section of data that we train on
seed = 0
number_randomize = 5 # Number of times we want to random shuffle
seeds = range(0,number_randomize)

#%% Functions
def randomize_data(x,seed):
    # Randomizes the data based on a seed value
    random.seed(a=seed)
    x = np.asarray(random.sample(x,len(x)))
    return x
def split_xt(xin):
    # Split data into xmatrix and t columns
    t = xin[:,3:5]
    x = xin[:,6:]
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
def model(x_train, x_test, t_train, t_test,segment,seed):
    # Run the analysis
    # Inputs: x, segment, random seeds
    # Output: MSE Error Value, Variance score
#    seg_index = int(segment * len(x))
#    x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed) # Shuffle, segment data
    
    regr = linear_model.LinearRegression(normalize=True) # Build model
    regr.fit(x_train, t_train)  # Fit model
    
#    MSE = np.mean((regr.predict(x_test) - t_test) **2)
    
    # Do error as euclidean
    MSE = np.mean(np.sqrt(np.sum((regr.predict(x_test) - t_test) **2,axis=1)))
    variance = regr.score(x_test,t_test)
    return MSE, variance
def model_multi(x_train,x_test,t_train,t_test,seed):
    # This function runs the model repeatedly based on number of random seeds and return the average MSE values and variances
    MSE = []
    variance = []
    for seed in seeds:
#        x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed)
        error, var = model(x_train, x_test, t_train, t_test,segment,seed)
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
    
#%% Prepare
path = ['../Data/june23/1/','../Data/june23/2/','../Data/june23/3/','../Data/june23/4/','../Data/june23/5/','../Data/june23/6/','../Data/june23/7/']
#x = []
#xx = []
#for apath in path:
#    x_temp, xx_temp = load(path = apath)
#    x.append(x_temp)
#    xx.append(xx_temp)
#
##x,xx = load(path = '../Data/june23/analysis/1415/') # Load Data
#x = np.vstack(x)
#m = len(x)
#seg_index = int(segment * len(x))

#%% Practice LOO model
# LOO = Leave one out. Train on 6 folders while testing on the last folder. Iterate through combinations 


#t_train=[]
#t_test=[]
error = []
var = []

for i in range(0,len(path)):
    x_train=[]
    x_test=[]
    
    single_path = path[i] # Single path
    rest_path = path[:i]+path[i+1:] # Rest of paths
    
    x_test = load(path=single_path) # Load into x matrix
    x_test,t_test = split_xt(x_test[0]) # Split to x and t

    
    for apath in rest_path:
        xx_train = load(path=apath)
        x_train.append(xx_train[0])
  
    x_train = np.vstack(x_train)
    x_train,t_train = split_xt(x_train)
#    print 'i is',i
#    print 'given path is ' ,path[i]
#    print 'remainders are', path[:i]+path[i+1:]
#    print '\n'
    
    # Run the model
    MSE_mean, variance_mean = model_multi(x_train,x_test,t_train,t_test,seed)
    error.append(MSE_mean)
    var.append(variance_mean)
    
    
# Results
print 'Number of random shuffles:',len(seeds)
print 'Number of folder iterations',len(path)
print 'Variances: ', var
print 'MSE:', error

