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

def segment_data(x,seg_index):
    # Segment data based on chosen train/test segmentation

    # t values based on distance calculation
    #t_train = np.reshape(x[:seg_index,5],(seg_index,1))
    #t_test =  np.reshape(x[seg_index:,5],(m-seg_index,1))

    # t value based on actual x y coord
    t_train = x[:seg_index,3:5]
    t_test = x[seg_index:,3:5]
    
    x_train = x[:seg_index,6:]
    x_test =  x[seg_index:,6:]
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
    
    MSE = np.mean((regr.predict(x_test) - t_test) **2)
    variance = regr.score(x_test,t_test)
    return MSE, variance
    
#def model2(x_train,x_test,t_train,t_test,segment,seed):
#    # Crude version where we duplicate model for simple input
#%% Prepare
path = ['../Data/june23/1/','../Data/june23/2/','../Data/june23/3/','../Data/june23/4/','../Data/june23/5/','../Data/june23/6/','../Data/june23/7/']
x = []
xx = []
for apath in path:
    x_temp, xx_temp = load(path = apath)
    x.append(x_temp)
    xx.append(xx_temp)

#x,xx = load(path = '../Data/june23/analysis/1415/') # Load Data
x = np.vstack(x)
m = len(x)
seg_index = int(segment * len(x))





#%% Run the model
MSE = []
variance = []
for seed in seeds:
    x_train, x_test, t_train, t_test = prep_model(x,seg_index,seed)
    error, var = model(x_train, x_test, t_train, t_test,segment,seed)
    MSE.append(error)
    variance.append(var)
    
# July 9 - Train on one recording and test on separate. Or at least separate files in same folder
    

#%% Do analysis on results

MSE_mean = np.mean(MSE)
variance_mean = np.mean(variance)
# Coefficients
#print('Coefficients: \n', regr.coef_)
## MSE
print 'Ran ',len(seeds),' randomizations'
print("MSE: %.2f" % MSE_mean)
print('Variance score: %.2f' %variance_mean)