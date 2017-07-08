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
seeds = range(0,25)

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
    
def model(x,segment,seed):
    # Run the analysis
    # Inputs: x, segment, random seeds
    # Output: MSE Error Value, Variance score
    x = randomize_data(x,seed)
    
    x_train, x_test, t_train, t_test = segment_data(x,seg_index)
    
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(x_train, t_train)
    
    MSE = np.mean((regr.predict(x_test) - t_test) **2)
    variance = regr.score(x_test,t_test)
    return MSE, variance
    

#%% Prepare
x,xx = load() # Load Data
m = len(x)
seg_index = int(segment * len(x))



#%% Run the model
MSE = []
variance = []
for seed in seeds:
    error, var = model(x,segment,seed)
    MSE.append(error)
    variance.append(var)
    


#%% Do analysis

MSE_mean = np.mean(MSE)
variance_mean = np.mean(variance)
# Coefficients
#print('Coefficients: \n', regr.coef_)
## MSE
print 'Ran ',len(seeds),' randomizations'
print("MSE: %.2f" % MSE_mean)
print('Variance score: %.2f' %variance_mean)